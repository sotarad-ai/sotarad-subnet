"""
SotaRad Subnet Validator
========================

What is measured
----------------
Each miner's model is scored with **Fβ** (default β = 2) on **public** evaluation
studies whose acquisition timestamps are strictly after the model's on-chain
submission time plus a configurable delay.

Miner interface (Chutes / OpenAI-compatible vision endpoint)
----------------------------------------------------------
Miners deploy a vision-language model to Chutes and commit JSON to chain
(`repo`, `revision`, `chute_id`, optional `params`). Validators call:

    POST {CHUTES_LLM_URL}/chat/completions

using `prompts/system_prompt.py` (`SYSTEM_PROMPT`, `build_user_message`,
`build_chutes_messages`) — same shape as `tests/test_model_request.py`.

The model must return a **top-level JSON array** of finding objects (may be `[]`).

Scoring V0 — flag only (see docs/ARCHITECTURE.md §4.4.3)
-------------------------------------------------------
- **Predicted positive** iff the parsed array has **len > 0**.
- **Predicted negative** iff the parsed array is **[]**.
- **Unparseable** reply → sample **skipped** for that miner (like a transport error).
- **Ground truth positive** iff any `positive_finding` has
  `condition ∈ TARGET_CONDITIONS` and `status == "active"`.

TP / FP / FN / TN and Fβ are computed from these study-level predictions.

Default **TARGET_CONDITIONS** match `prompts/system_prompt.TARGET_CONDITIONS`
(four lung conditions). Override with env `TARGET_CONDITIONS` or `--target-conditions`.

Tier incentives, duplicate policy, and dataset API contract are unchanged; see
ARCHITECTURE.md and inline code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import aiohttp
import bittensor as bt
import click
from bittensor_wallet import Wallet

from local_sglang import SglangSubprocessServer
from prompts.response_parse import parse_findings_json_array
from prompts.system_prompt import (
    TARGET_CONDITIONS as PROMPT_TARGET_CONDITIONS,
    build_chutes_messages,
)


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Target condition config ───────────────────────────────────────────────────
# Conditions (as in report_findings[].condition) that count as screen-positive
# when status == "active". Default matches prompts/system_prompt.py.
def _parse_target_conditions(raw: str) -> frozenset[str]:
    if not raw.strip():
        return frozenset(PROMPT_TARGET_CONDITIONS)
    return frozenset(c.strip() for c in raw.split(",") if c.strip())


# ── Tier configuration ────────────────────────────────────────────────────────

@dataclass
class TierConfig:
    """Defines one emission tier."""
    name: str
    lookback_days: int
    top_n: Optional[int]       # e.g. 1 for "top-1 miner"; mutually exclusive with top_pct
    top_pct: Optional[float]   # e.g. 0.10 for "top 10 %"; mutually exclusive with top_n
    emission_share: float      # fraction of total weight budget, e.g. 0.95


DEFAULT_TIERS: list[TierConfig] = [
    TierConfig("A", lookback_days=5, top_n=1,    top_pct=None, emission_share=0.95),
    TierConfig("B", lookback_days=4, top_n=None, top_pct=0.10, emission_share=0.02),
    TierConfig("C", lookback_days=3, top_n=None, top_pct=0.20, emission_share=0.015),
    TierConfig("D", lookback_days=2, top_n=None, top_pct=0.30, emission_share=0.010),
    TierConfig("E", lookback_days=1, top_n=None, top_pct=0.40, emission_share=0.005),
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MinerCommit:
    uid: int
    hotkey: str
    repo: str
    revision: str
    chute_id: str
    params: int           # 0 if not provided by the miner
    commit_block: int
    commit_ts: float      # unix timestamp approximated from commit_block

    @property
    def duplicate_key(self) -> tuple[str, str]:
        """Two commits are duplicates iff they share (repo, revision)."""
        return (self.repo.lower().strip(), self.revision.lower().strip())


@dataclass
class EvalSample:
    sample_id: str
    image_url: str
    label: int        # 1 = screen-positive (ground truth), 0 = negative
    timestamp: float  # unix timestamp of this sample
    patient_demographics: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    uid: int
    eval_date: str    # "YYYY-MM-DD"
    tp: int
    fp: int
    fn: int
    tn: int
    sample_count: int
    fb_score: float
    precision_score: float
    recall_score: float
    chute_id: str
    revision: str


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS daily_scores (
                uid              INTEGER NOT NULL,
                eval_date        TEXT    NOT NULL,
                fb_score         REAL    NOT NULL,
                precision_score  REAL    NOT NULL,
                recall_score     REAL    NOT NULL,
                tp               INTEGER NOT NULL,
                fp               INTEGER NOT NULL,
                fn               INTEGER NOT NULL,
                tn               INTEGER NOT NULL,
                sample_count     INTEGER NOT NULL,
                chute_id         TEXT    NOT NULL,
                revision         TEXT    NOT NULL,
                created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (uid, eval_date)
            )
        """)
        con.commit()


def upsert_daily_score(db_path: str, result: EvalResult) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute("""
            INSERT INTO daily_scores
                (uid, eval_date, fb_score, precision_score, recall_score,
                 tp, fp, fn, tn, sample_count, chute_id, revision)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(uid, eval_date) DO UPDATE SET
                fb_score        = excluded.fb_score,
                precision_score = excluded.precision_score,
                recall_score    = excluded.recall_score,
                tp              = excluded.tp,
                fp              = excluded.fp,
                fn              = excluded.fn,
                tn              = excluded.tn,
                sample_count    = excluded.sample_count,
                chute_id        = excluded.chute_id,
                revision        = excluded.revision
        """, (
            result.uid, result.eval_date, result.fb_score,
            result.precision_score, result.recall_score,
            result.tp, result.fp, result.fn, result.tn,
            result.sample_count, result.chute_id, result.revision,
        ))
        con.commit()


def query_scores_for_uid(
    db_path: str,
    uid: int,
    since_date: str,
    until_date: str,
) -> list[dict]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT * FROM daily_scores
            WHERE uid = ? AND eval_date >= ? AND eval_date <= ?
            ORDER BY eval_date DESC
        """, (uid, since_date, until_date)).fetchall()
    return [dict(r) for r in rows]


def eval_already_ran_today(db_path: str, eval_date: str) -> bool:
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT COUNT(*) FROM daily_scores WHERE eval_date = ?",
            (eval_date,),
        ).fetchone()
    return row[0] > 0


# ── Chain helpers ─────────────────────────────────────────────────────────────

def _get_commit_block(
    subtensor: bt.Subtensor,
    netuid: int,
    hotkey: str,
    fallback: int,
) -> int:
    """
    Query Commitments.CommitmentOf to get the block at which a miner committed.
    Falls back to the provided value (e.g. metagraph.last_update[uid]) on any error.
    """
    try:
        result = subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
        )
        if result is not None and result.value is not None:
            val = result.value
            if isinstance(val, dict):
                return int(val.get("block", fallback))
    except Exception as exc:
        logger.debug(f"_get_commit_block({hotkey[:8]}): {exc}")
    return fallback


def _block_to_timestamp(
    commit_block: int,
    current_block: int,
    block_time_s: float = 12.0,
) -> float:
    """Approximate unix timestamp of a historical block."""
    blocks_ago = max(0, current_block - commit_block)
    return time.time() - blocks_ago * block_time_s


def fetch_all_commits(
    subtensor: bt.Subtensor,
    metagraph: bt.Metagraph,
    netuid: int,
    current_block: int,
    *,
    allow_local: bool = False,
) -> list[MinerCommit]:
    """
    Read every miner's on-chain commitment, parse the JSON payload, and return
    a list of MinerCommit objects.  UIDs with missing or malformed commitments
    are silently skipped.

    If ``allow_local`` is False, ``chute_id`` must be non-empty (Chutes-only).
    If True, empty ``chute_id`` is allowed and the validator may load the HF
    ``repo`` + ``revision`` via local SGLang instead.
    """
    try:
        raw: dict = subtensor.get_all_commitments(netuid)
    except Exception as exc:
        logger.warning(f"get_all_commitments failed ({exc}); falling back to per-UID queries")
        raw = {}
        for uid in range(int(metagraph.n)):
            try:
                data = subtensor.get_commitment(netuid, uid)
                if data:
                    raw[uid] = data
            except Exception:
                pass

    commits: list[MinerCommit] = []
    for uid_key, data_str in raw.items():
        try:
            uid = int(uid_key)
        except (ValueError, TypeError):
            continue

        if uid >= len(metagraph.hotkeys):
            continue

        hotkey = metagraph.hotkeys[uid]

        if isinstance(data_str, (bytes, bytearray)):
            data_str = data_str.decode("utf-8", errors="replace")
        data_str = str(data_str).strip()
        if not data_str:
            continue

        try:
            payload = json.loads(data_str)
            repo     = str(payload["repo"]).strip()
            revision = str(payload["revision"]).strip()
            chute_id = str(payload.get("chute_id") or "").strip()
            params   = int(payload.get("params", 0))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.debug(f"UID {uid}: invalid commitment JSON – {exc}")
            continue

        if not repo or not revision:
            logger.debug(f"UID {uid}: commitment missing repo or revision")
            continue

        if not chute_id and not allow_local:
            logger.debug(f"UID {uid}: missing chute_id (use --allow-local for HF-only commits)")
            continue

        try:
            fallback_block = int(metagraph.last_update[uid])
        except Exception:
            fallback_block = current_block

        commit_block = _get_commit_block(subtensor, netuid, hotkey, fallback_block)
        commit_ts    = _block_to_timestamp(commit_block, current_block)

        commits.append(MinerCommit(
            uid=uid,
            hotkey=hotkey,
            repo=repo,
            revision=revision,
            chute_id=chute_id,
            params=params,
            commit_block=commit_block,
            commit_ts=commit_ts,
        ))

    logger.info(f"Parsed {len(commits)} valid commitments from chain")
    return commits


# ── Duplicate detection ───────────────────────────────────────────────────────

def deduplicate_commits(commits: list[MinerCommit]) -> list[MinerCommit]:
    """
    For each (repo, revision) group keep only the commit with the earliest
    commit_block.  Tie-break: lower UID wins.  All other duplicates are
    logged and excluded.
    """
    groups: dict[tuple[str, str], list[MinerCommit]] = {}
    for c in commits:
        groups.setdefault(c.duplicate_key, []).append(c)

    eligible: list[MinerCommit] = []
    for _key, group in groups.items():
        winner = min(group, key=lambda c: (c.commit_block, c.uid))
        eligible.append(winner)
        for c in group:
            if c.uid != winner.uid:
                logger.info(
                    f"UID {c.uid} ({c.repo}@{c.revision[:8]}) is a duplicate of "
                    f"UID {winner.uid} (earliest submitter) – excluded from evaluation"
                )
    return eligible


# ── Temporal eligibility filter ───────────────────────────────────────────────

def filter_eligible(
    commits: list[MinerCommit],
    eval_delay_days: int,
) -> list[MinerCommit]:
    """
    Keep only commits whose submission timestamp is at least eval_delay_days
    before now (i.e. the model could not have been trained on today's data).
    First eligible evaluation day is D+1 per ARCHITECTURE.md §2.3.
    """
    cutoff_ts = time.time() - eval_delay_days * 86_400
    eligible = [c for c in commits if c.commit_ts < cutoff_ts]
    excluded = len(commits) - len(eligible)
    if excluded:
        logger.info(f"{excluded} miner(s) excluded: committed too recently (< {eval_delay_days}d ago)")
    return eligible


# ── Dataset client ────────────────────────────────────────────────────────────

def _acquisition_date_to_ts(date_str: str) -> float:
    """Convert "YYYY-MM-DD" acquisition date to a UTC unix timestamp (start of day)."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _is_screen_positive(
    positive_findings: list[dict],
    target_conditions: frozenset[str],
) -> int:
    """
    Return 1 if any finding matches a target condition with status "active",
    else 0.  Case-sensitive on condition name; status comparison is exact.
    """
    for finding in positive_findings:
        if (
            finding.get("condition") in target_conditions
            and finding.get("status") == "active"
        ):
            return 1
    return 0


def _resolve_image_url(image_field: str, image_base_url: str) -> str:
    """
    Return a usable image URL.
    If image_field is already a full URL, return it as-is.
    Otherwise join image_base_url + "/" + image_field.
    """
    if image_field.startswith("http://") or image_field.startswith("https://"):
        return image_field
    base = image_base_url.rstrip("/")
    return f"{base}/{image_field}"


async def fetch_eval_samples(
    session: aiohttp.ClientSession,
    dataset_base_url: str,
    dataset_api_key: str,
    image_base_url: str,
    target_conditions: frozenset[str],
    after_ts: float,
    before_ts: float,
    limit: int = 100,
) -> list[EvalSample]:
    """
    Fetch labelled evaluation studies from the configured public dataset API.

    API contract:
        GET {dataset_base_url}/studies?after=<YYYY-MM-DD>&before=<YYYY-MM-DD>&limit=<n>
        → { "studies": [ { "study_id", "image_file", "acquisition_date",
                            "report_findings": { "positive_findings": [...] } } ] }

    Label is derived from positive_findings: 1 if any finding has
    condition ∈ target_conditions AND status == "active", else 0.
    """
    if not dataset_base_url:
        logger.warning("DATASET_BASE_URL not configured – skipping evaluation")
        return []

    # Convert unix timestamps to ISO date strings for the API query
    after_date  = datetime.utcfromtimestamp(after_ts).strftime("%Y-%m-%d")
    before_date = datetime.utcfromtimestamp(before_ts).strftime("%Y-%m-%d")

    url     = f"{dataset_base_url.rstrip('/')}/studies"
    params  = {"after": after_date, "before": before_date, "limit": limit}
    headers = {}
    if dataset_api_key:
        headers["Authorization"] = f"Bearer {dataset_api_key}"

    try:
        async with session.get(
            url, params=params, headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logger.error(f"Dataset API error: {exc}")
        return []

    samples: list[EvalSample] = []
    for study in data.get("studies", []):
        try:
            study_id         = str(study["study_id"])
            image_field      = str(study.get("image_url") or study["image_file"])
            acq_date         = str(study["acquisition_date"])
            positive_findings = study.get("report_findings", {}).get("positive_findings", [])

            image_url = _resolve_image_url(image_field, image_base_url)
            label       = _is_screen_positive(positive_findings, target_conditions)
            timestamp   = _acquisition_date_to_ts(acq_date)
            demographics = study.get("patient_demographics") or {}
            if not isinstance(demographics, dict):
                demographics = {}

            samples.append(EvalSample(
                sample_id=study_id,
                image_url=image_url,
                label=label,
                timestamp=timestamp,
                patient_demographics=demographics,
            ))
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug(f"Skipping malformed study entry: {exc}")
            continue

    pos = sum(s.label for s in samples)
    logger.info(
        f"Dataset: {len(samples)} studies parsed | "
        f"{pos} positive / {len(samples) - pos} negative"
    )
    return samples


# ── Vision inference (Chutes or local SGLang OpenAI-compatible) ─────────────


def _chat_completions_url(base_url: str) -> str:
    """Chutes base often ends with ``/v1``; SGLang root is host:port only."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        return f"{b}/chat/completions"
    return f"{b}/v1/chat/completions"


async def query_vision_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    bearer_token: str,
    model: str,
    image_url: str,
    patient_demographics: dict,
    *,
    merge_system_into_user: bool,
    max_tokens: int,
    timeout_s: int = 60,
    log_label: str = "",
) -> Optional[int]:
    """
    POST /v1/chat/completions; parse JSON findings array (same rules as tests).

    V0 scoring: 1 if len(parsed array) > 0, 0 if []. None if HTTP/parse failure.
    ``bearer_token`` may be empty for local servers without auth.
    """
    payload = {
        "model": model,
        "messages": build_chutes_messages(
            image_url,
            patient_demographics,
            merge_system_into_user=merge_system_into_user,
        ),
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    url     = _chat_completions_url(base_url)
    headers: dict[str, str] = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    try:
        async with session.post(
            url, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            logger.debug("%s: non-string message content", log_label or model)
            return None
        findings = parse_findings_json_array(content)
        if findings is None:
            logger.debug("%s: could not parse JSON array from reply", log_label or model)
            return None
        return 1 if len(findings) > 0 else 0
    except Exception as exc:
        logger.debug("%s: inference error – %s", log_label or model, exc)
        return None


async def query_chute(
    session: aiohttp.ClientSession,
    chutes_llm_url: str,
    chutes_api_key: str,
    chute_id: str,
    image_url: str,
    patient_demographics: dict,
    *,
    merge_system_into_user: bool,
    max_tokens: int,
    timeout_s: int = 60,
) -> Optional[int]:
    """Chutes-hosted model (requires API key)."""
    return await query_vision_completion(
        session,
        chutes_llm_url,
        chutes_api_key,
        chute_id,
        image_url,
        patient_demographics,
        merge_system_into_user=merge_system_into_user,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        log_label=f"Chute {chute_id[:16]}",
    )


# ── Fβ scoring ────────────────────────────────────────────────────────────────

def fbeta_score(precision: float, recall: float, beta: float) -> float:
    """Standard Fβ formula.  Returns 0.0 when the denominator is zero."""
    b2    = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def compute_metrics(
    tp: int, fp: int, fn: int, beta: float
) -> tuple[float, float, float]:
    """Returns (precision, recall, fb_score) with safe zero-denominator handling."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fb        = fbeta_score(precision, recall, beta)
    return precision, recall, fb


# ── Evaluator ─────────────────────────────────────────────────────────────────

async def evaluate_miner(
    session: aiohttp.ClientSession,
    commit: MinerCommit,
    samples: list[EvalSample],
    chutes_llm_url: str,
    chutes_api_key: str,
    chutes_timeout: int,
    chutes_max_tokens: int,
    chutes_merge_system_into_user: bool,
    beta: float,
    eval_date: str,
    *,
    allow_local: bool,
    local_sglang_host: str,
    local_sglang_port: int,
    sglang_extra_argv: list[str],
    sglang_startup_timeout: float,
) -> Optional[EvalResult]:
    """
    Run inference for one miner across the sample batch and compute Fβ (V0 flag-only).
    Returns None if all inference calls failed or were unparseable.

    With a non-empty ``chute_id``, calls Chutes. Otherwise (HF-only commit) starts
    a local SGLang server when ``allow_local`` is True.
    """
    use_chutes = bool(commit.chute_id.strip())
    if not use_chutes:
        if not allow_local:
            logger.warning("UID %s: empty chute_id but allow_local is off", commit.uid)
            return None
        server = SglangSubprocessServer(
            commit.repo,
            commit.revision,
            local_sglang_host,
            local_sglang_port,
            extra_argv=sglang_extra_argv,
            startup_timeout_s=sglang_startup_timeout,
        )
        await asyncio.to_thread(server.start)
        try:
            if not await server.wait_until_ready(session):
                return None
            tasks = [
                query_vision_completion(
                    session,
                    server.client_base_url,
                    "",
                    commit.repo,
                    s.image_url,
                    s.patient_demographics,
                    merge_system_into_user=chutes_merge_system_into_user,
                    max_tokens=chutes_max_tokens,
                    timeout_s=chutes_timeout,
                    log_label=f"local {commit.repo}",
                )
                for s in samples
            ]
            predictions = await asyncio.gather(*tasks)
        finally:
            await asyncio.to_thread(server.stop)
    else:
        tasks = [
            query_chute(
                session,
                chutes_llm_url,
                chutes_api_key,
                commit.chute_id,
                s.image_url,
                s.patient_demographics,
                merge_system_into_user=chutes_merge_system_into_user,
                max_tokens=chutes_max_tokens,
                timeout_s=chutes_timeout,
            )
            for s in samples
        ]
        predictions = await asyncio.gather(*tasks)

    tp = fp = fn = tn = evaluated = 0
    for sample, pred in zip(samples, predictions):
        if pred is None:
            continue
        evaluated += 1
        actual = sample.label
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1
        else:
            tn += 1

    id_short = (commit.chute_id.strip() or f"local:{commit.repo}")[:32]
    if evaluated == 0:
        logger.warning("UID %s (%s): all inference calls failed", commit.uid, id_short)
        return None

    precision, recall, fb = compute_metrics(tp, fp, fn, beta)
    logger.info(
        f"UID {commit.uid} | {commit.repo}@{commit.revision[:8]} | "
        f"Fβ={fb:.4f}  P={precision:.4f}  R={recall:.4f}  "
        f"TP={tp}  FP={fp}  FN={fn}  TN={tn}  n={evaluated}"
    )
    stored_chute_id = commit.chute_id.strip() or f"local:{commit.repo}"
    return EvalResult(
        uid=commit.uid,
        eval_date=eval_date,
        tp=tp, fp=fp, fn=fn, tn=tn,
        sample_count=evaluated,
        fb_score=fb,
        precision_score=precision,
        recall_score=recall,
        chute_id=stored_chute_id,
        revision=commit.revision,
    )


async def run_daily_evaluation(
    subtensor: bt.Subtensor,
    metagraph: bt.Metagraph,
    netuid: int,
    current_block: int,
    db_path: str,
    dataset_base_url: str,
    dataset_api_key: str,
    image_base_url: str,
    target_conditions: frozenset[str],
    chutes_llm_url: str,
    chutes_api_key: str,
    chutes_timeout: int,
    chutes_max_tokens: int,
    chutes_merge_system_into_user: bool,
    chutes_max_concurrent: int,
    eval_samples_per_day: int,
    eval_delay_days: int,
    beta: float,
    *,
    allow_local: bool,
    local_sglang_host: str,
    local_sglang_port: int,
    sglang_extra_argv: list[str],
    sglang_startup_timeout: float,
) -> list[MinerCommit]:
    """
    One complete daily evaluation pass.  Returns the list of eligible commits
    so the caller can cache them for weight-setting between evaluations.
    """
    today = date.today().isoformat()
    logger.info(f"=== Daily evaluation starting for {today} ===")

    # 1. Discover all miner commitments
    commits = await asyncio.to_thread(
        fetch_all_commits,
        subtensor,
        metagraph,
        netuid,
        current_block,
        allow_local=allow_local,
    )

    # 2. Dedup: only the earliest submitter per (repo, revision) is eligible
    eligible = deduplicate_commits(commits)

    # 3. Temporal filter: submission must predate today by at least eval_delay_days
    eligible = filter_eligible(eligible, eval_delay_days)
    logger.info(f"{len(eligible)} miner(s) eligible for today's evaluation")

    if not eligible:
        logger.info("No eligible miners – skipping inference pass")
        return eligible

    # 4. Fetch labelled evaluation samples from the public dataset API
    #    Window: last 24 h (today's window); each miner also filtered by its own cutoff below
    now           = time.time()
    eval_end_ts   = now
    eval_start_ts = now - 86_400

    async with aiohttp.ClientSession() as session:
        samples = await fetch_eval_samples(
            session, dataset_base_url, dataset_api_key,
            image_base_url=image_base_url,
            target_conditions=target_conditions,
            after_ts=eval_start_ts,
            before_ts=eval_end_ts,
            limit=eval_samples_per_day,
        )

        if not samples:
            logger.warning("No evaluation samples returned – aborting evaluation pass")
            return eligible

        logger.info(f"Fetched {len(samples)} evaluation samples from dataset API")

        # 5. Evaluate each eligible miner (bounded concurrency via semaphore;
        #    local SGLang is serialized — one subprocess / GPU at a time)
        sem = asyncio.Semaphore(chutes_max_concurrent)
        local_slot = asyncio.Semaphore(1)

        async def _eval_one(commit: MinerCommit) -> Optional[EvalResult]:
            miner_cutoff = commit.commit_ts + eval_delay_days * 86_400
            miner_samples = [s for s in samples if s.timestamp > miner_cutoff]
            if not miner_samples:
                logger.debug(
                    f"UID {commit.uid}: no samples pass temporal filter "
                    f"(need ts > {miner_cutoff:.0f})"
                )
                return None
            is_local_hf = allow_local and not commit.chute_id.strip()
            if is_local_hf:
                async with local_slot:
                    return await evaluate_miner(
                        session,
                        commit,
                        miner_samples,
                        chutes_llm_url,
                        chutes_api_key,
                        chutes_timeout,
                        chutes_max_tokens,
                        chutes_merge_system_into_user,
                        beta,
                        today,
                        allow_local=allow_local,
                        local_sglang_host=local_sglang_host,
                        local_sglang_port=local_sglang_port,
                        sglang_extra_argv=sglang_extra_argv,
                        sglang_startup_timeout=sglang_startup_timeout,
                    )
            async with sem:
                return await evaluate_miner(
                    session,
                    commit,
                    miner_samples,
                    chutes_llm_url,
                    chutes_api_key,
                    chutes_timeout,
                    chutes_max_tokens,
                    chutes_merge_system_into_user,
                    beta,
                    today,
                    allow_local=allow_local,
                    local_sglang_host=local_sglang_host,
                    local_sglang_port=local_sglang_port,
                    sglang_extra_argv=sglang_extra_argv,
                    sglang_startup_timeout=sglang_startup_timeout,
                )

        results = await asyncio.gather(*[_eval_one(c) for c in eligible])

    # 6. Persist non-None results to SQLite
    saved = 0
    for r in results:
        if r is not None:
            await asyncio.to_thread(upsert_daily_score, db_path, r)
            saved += 1

    logger.info(f"=== Daily evaluation complete: {saved}/{len(eligible)} miners scored ===")
    return eligible


# ── Tier engine ───────────────────────────────────────────────────────────────

def _lookback_date_range(today: str, lookback_days: int) -> tuple[str, str]:
    """
    Return (since_date, today) covering exactly lookback_days calendar days.

    Convention (fixed globally per ARCHITECTURE.md §5.2): the window **includes
    today's scores** (i.e. the k days are today, yesterday, … today-(k-1)).
    If today's evaluation has not yet run, the DB simply returns no row for
    today and the mean is computed from the available completed days only.
    """
    d     = date.fromisoformat(today)
    since = d - timedelta(days=lookback_days - 1)
    return since.isoformat(), today


def compute_tier_weights(
    db_path: str,
    eligible_commits: list[MinerCommit],
    tiers: list[TierConfig],
    today: str,
) -> dict[int, float]:
    """
    Compute raw emission shares for each eligible UID across all tiers.
    A UID can accumulate emissions from multiple tiers; shares are additive.
    Returns {uid: raw_emission} (not yet normalised to sum=1).
    """
    eligible_uids  = [c.uid for c in eligible_commits]
    uid_to_commit  = {c.uid: c for c in eligible_commits}
    emissions: dict[int, float] = {uid: 0.0 for uid in eligible_uids}

    for tier in tiers:
        since_date, until_date = _lookback_date_range(today, tier.lookback_days)

        # Build aggregate Fβ (mean) over the lookback window for each UID
        # Ranking tuple: (-mean_fb, -mean_recall, -mean_precision, params, commit_block)
        ranked: list[tuple[float, float, float, int, int, int]] = []
        for uid in eligible_uids:
            scores = query_scores_for_uid(db_path, uid, since_date, until_date)
            if not scores:
                continue
            n          = len(scores)
            mean_fb    = sum(s["fb_score"]        for s in scores) / n
            mean_rec   = sum(s["recall_score"]    for s in scores) / n
            mean_prec  = sum(s["precision_score"] for s in scores) / n
            c          = uid_to_commit[uid]
            params     = c.params if c.params > 0 else 10**12  # unknown → treated as very large
            ranked.append((mean_fb, mean_rec, mean_prec, params, c.commit_block, uid))

        if not ranked:
            logger.debug(f"Tier {tier.name}: no UIDs with scores in [{since_date}, {until_date}]")
            continue

        # Sort by tie-breaking rules (§6): higher Fβ > recall > precision > fewer params > earlier block
        ranked.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3], x[4]))

        n_total = len(ranked)
        if tier.top_n is not None:
            qualifiers = [row[5] for row in ranked[: tier.top_n]]
        else:
            top_k      = max(1, round(tier.top_pct * n_total))
            qualifiers = [row[5] for row in ranked[: top_k]]

        if not qualifiers:
            continue

        share_per_uid = tier.emission_share / len(qualifiers)
        for uid in qualifiers:
            emissions[uid] += share_per_uid
            logger.debug(f"Tier {tier.name}: UID {uid} += {share_per_uid:.4f}")

        logger.info(
            f"Tier {tier.name}: {len(qualifiers)} qualifier(s) | "
            f"{share_per_uid:.4f} each | window [{since_date}, {until_date}]"
        )

    return emissions


# ── Weight setting ────────────────────────────────────────────────────────────

async def set_weights_from_tiers(
    subtensor: bt.Subtensor,
    wallet: Wallet,
    netuid: int,
    db_path: str,
    tiers: list[TierConfig],
    eligible_commits: list[MinerCommit],
) -> bool:
    """Derive weights from tier emissions and submit them on chain."""
    async def _burn_to_uid0(reason: str) -> bool:
        logger.warning(f"{reason} – burning weight to UID 0")
        return bool(await asyncio.to_thread(
            subtensor.set_weights,
            wallet=wallet, netuid=netuid, uids=[0], weights=[1.0],
            wait_for_inclusion=True, wait_for_finalization=False,
        ))

    if not eligible_commits:
        return await _burn_to_uid0("No eligible commits")

    today     = date.today().isoformat()
    emissions = await asyncio.to_thread(
        compute_tier_weights, db_path, eligible_commits, tiers, today
    )

    # Only submit UIDs that actually earned emissions; zero-weight UIDs are
    # excluded from the set_weights call to keep the payload clean.
    active = {uid: em for uid, em in emissions.items() if em > 0.0}
    if not active:
        return await _burn_to_uid0("All tier emissions are zero (no daily scores yet)")

    total   = sum(active.values())
    uids    = list(active.keys())
    weights = [active[u] / total for u in uids]

    logger.info(f"Setting weights for {len(uids)} UID(s) | total raw emission = {total:.4f}")
    top_display = sorted(zip(uids, weights), key=lambda x: -x[1])[:10]
    for uid, w in top_display:
        logger.info(f"  UID {uid:4d}  weight={w:.4f}")

    success = await asyncio.to_thread(
        subtensor.set_weights,
        wallet=wallet,
        netuid=netuid,
        uids=uids,
        weights=weights,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    return bool(success)


# ── Heartbeat monitor ─────────────────────────────────────────────────────────

def _heartbeat_monitor(
    last_heartbeat: list[float],
    stop_event: threading.Event,
    timeout: int,
) -> None:
    while not stop_event.is_set():
        time.sleep(5)
        if time.time() - last_heartbeat[0] > timeout:
            logger.error(f"No heartbeat in {timeout}s – restarting process")
            logging.shutdown()
            os.execv(sys.executable, [sys.executable] + sys.argv)


# ── Main validator loop ───────────────────────────────────────────────────────

async def validator_loop(
    network: str,
    netuid: int,
    coldkey: str,
    hotkey: str,
    db_path: str,
    dataset_base_url: str,
    dataset_api_key: str,
    image_base_url: str,
    target_conditions: frozenset[str],
    chutes_llm_url: str,
    chutes_api_key: str,
    chutes_timeout: int,
    chutes_max_tokens: int,
    chutes_merge_system_into_user: bool,
    chutes_max_concurrent: int,
    eval_samples_per_day: int,
    eval_delay_days: int,
    beta: float,
    tiers: list[TierConfig],
    heartbeat_timeout: int,
    *,
    allow_local: bool,
    local_sglang_host: str,
    local_sglang_port: int,
    sglang_extra_argv: list[str],
    sglang_startup_timeout: float,
) -> None:
    # ── Heartbeat thread ──────────────────────────────────────────────────────
    last_heartbeat: list[float] = [time.time()]
    stop_event = threading.Event()
    hb_thread  = threading.Thread(
        target=_heartbeat_monitor,
        args=(last_heartbeat, stop_event, heartbeat_timeout),
        daemon=True,
    )
    hb_thread.start()

    try:
        # ── Initialise chain connections ──────────────────────────────────────
        wallet     = Wallet(name=coldkey, hotkey=hotkey)
        subtensor  = bt.Subtensor(network=network)
        metagraph  = bt.Metagraph(netuid=netuid, network=network)
        await asyncio.to_thread(metagraph.sync, subtensor=subtensor)
        logger.info(f"Metagraph synced: {metagraph.n} neurons @ block {metagraph.block}")

        my_hotkey = wallet.hotkey.ss58_address
        if my_hotkey not in metagraph.hotkeys:
            logger.error(f"Hotkey {my_hotkey} not registered on netuid {netuid}")
            return
        my_uid = metagraph.hotkeys.index(my_hotkey)
        logger.info(f"Validator UID: {my_uid}")

        hyperparams       = await asyncio.to_thread(subtensor.get_subnet_hyperparameters, netuid)
        tempo             = int(hyperparams.tempo)
        weights_rate_limit = int(getattr(hyperparams, "weights_rate_limit", tempo))
        logger.info(f"Tempo: {tempo} blocks | Weights rate limit: {weights_rate_limit} blocks")

        # ── Initialise DB ─────────────────────────────────────────────────────
        await asyncio.to_thread(init_db, db_path)

        # ── State ─────────────────────────────────────────────────────────────
        last_weight_block: int        = 0
        last_eval_date:    str        = ""
        eligible_commits:  list[MinerCommit] = []

        # Pre-populate eligible_commits from chain so weight-setting works immediately
        current_block = await asyncio.to_thread(subtensor.get_current_block)
        commits = await asyncio.to_thread(
            fetch_all_commits,
            subtensor,
            metagraph,
            netuid,
            current_block,
            allow_local=allow_local,
        )
        eligible_commits = filter_eligible(deduplicate_commits(commits), eval_delay_days)

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            try:
                await asyncio.to_thread(metagraph.sync, subtensor=subtensor)
                current_block      = await asyncio.to_thread(subtensor.get_current_block)
                last_heartbeat[0]  = time.time()
                today              = date.today().isoformat()

                # ── Daily evaluation trigger ──────────────────────────────────
                already_ran = await asyncio.to_thread(eval_already_ran_today, db_path, today)
                if today != last_eval_date and not already_ran:
                    last_eval_date   = today  # set before to avoid tight-loop retries on failure
                    eligible_commits = await run_daily_evaluation(
                        subtensor=subtensor,
                        metagraph=metagraph,
                        netuid=netuid,
                        current_block=current_block,
                        db_path=db_path,
                        dataset_base_url=dataset_base_url,
                        dataset_api_key=dataset_api_key,
                        image_base_url=image_base_url,
                        target_conditions=target_conditions,
                        chutes_llm_url=chutes_llm_url,
                        chutes_api_key=chutes_api_key,
                        chutes_timeout=chutes_timeout,
                        chutes_max_tokens=chutes_max_tokens,
                        chutes_merge_system_into_user=chutes_merge_system_into_user,
                        chutes_max_concurrent=chutes_max_concurrent,
                        eval_samples_per_day=eval_samples_per_day,
                        eval_delay_days=eval_delay_days,
                        beta=beta,
                        allow_local=allow_local,
                        local_sglang_host=local_sglang_host,
                        local_sglang_port=local_sglang_port,
                        sglang_extra_argv=sglang_extra_argv,
                        sglang_startup_timeout=sglang_startup_timeout,
                    )

                # ── Weight-setting trigger ────────────────────────────────────
                blocks_since_weights = current_block - last_weight_block
                if blocks_since_weights >= weights_rate_limit:
                    logger.info(
                        f"Block {current_block}: setting weights "
                        f"({blocks_since_weights} blocks since last update)"
                    )
                    success = await set_weights_from_tiers(
                        subtensor=subtensor,
                        wallet=wallet,
                        netuid=netuid,
                        db_path=db_path,
                        tiers=tiers,
                        eligible_commits=eligible_commits,
                    )
                    if success:
                        last_weight_block = current_block
                        logger.info(f"Weights set at block {current_block}")
                    else:
                        logger.warning("set_weights returned False; will retry next tempo")
                else:
                    logger.debug(
                        f"Block {current_block}: waiting for weight update "
                        f"({blocks_since_weights}/{weights_rate_limit} blocks)"
                    )

                await asyncio.sleep(12)

            except KeyboardInterrupt:
                logger.info("Validator stopped by user")
                break
            except Exception as exc:
                logger.error(f"Loop error: {exc}", exc_info=True)
                await asyncio.sleep(30)

    finally:
        stop_event.set()
        hb_thread.join(timeout=2)


# ── CLI entry point ───────────────────────────────────────────────────────────

@click.command()
@click.option("--network",              default=lambda: os.getenv("NETWORK", "finney"),
              help="Subtensor network (finney | test | local)")
@click.option("--netuid",               type=int, default=lambda: int(os.getenv("NETUID", "1")),
              help="Subnet netuid")
@click.option("--coldkey",              default=lambda: os.getenv("WALLET_NAME", "default"),
              help="Wallet coldkey name")
@click.option("--hotkey",               default=lambda: os.getenv("HOTKEY_NAME", "default"),
              help="Wallet hotkey name")
@click.option("--log-level",            type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              default=lambda: os.getenv("LOG_LEVEL", "INFO"))
@click.option("--db-path",              default=lambda: os.getenv("DB_PATH", "validator_scores.db"),
              help="SQLite file for daily evaluation scores")
@click.option("--dataset-base-url",     default=lambda: os.getenv("DATASET_BASE_URL", ""),
              help="Public dataset API base URL (studies endpoint)")
@click.option("--dataset-api-key",      default=lambda: os.getenv("DATASET_API_KEY", ""),
              help="Bearer token for dataset API (if required)")
@click.option("--image-base-url",       default=lambda: os.getenv("IMAGE_BASE_URL", ""),
              help="Base URL prepended to image_file filenames (e.g. https://data.sotarad.ai/images)")
@click.option("--target-conditions",    default=lambda: os.getenv("TARGET_CONDITIONS", ""),
              help="Comma-separated conditions for ground-truth screen-positive (default: four targets from prompts/system_prompt.py)")
@click.option("--chutes-api-key",       default=lambda: os.getenv("CHUTES_API_KEY", ""),
              help="Chutes API key (cpk_...)")
@click.option("--chutes-llm-url",       default=lambda: os.getenv("CHUTES_LLM_URL", "https://llm.chutes.ai/v1"),
              help="Chutes OpenAI-compatible base URL")
@click.option("--chutes-timeout",       type=int, default=lambda: int(os.getenv("CHUTES_TIMEOUT", "60")),
              help="Per-inference request timeout (seconds)")
@click.option("--chutes-max-tokens",    type=int, default=lambda: int(os.getenv("CHUTES_MAX_TOKENS", "1024")),
              help="max_tokens for JSON findings output (vision chat completion)")
@click.option(
    "--chutes-separate-system",
    is_flag=True,
    default=False,
    help="Send SYSTEM_PROMPT as role=system (default: merge into user text for API compatibility)",
)
@click.option("--chutes-max-concurrent",type=int, default=lambda: int(os.getenv("CHUTES_MAX_CONCURRENT", "4")),
              help="Max parallel miner evaluations")
@click.option("--eval-samples-per-day", type=int, default=lambda: int(os.getenv("EVAL_SAMPLES_PER_DAY", "100")),
              help="Max dataset samples fetched per daily evaluation")
@click.option("--eval-delay-days",      type=int, default=lambda: int(os.getenv("EVAL_DELAY_DAYS", "1")),
              help="Min days after model submission before it qualifies for evaluation")
@click.option("--fbeta-beta",           type=float, default=lambda: float(os.getenv("FBETA_BETA", "2.0")),
              help="β parameter for Fβ scoring (>1 weights recall more than precision)")
@click.option("--heartbeat-timeout",    type=int, default=lambda: int(os.getenv("HEARTBEAT_TIMEOUT", "600")),
              help="Seconds without loop heartbeat before auto-restart")
@click.option(
    "--allow-local",
    is_flag=True,
    default=False,
    help="If set, miners without chute_id are evaluated via HF repo+revision on local SGLang",
)
@click.option(
    "--local-sglang-host",
    default=lambda: os.getenv("LOCAL_SGLANG_HOST", "127.0.0.1"),
    help="Bind address for subprocess sglang.launch_server",
)
@click.option(
    "--local-sglang-port",
    type=int,
    default=lambda: int(os.getenv("LOCAL_SGLANG_PORT", "30000")),
    help="Port for local SGLang (one server at a time when evaluating HF-only miners)",
)
@click.option(
    "--sglang-startup-timeout",
    type=float,
    default=lambda: float(os.getenv("SGLANG_STARTUP_TIMEOUT", "600")),
    help="Seconds to wait for local SGLang HTTP readiness",
)
@click.option(
    "--sglang-extra-args",
    default=lambda: os.getenv("SGLANG_EXTRA_ARGS", ""),
    help="Extra arguments for sglang.launch_server (shell-quoted, e.g. \"--tp 1 --mem-fraction-static 0.9\")",
)
@click.option("--mock",                 is_flag=True, default=lambda: os.getenv("MOCK", "").lower() == "true",
              help="Use local mock servers: dataset API on :8100, Chutes on :8200 (see mock/)")
def main(
    network: str,
    netuid: int,
    coldkey: str,
    hotkey: str,
    log_level: str,
    db_path: str,
    dataset_base_url: str,
    dataset_api_key: str,
    image_base_url: str,
    target_conditions: str,
    chutes_api_key: str,
    chutes_llm_url: str,
    chutes_timeout: int,
    chutes_max_tokens: int,
    chutes_separate_system: bool,
    chutes_max_concurrent: int,
    eval_samples_per_day: int,
    eval_delay_days: int,
    fbeta_beta: float,
    heartbeat_timeout: int,
    allow_local: bool,
    local_sglang_host: str,
    local_sglang_port: int,
    sglang_startup_timeout: float,
    sglang_extra_args: str,
    mock: bool,
) -> None:
    """SotaRad subnet validator: scores radiology pre-screening models via Fβ."""
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    chutes_merge_system_into_user = not chutes_separate_system

    if mock:
        dataset_base_url = dataset_base_url or "http://localhost:8100"
        image_base_url   = image_base_url   or "http://localhost:8100/images"
        chutes_llm_url   = "http://localhost:8200/v1"
        chutes_api_key   = chutes_api_key   or "mock-key"
        eval_delay_days  = 0  # bypass temporal filter so historical data is eligible
        logger.info("Mock mode: dataset=:8100  chutes=:8200  eval_delay_days=0")

    parsed_conditions = _parse_target_conditions(target_conditions)
    sglang_argv = shlex.split(sglang_extra_args) if sglang_extra_args.strip() else []
    logger.info(
        f"Starting SotaRad validator | network={network} netuid={netuid} "
        f"β={fbeta_beta} eval_delay={eval_delay_days}d samples/day={eval_samples_per_day} "
        f"chutes_max_tokens={chutes_max_tokens} merge_system_into_user={chutes_merge_system_into_user} "
        f"allow_local={allow_local} target_conditions={sorted(parsed_conditions)}"
    )
    asyncio.run(validator_loop(
        network=network,
        netuid=netuid,
        coldkey=coldkey,
        hotkey=hotkey,
        db_path=db_path,
        dataset_base_url=dataset_base_url,
        dataset_api_key=dataset_api_key,
        image_base_url=image_base_url,
        target_conditions=parsed_conditions,
        chutes_llm_url=chutes_llm_url,
        chutes_api_key=chutes_api_key,
        chutes_timeout=chutes_timeout,
        chutes_max_tokens=chutes_max_tokens,
        chutes_merge_system_into_user=chutes_merge_system_into_user,
        chutes_max_concurrent=chutes_max_concurrent,
        eval_samples_per_day=eval_samples_per_day,
        eval_delay_days=eval_delay_days,
        beta=fbeta_beta,
        tiers=DEFAULT_TIERS,
        heartbeat_timeout=heartbeat_timeout,
        allow_local=allow_local,
        local_sglang_host=local_sglang_host,
        local_sglang_port=local_sglang_port,
        sglang_extra_argv=sglang_argv,
        sglang_startup_timeout=sglang_startup_timeout,
    ))


if __name__ == "__main__":
    main()
