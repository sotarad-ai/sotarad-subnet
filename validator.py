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
(``repo``, ``revision``, ``chute_id``). Parameter counts for tier tie-breaks are
read from Hugging Face, not from the commitment. Validators call:

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
  `condition ∈ EVAL_TARGET_CONDITIONS` and `status == "active"`.

TP / FP / FN / TN and Fβ are computed from these study-level predictions.

``EVAL_TARGET_CONDITIONS`` is fixed in this file (keep aligned with
``prompts/system_prompt.TARGET_CONDITIONS`` for miner prompt vocabulary).

Evaluation **period** (default 1440 minutes): ``period_id = floor(unix_timestamp //
(60 * eval_period_minutes))``. SQLite ``eval_date`` stores zero-padded ``period_id``.
Commit eligibility and per-miner sample cutoffs use absolute unix times
(``eval_delay_minutes``). Tier ``lookback_days`` is converted to a number of periods.

Tier incentives, duplicate policy, and dataset API contract are unchanged; see
ARCHITECTURE.md and inline code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shlex
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import bittensor as bt
import click
from bittensor_wallet import Wallet

from local_sglang import SglangSubprocessServer
from prompts.response_parse import parse_findings_json_array
from prompts.system_prompt import build_chutes_messages


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Dataset ground-truth labels ───────────────────────────────────────────────
# ``report_findings.positive_findings[].condition`` must match exactly (case-sensitive).
# Keep in sync with ``prompts/system_prompt.TARGET_CONDITIONS``.
EVAL_TARGET_CONDITIONS: frozenset[str] = frozenset(
    ("Pneumonia", "Tuberculosis", "Bronchitis", "Silicosis")
)


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
    eval_date: str    # zero-padded eval period id (see format_eval_period_key)
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
        # eval_date stores format_eval_period_key(period_id), not calendar dates.
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


def eval_already_ran_for_period(db_path: str, eval_period_key: str) -> bool:
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT COUNT(*) FROM daily_scores WHERE eval_date = ?",
            (eval_period_key,),
        ).fetchone()
    return row[0] > 0


def eval_period_seconds(eval_period_minutes: int) -> int:
    """Length of one evaluation period in seconds (minimum 60s)."""
    return max(60, int(eval_period_minutes) * 60)


def eval_period_id_at(timestamp: float, period_s: int) -> int:
    """Floor bucket: ``int(timestamp) // period_s``."""
    return int(timestamp) // period_s


def format_eval_period_key(period_id: int) -> str:
    """Lexicographic sort matches numeric order (fixed width)."""
    return f"{period_id:012d}"


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


def _parameter_count_from_config_json(cfg: object) -> int:
    """Best-effort total parameters from a ``config.json``-like dict."""
    if not isinstance(cfg, dict):
        return 0
    for key in ("num_parameters", "num_params", "n_parameters", "total_parameters"):
        v = cfg.get(key)
        if isinstance(v, int) and v > 0:
            return v
    for child in (
        "text_config",
        "vision_config",
        "audio_config",
        "encoder_config",
        "decoder_config",
    ):
        if child in cfg:
            n = _parameter_count_from_config_json(cfg[child])
            if n > 0:
                return n
    return 0


def fetch_model_parameter_count_from_hf(repo: str, revision: str) -> int:
    """
    Resolve parameter count for ``repo`` @ ``revision`` from the Hugging Face Hub.

    Uses safetensors metadata from the Hub API when available, otherwise scans
    ``config.json``. Returns 0 if unknown (tier tie-break treats as very large).
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed; cannot resolve parameter counts "
            "(install huggingface_hub or tie-breaks assume unknown size)"
        )
        return 0

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()

    try:
        info = api.model_info(repo_id=repo, revision=revision)
        st = getattr(info, "safetensors", None)
        if st is not None:
            total = int(getattr(st, "total", 0) or 0)
            if total > 0:
                return total
            params_map = getattr(st, "parameters", None) or (
                st.get("parameters", {}) if isinstance(st, dict) else {}
            )
            if isinstance(params_map, dict) and params_map:
                s = sum(int(v) for v in params_map.values() if isinstance(v, int))
                if s > 0:
                    return s
    except Exception as exc:
        logger.debug("HfApi.model_info(%s @ %s): %s", repo, revision[:12], exc)

    try:
        path = hf_hub_download(
            repo_id=repo,
            filename="config.json",
            revision=revision,
            token=token,
        )
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        n = _parameter_count_from_config_json(cfg)
        if n > 0:
            return n
    except Exception as exc:
        logger.debug("config.json for %s @ %s: %s", repo, revision[:12], exc)

    return 0


def resolve_uid_parameter_counts(commits: list[MinerCommit]) -> dict[int, int]:
    """
    Parameter counts from Hugging Face for tier tie-breaks only — never from chain.

    Missing / unknown counts are omitted from the dict; ``compute_tier_weights`` treats
    missing UID as unknown size (worst tie-break).
    """
    out: dict[int, int] = {}
    for c in commits:
        n = fetch_model_parameter_count_from_hf(c.repo, c.revision)
        if n > 0:
            out[c.uid] = n
            logger.info("UID %s: parameter count %s (from Hugging Face)", c.uid, f"{n:,}")
        else:
            logger.info(
                "UID %s: parameter count unknown (HF); tier tie-break uses unknown size",
                c.uid,
            )
    return out


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
        # get_all_commitments() returns {ss58_hotkey: commitment_str}. Some paths may use int UID keys.
        if isinstance(uid_key, int):
            uid = uid_key
        else:
            key_str = str(uid_key).strip()
            try:
                uid = int(key_str)
            except ValueError:
                if key_str not in metagraph.hotkeys:
                    logger.debug(
                        "Skipping commitment: hotkey %s… not in metagraph (synced n=%s)",
                        key_str[:16],
                        metagraph.n,
                    )
                    continue
                uid = metagraph.hotkeys.index(key_str)

        if uid < 0 or uid >= len(metagraph.hotkeys):
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
    eval_delay_minutes: int,
) -> list[MinerCommit]:
    """
    Keep only commits whose on-chain submission time is strictly before
    ``now - eval_delay_minutes`` (absolute unix time).
    """
    delay_s = max(0, int(eval_delay_minutes)) * 60
    cutoff_ts = time.time() - delay_s
    eligible = [c for c in commits if c.commit_ts < cutoff_ts]
    excluded = len(commits) - len(eligible)
    if excluded:
        logger.info(
            "%s miner(s) excluded: committed too recently (< %s min before now)",
            excluded,
            eval_delay_minutes,
        )
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
    condition ∈ ``EVAL_TARGET_CONDITIONS`` AND status == "active", else 0.
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
            label       = _is_screen_positive(positive_findings, EVAL_TARGET_CONDITIONS)
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
    eval_period_key: str,
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
        eval_date=eval_period_key,
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
    chutes_llm_url: str,
    chutes_api_key: str,
    chutes_timeout: int,
    chutes_max_tokens: int,
    chutes_merge_system_into_user: bool,
    chutes_max_concurrent: int,
    eval_samples_per_day: int,
    eval_period_minutes: int,
    eval_delay_minutes: int,
    eval_period_key: str,
    beta: float,
    *,
    allow_local: bool,
    local_sglang_host: str,
    local_sglang_port: int,
    sglang_extra_argv: list[str],
    sglang_startup_timeout: float,
) -> tuple[list[MinerCommit], dict[int, int]]:
    """
    One evaluation pass for the current eval period (unix bucket).

    Returns eligible commits and a UID → parameter-count map from Hugging Face
    (for tier tie-breaks only; not from chain).
    """
    period_s = eval_period_seconds(eval_period_minutes)
    logger.info(
        "=== Evaluation pass starting | period_key=%s (period_len=%s min) ===",
        eval_period_key,
        eval_period_minutes,
    )

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

    # 3. Temporal filter: commit_ts must be older than eval_delay_minutes (absolute time)
    eligible = filter_eligible(eligible, eval_delay_minutes)
    logger.info("%s miner(s) eligible for this evaluation pass", len(eligible))

    if not eligible:
        logger.info("No eligible miners – skipping inference pass")
        return eligible, {}

    # 4. Fetch labelled evaluation samples — window = current eval period so far
    #    [period_id * period_s, min(now, (period_id+1)*period_s)); period_id from bucket key
    now_ts = time.time()
    period_id = int(eval_period_key)
    eval_start_ts = period_id * period_s
    eval_end_ts = min(now_ts, (period_id + 1) * period_s)

    async with aiohttp.ClientSession() as session:
        samples = await fetch_eval_samples(
            session, dataset_base_url, dataset_api_key,
            image_base_url=image_base_url,
            after_ts=eval_start_ts,
            before_ts=eval_end_ts,
            limit=eval_samples_per_day,
        )

        if not samples:
            logger.warning("No evaluation samples returned – aborting evaluation pass")
            uid_params = await asyncio.to_thread(resolve_uid_parameter_counts, eligible)
            return eligible, uid_params

        logger.info(f"Fetched {len(samples)} evaluation samples from dataset API")

        # 5. Evaluate each eligible miner (bounded concurrency via semaphore;
        #    local SGLang is serialized — one subprocess / GPU at a time)
        sem = asyncio.Semaphore(chutes_max_concurrent)
        local_slot = asyncio.Semaphore(1)

        async def _eval_one(commit: MinerCommit) -> Optional[EvalResult]:
            miner_cutoff = commit.commit_ts + max(0, int(eval_delay_minutes)) * 60
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
                        eval_period_key,
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
                    eval_period_key,
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

    logger.info("=== Evaluation pass complete: %s/%s miners scored ===", saved, len(eligible))

    uid_params = await asyncio.to_thread(resolve_uid_parameter_counts, eligible)
    return eligible, uid_params


# ── Tier engine ───────────────────────────────────────────────────────────────

def _lookback_period_keys(
    current_period_id: int,
    lookback_days: int,
    eval_period_minutes: int,
) -> tuple[str, str]:
    """
    Map nominal tier ``lookback_days`` to a contiguous range of eval period keys
    ending at ``current_period_id``. Uses ``ceil(days * 24h / period_len)`` periods.
    """
    n_periods = max(1, math.ceil(lookback_days * 24 * 60 / eval_period_minutes))
    since_pid = current_period_id - (n_periods - 1)
    return format_eval_period_key(since_pid), format_eval_period_key(current_period_id)


def compute_tier_weights(
    db_path: str,
    eligible_commits: list[MinerCommit],
    tiers: list[TierConfig],
    current_period_id: int,
    eval_period_minutes: int,
    uid_param_counts: dict[int, int],
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
        since_key, until_key = _lookback_period_keys(
            current_period_id, tier.lookback_days, eval_period_minutes
        )

        # Build aggregate Fβ (mean) over the lookback window for each UID
        # Ranking tuple: (-mean_fb, -mean_recall, -mean_precision, params, commit_block)
        ranked: list[tuple[float, float, float, int, int, int]] = []
        for uid in eligible_uids:
            scores = query_scores_for_uid(db_path, uid, since_key, until_key)
            if not scores:
                continue
            n          = len(scores)
            mean_fb    = sum(s["fb_score"]        for s in scores) / n
            mean_rec   = sum(s["recall_score"]    for s in scores) / n
            mean_prec  = sum(s["precision_score"] for s in scores) / n
            c          = uid_to_commit[uid]
            nparams    = uid_param_counts.get(uid, 0)
            params_tb  = nparams if nparams > 0 else 10**12  # unknown → treated as very large
            ranked.append((mean_fb, mean_rec, mean_prec, params_tb, c.commit_block, uid))

        if not ranked:
            logger.debug(
                "Tier %s: no UIDs with scores in period range [%s, %s]",
                tier.name,
                since_key,
                until_key,
            )
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
            "Tier %s: %s qualifier(s) | %.4f each | period window [%s, %s]",
            tier.name,
            len(qualifiers),
            share_per_uid,
            since_key,
            until_key,
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
    uid_param_counts: dict[int, int],
    eval_period_minutes: int,
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

    now_ts = time.time()
    period_s = eval_period_seconds(eval_period_minutes)
    current_period_id = eval_period_id_at(now_ts, period_s)
    emissions = await asyncio.to_thread(
        compute_tier_weights,
        db_path,
        eligible_commits,
        tiers,
        current_period_id,
        eval_period_minutes,
        uid_param_counts,
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
    chutes_llm_url: str,
    chutes_api_key: str,
    chutes_timeout: int,
    chutes_max_tokens: int,
    chutes_merge_system_into_user: bool,
    chutes_max_concurrent: int,
    eval_samples_per_day: int,
    eval_period_minutes: int,
    eval_delay_minutes: int,
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
        last_eval_period_key: str     = ""
        eligible_commits:  list[MinerCommit] = []
        uid_param_counts:  dict[int, int]    = {}
        period_s_loop = eval_period_seconds(eval_period_minutes)

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
        eligible_commits = filter_eligible(deduplicate_commits(commits), eval_delay_minutes)
        uid_param_counts = await asyncio.to_thread(
            resolve_uid_parameter_counts, eligible_commits
        )

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            try:
                await asyncio.to_thread(metagraph.sync, subtensor=subtensor)
                current_block      = await asyncio.to_thread(subtensor.get_current_block)
                last_heartbeat[0]  = time.time()
                now_ts = time.time()
                period_key = format_eval_period_key(eval_period_id_at(now_ts, period_s_loop))

                # ── Evaluation pass (once per eval period) ───────────────────
                if period_key != last_eval_period_key:
                    last_eval_period_key = period_key
                    already_ran = await asyncio.to_thread(
                        eval_already_ran_for_period, db_path, period_key
                    )
                    if not already_ran:
                        eligible_commits, uid_param_counts = await run_daily_evaluation(
                            subtensor=subtensor,
                            metagraph=metagraph,
                            netuid=netuid,
                            current_block=current_block,
                            db_path=db_path,
                            dataset_base_url=dataset_base_url,
                            dataset_api_key=dataset_api_key,
                            image_base_url=image_base_url,
                            chutes_llm_url=chutes_llm_url,
                            chutes_api_key=chutes_api_key,
                            chutes_timeout=chutes_timeout,
                            chutes_max_tokens=chutes_max_tokens,
                            chutes_merge_system_into_user=chutes_merge_system_into_user,
                            chutes_max_concurrent=chutes_max_concurrent,
                            eval_samples_per_day=eval_samples_per_day,
                            eval_period_minutes=eval_period_minutes,
                            eval_delay_minutes=eval_delay_minutes,
                            eval_period_key=period_key,
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
                        uid_param_counts=uid_param_counts,
                        eval_period_minutes=eval_period_minutes,
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
              help="Max dataset samples fetched per evaluation pass")
@click.option(
    "--eval-period-minutes",
    type=int,
    default=lambda: int(os.getenv("EVAL_PERIOD_MINUTES", "1440")),
    help="Evaluation period length in minutes (default 1440 = 24h). Period id = floor(unix_ts / (60*minutes)).",
)
@click.option(
    "--eval-delay-minutes",
    type=int,
    default=lambda: int(os.getenv("EVAL_DELAY_MINUTES", "1440")),
    help="Min minutes after on-chain commit before a miner is evaluated (default 1440 ≈ former 1 day).",
)
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
    chutes_api_key: str,
    chutes_llm_url: str,
    chutes_timeout: int,
    chutes_max_tokens: int,
    chutes_separate_system: bool,
    chutes_max_concurrent: int,
    eval_samples_per_day: int,
    eval_period_minutes: int,
    eval_delay_minutes: int,
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
        eval_delay_minutes = 0  # bypass temporal filter so historical data is eligible
        logger.info("Mock mode: dataset=:8100  chutes=:8200  eval_delay_minutes=0")

    sglang_argv = shlex.split(sglang_extra_args) if sglang_extra_args.strip() else []
    logger.info(
        f"Starting SotaRad validator | network={network} netuid={netuid} "
        f"β={fbeta_beta} eval_period={eval_period_minutes}m eval_delay={eval_delay_minutes}m "
        f"samples/pass={eval_samples_per_day} "
        f"chutes_max_tokens={chutes_max_tokens} merge_system_into_user={chutes_merge_system_into_user} "
        f"allow_local={allow_local} eval_target_conditions={sorted(EVAL_TARGET_CONDITIONS)}"
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
        chutes_llm_url=chutes_llm_url,
        chutes_api_key=chutes_api_key,
        chutes_timeout=chutes_timeout,
        chutes_max_tokens=chutes_max_tokens,
        chutes_merge_system_into_user=chutes_merge_system_into_user,
        chutes_max_concurrent=chutes_max_concurrent,
        eval_samples_per_day=eval_samples_per_day,
        eval_period_minutes=eval_period_minutes,
        eval_delay_minutes=eval_delay_minutes,
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
