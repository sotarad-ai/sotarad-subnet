"""
SotaRad Model Registration CLI
================================

A chain-interaction utility that helps miners submit and inspect their
on-chain model commitments for the SotaRad subnet.

This tool writes and reads the commitment JSON that validator.py parses.
It does NOT contain any inference logic or training code.

Commitment schema (JSON, ≤ 512 bytes)
--------------------------------------
    {
        "repo":     "hf-username/model-name",   # Hugging Face repo ID
        "revision": "abc1234...",               # git commit SHA (full or short)
        "chute_id": "chutes-deployment-uuid",   # optional; empty if no Chutes deploy (HF + local SGLang)
    }

Parameter counts for tier tie-breaks are **not** committed on-chain; validators
resolve them from the Hugging Face model (``repo`` + ``revision``).

Typical miner workflow (reference: docs/ARCHITECTURE.md §3)
-----------------------------------------------------------
1.  Train / improve a radiology model off-chain.
2.  Upload to Hugging Face, record the git revision SHA.
3.  (Optional) Deploy to Chutes and obtain ``chute_id``.
4.  Register on-chain: python register.py commit --repo user/model --revision <SHA> [--chute-id <id>]
    Omit or pass empty ``--chute-id`` when serving via HF + local inference only; validators must use ``--allow-local``.
5.  Verify:            python register.py status

Commands
--------
  commit    Submit a new commitment to the Bittensor chain.
  status    Display your current commitment and evaluation eligibility.
  check     Validate commitment fields locally without writing to chain.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import bittensor as bt
import click
from bittensor.core.chain_data.utils import decode_metadata
from bittensor_wallet import Wallet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Chain rate limit: one commitment per ~100 blocks (~20 minutes).
_COMMIT_RATE_LIMIT_BLOCKS = 100
_BLOCK_TIME_S = 12.0


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_repo(repo: str) -> str:
    """Ensure repo looks like 'owner/model-name'."""
    repo = repo.strip()
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise click.BadParameter(
            f"'{repo}' is not a valid Hugging Face repo ID. "
            "Expected format: owner/model-name  (e.g. myuser/qwen2-tb-v1)"
        )
    return repo


def _validate_revision(revision: str) -> str:
    """Ensure revision looks like a git SHA (hex, 7–40 chars)."""
    revision = revision.strip()
    if not revision:
        raise click.BadParameter("revision cannot be empty")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", revision):
        raise click.BadParameter(
            f"'{revision}' does not look like a git SHA. "
            "Run `git rev-parse HEAD` inside your HF repo clone to get the full SHA."
        )
    return revision.lower()


def _normalize_chute_id(chute_id: str) -> str:
    """Chutes deployment id, or empty string for HF-only (validator ``--allow-local``)."""
    return chute_id.strip()


def _build_payload(repo: str, revision: str, chute_id: str) -> str:
    """Serialize the commitment to a compact JSON string."""
    data: dict = {
        "repo":     repo,
        "revision": revision,
        "chute_id": chute_id,
    }
    payload = json.dumps(data, separators=(",", ":"))
    if len(payload) > 512:
        raise click.UsageError(
            f"Commitment payload is {len(payload)} bytes (max 512). "
            "Shorten your repo or chute_id."
        )
    return payload


# ── Chain helpers ─────────────────────────────────────────────────────────────

def _get_uid(subtensor: bt.Subtensor, netuid: int, hotkey_ss58: str) -> int | None:
    try:
        metagraph = bt.Metagraph(netuid=netuid, network=subtensor.network)
        metagraph.sync(subtensor=subtensor)
        if hotkey_ss58 in metagraph.hotkeys:
            return metagraph.hotkeys.index(hotkey_ss58)
    except Exception as exc:
        logger.debug(f"_get_uid error: {exc}")
    return None


def _get_commit_info(
    subtensor: bt.Subtensor,
    netuid: int,
    hotkey_ss58: str,
) -> tuple[str | None, int]:
    """
    Returns (commitment_data_str, commit_block).
    commit_block is 0 if unavailable.

    Uses a direct ``CommitmentOf`` query and decodes only when the value is a
    dict. We avoid ``subtensor.get_commitment`` here: on an empty / missing
    commitment, bittensor's ``get_commitment_metadata`` can return ``""``, and
    ``decode_metadata`` then does ``metadata[\"info\"]`` and logs
    ``TypeError: string indices must be integers, not 'str'``.
    """
    try:
        result = subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey_ss58],
            block_hash=subtensor.determine_block_hash(None),
        )
    except Exception as exc:
        logger.debug("CommitmentOf query failed: %s", exc)
        return None, 0

    if result is None:
        return None, 0

    # py-substrate-interface: sometimes a ScaleObj with .value, sometimes a raw dict.
    if isinstance(result, dict):
        val = result
    else:
        val = getattr(result, "value", None)
    if val is None or not isinstance(val, dict):
        return None, 0

    commit_block = int(val.get("block", 0))
    data_str: str | None = None
    try:
        data_str = decode_metadata(val)
    except Exception as exc:
        logger.debug("decode_metadata failed: %s", exc)
        data_str = None
    return data_str, commit_block


def _block_to_human_ts(commit_block: int, current_block: int) -> str:
    if commit_block == 0:
        return "unknown"
    blocks_ago = max(0, current_block - commit_block)
    approx_ts  = time.time() - blocks_ago * _BLOCK_TIME_S
    dt         = datetime.fromtimestamp(approx_ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _blocks_since_commit(commit_block: int, current_block: int) -> int:
    return max(0, current_block - commit_block)


# ── Shared CLI options ────────────────────────────────────────────────────────

_CHAIN_OPTIONS = [
    click.option("--network", default=lambda: os.getenv("NETWORK", "finney"),
                 show_default=True, help="Subtensor network (finney | test | local)"),
    click.option("--netuid",  type=int, default=lambda: int(os.getenv("NETUID", "1")),
                 show_default=True, help="Subnet netuid"),
    click.option("--coldkey", default=lambda: os.getenv("WALLET_NAME", "default"),
                 show_default=True, help="Wallet coldkey name"),
    click.option("--hotkey",  default=lambda: os.getenv("HOTKEY_NAME", "default"),
                 show_default=True, help="Wallet hotkey name"),
]


def _chain_options(fn):
    for opt in reversed(_CHAIN_OPTIONS):
        fn = opt(fn)
    return fn


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """SotaRad model registration tool."""


# ── commit ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--repo",      required=True,
              help="Hugging Face repo ID, e.g. myuser/my-tb-model")
@click.option("--revision",  required=True,
              help="Git commit SHA of the HF artifact (40-char hex)")
@click.option(
    "--chute-id",
    default="",
    show_default=False,
    help="Chutes deployment ID from `af chutes_push` (optional; omit or '' for HF-only / local SGLang)",
)
@click.option("--dry-run",   is_flag=True, default=False,
              help="Validate and print the payload without writing to chain")
@_chain_options
def commit(
    repo: str,
    revision: str,
    chute_id: str,
    dry_run: bool,
    network: str,
    netuid: int,
    coldkey: str,
    hotkey: str,
) -> None:
    """Submit a model commitment to the Bittensor chain."""
    repo     = _validate_repo(repo)
    revision = _validate_revision(revision)
    chute_id = _normalize_chute_id(chute_id)
    payload  = _build_payload(repo, revision, chute_id)

    click.echo("\n── Commitment payload ───────────────────────────────────────")
    click.echo(json.dumps(json.loads(payload), indent=2))
    click.echo(f"   Size: {len(payload)} bytes")

    if dry_run:
        click.secho("\n[dry-run] Not submitting to chain.", fg="yellow")
        return

    wallet    = Wallet(name=coldkey, hotkey=hotkey)
    subtensor = bt.Subtensor(network=network)

    hotkey_ss58 = wallet.hotkey.ss58_address
    if not subtensor.is_hotkey_registered(netuid=netuid, hotkey_ss58=hotkey_ss58):
        raise click.ClickException(
            f"Hotkey {hotkey_ss58} is not registered on netuid {netuid}. "
            "Register first:  btcli subnet register"
        )

    # Warn if the last commit was too recent (chain will reject it anyway)
    uid = _get_uid(subtensor, netuid, hotkey_ss58)
    if uid is not None:
        current_block = subtensor.get_current_block()
        _, last_block = _get_commit_info(subtensor, netuid, hotkey_ss58)
        blocks_since  = (current_block - last_block) if last_block else _COMMIT_RATE_LIMIT_BLOCKS
        if 0 < blocks_since < _COMMIT_RATE_LIMIT_BLOCKS:
            remaining_s = int((_COMMIT_RATE_LIMIT_BLOCKS - blocks_since) * _BLOCK_TIME_S)
            click.secho(
                f"\nWarning: last commitment was {blocks_since} blocks ago "
                f"(~{remaining_s}s until rate limit resets). "
                "The transaction may be rejected.",
                fg="yellow",
            )
            if not click.confirm("Submit anyway?"):
                raise click.Abort()

    click.echo("\nSubmitting commitment to chain…")
    try:
        subtensor.set_commitment(wallet=wallet, netuid=netuid, data=payload)
    except Exception as exc:
        raise click.ClickException(f"set_commitment failed: {exc}") from exc

    click.secho("\n✓ Commitment submitted successfully.", fg="green")
    click.echo(
        f"\nVerify with:\n"
        f"  python register.py status --network {network} --netuid {netuid} "
        f"--coldkey {coldkey} --hotkey {hotkey}"
    )


# ── status ────────────────────────────────────────────────────────────────────

@cli.command()
@_chain_options
def status(
    network: str,
    netuid: int,
    coldkey: str,
    hotkey: str,
) -> None:
    """Display current commitment and evaluation eligibility for your hotkey."""
    wallet      = Wallet(name=coldkey, hotkey=hotkey)
    subtensor   = bt.Subtensor(network=network)
    hotkey_ss58 = wallet.hotkey.ss58_address

    click.echo("\n── SotaRad registration status ──────────────────────────────")
    click.echo(f"  Network : {network}")
    click.echo(f"  Netuid  : {netuid}")
    click.echo(f"  Hotkey  : {hotkey_ss58}")

    if not subtensor.is_hotkey_registered(netuid=netuid, hotkey_ss58=hotkey_ss58):
        click.secho(
            "\n  ✗ Hotkey is NOT registered on this subnet.\n"
            "  Register with:  btcli subnet register",
            fg="red",
        )
        return

    uid = _get_uid(subtensor, netuid, hotkey_ss58)
    if uid is None:
        click.secho("  ✗ Could not resolve UID (metagraph sync failed).", fg="red")
        return

    click.echo(f"  UID     : {uid}")
    click.secho("  ✓ Registered", fg="green")

    current_block = subtensor.get_current_block()
    click.echo(f"  Block   : {current_block}")

    # ── Commitment ────────────────────────────────────────────────────────────
    click.echo("\n── Commitment ───────────────────────────────────────────────")
    data_str, commit_block = _get_commit_info(subtensor, netuid, hotkey_ss58)

    if not data_str:
        click.secho(
            "  No commitment found.\n"
            "  Submit one with:  python register.py commit",
            fg="yellow",
        )
        return

    try:
        payload = json.loads(data_str)
    except json.JSONDecodeError:
        click.secho(f"  ✗ Commitment is not valid JSON: {data_str!r}", fg="red")
        return

    click.echo(f"  Repo     : {payload.get('repo',     '—')}")
    click.echo(f"  Revision : {payload.get('revision', '—')}")
    _cid = str(payload.get("chute_id") or "").strip()
    click.echo(
        f"  Chute ID : {_cid or '— (empty; validators need --allow-local)'}"
    )
    if "params" in payload:
        click.echo(
            f"  Params   : {payload.get('params')} (legacy field; validators ignore — resolved from HF)"
        )
    else:
        click.echo("  Params   : (not on-chain; validator resolves from Hugging Face)")
    click.echo(f"  Block    : {commit_block or 'unknown'}")
    click.echo(f"  Time     : {_block_to_human_ts(commit_block, current_block)}")

    missing = [
        f
        for f in ("repo", "revision")
        if not str(payload.get(f) or "").strip()
    ]
    if missing:
        click.secho(f"\n  ✗ Commitment missing required fields: {missing}", fg="red")
        return

    # ── Age ───────────────────────────────────────────────────────────────────
    blocks_old = _blocks_since_commit(commit_block, current_block)
    age_s      = int(blocks_old * _BLOCK_TIME_S)
    age_h      = age_s // 3600
    age_m      = (age_s % 3600) // 60
    click.echo(f"\n── Commitment age ───────────────────────────────────────────")
    click.echo(f"  Committed {blocks_old} blocks ago (~{age_h}h {age_m}m)")
    click.echo(
        "  Evaluation eligibility is controlled by the validator's "
        "EVAL_DELAY_MINUTES setting (default: 1440 ≈ one day after submission)."
    )


# ── check ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--repo",     required=True, help="Hugging Face repo ID")
@click.option("--revision", required=True, help="Git commit SHA")
@click.option(
    "--chute-id",
    default="",
    show_default=False,
    help="Chutes deployment ID (optional; empty for HF-only)",
)
def check(repo: str, revision: str, chute_id: str) -> None:
    """Validate commitment fields locally without touching the chain."""
    errors: list[str] = []

    try:
        repo = _validate_repo(repo)
        click.secho(f"  ✓ repo     : {repo}", fg="green")
    except click.BadParameter as exc:
        errors.append(str(exc))
        click.secho(f"  ✗ repo     : {exc}", fg="red")

    try:
        revision = _validate_revision(revision)
        click.secho(f"  ✓ revision : {revision}", fg="green")
    except click.BadParameter as exc:
        errors.append(str(exc))
        click.secho(f"  ✗ revision : {exc}", fg="red")

    chute_id = _normalize_chute_id(chute_id)
    if chute_id:
        click.secho(f"  ✓ chute_id : {chute_id}", fg="green")
    else:
        click.echo("  – chute_id : (empty — HF-only; validators need --allow-local)")

    click.echo("  – params   : not in commitment (validator resolves from Hugging Face)")

    if not errors:
        try:
            payload = _build_payload(repo, revision, chute_id)
            click.secho(f"\n✓ Commitment is valid ({len(payload)} bytes):", fg="green")
            click.echo(json.dumps(json.loads(payload), indent=2))
        except click.UsageError as exc:
            click.secho(f"\n✗ {exc}", fg="red")
    else:
        click.secho(f"\n✗ {len(errors)} error(s) — fix above before submitting.", fg="red")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
