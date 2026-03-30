"""
Download a Hugging Face model snapshot to disk, then host it with
``local_sglang.SglangSubprocessServer`` (same subprocess + readiness checks as
the validator’s local-dir path).

Dependencies
------------
- ``pip install huggingface-hub`` (download step)
- ``pip install sglang`` (+ GPU as required by your model)

Examples
--------
Download ``org/model`` at revision ``main`` into ``./models/org__model`` and serve::

    python3 tests/download_and_host_sglang.py --repo org/model --revision main

Use an existing checkout (no download)::

    python3 tests/download_and_host_sglang.py \\
      --repo org/model --revision main \\
      --local-dir ./models/Llama-3.2-11B-Vision-Radiology-mini \\
      --skip-download

Then in another shell::

    python3 tests/test_model_request.py --image x.png --host 127.0.0.1 --port 40035
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shlex
import sys
import time
from pathlib import Path

import aiohttp

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from local_sglang import SglangSubprocessServer


def _default_local_dir(repo_id: str) -> Path:
    safe = repo_id.replace("/", "__")
    return _REPO_ROOT / "models" / safe


def _download_snapshot(repo_id: str, revision: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is required for download. Install with:\n"
            "  pip install huggingface-hub"
        ) from exc

    local_dir = local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s @ %s -> %s", repo_id, revision, local_dir)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
    )
    return local_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="Hugging Face repo id (e.g. org/model).")
    p.add_argument(
        "--revision",
        default="main",
        help="Git revision / branch / tag (default: main).",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Directory for snapshot (default: ./models/<repo with / -> __>).",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Use --local-dir as-is (must already contain weights).",
    )
    p.add_argument("--host", default="0.0.0.0", help="sglang.launch_server bind host.")
    p.add_argument("--port", type=int, default=40035, help="Listen port.")
    p.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for HTTP readiness.",
    )
    p.add_argument(
        "--extra-args",
        default="",
        help="Extra launch_server args (shell-quoted).",
    )
    p.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip GET /v1/models after the server is ready.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return p.parse_args()


async def _probe_v1_models(client_base: str) -> None:
    url = f"{client_base.rstrip('/')}/v1/models"
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logging.error("Probe failed %s: HTTP %s\n%s", url, resp.status, body[:2000])
                return
            try:
                data = await resp.json()
            except Exception:
                logging.info("Probe %s: HTTP %s (non-JSON body)", url, resp.status)
                return
    logging.info(
        "Probe OK %s — top-level keys: %s",
        url,
        list(data.keys()) if isinstance(data, dict) else type(data).__name__,
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    local_dir = args.local_dir or _default_local_dir(args.repo)
    if args.skip_download:
        if not local_dir.is_dir():
            logging.error("--skip-download: not a directory: %s", local_dir)
            sys.exit(1)
        model_path = str(local_dir.resolve())
        logging.info("Using existing weights at %s", model_path)
    else:
        model_path = str(_download_snapshot(args.repo, args.revision, local_dir))

    extra = shlex.split(args.extra_args.strip()) if args.extra_args.strip() else []
    # Local directory: revision is ignored by local_sglang (omitted from subprocess).
    server = SglangSubprocessServer(
        model_path,
        args.revision,
        args.host,
        args.port,
        extra_argv=extra,
        startup_timeout_s=args.startup_timeout,
    )

    async def wait_and_probe() -> bool:
        async with aiohttp.ClientSession() as session:
            ok = await server.wait_until_ready(session)
            if ok and not args.no_probe:
                await _probe_v1_models(server.client_base_url)
            return ok

    logging.info("Starting subprocess (listen %s, clients use %s)", server.base_url, server.client_base_url)
    server.start()
    try:
        ok = asyncio.run(wait_and_probe())
        if not ok:
            logging.error("Server did not become ready; exiting.")
            sys.exit(1)
        logging.info("OpenAI-compatible base: %s", server.client_base_url)
        logging.info("Ctrl+C to stop.")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logging.info("Shutdown requested.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
