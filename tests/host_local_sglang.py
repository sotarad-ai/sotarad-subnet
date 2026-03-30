"""
Start a vision/LLM model with the same ``local_sglang.SglangSubprocessServer``
wrapper the validator uses (``python -m sglang.launch_server`` + readiness poll).

Requires ``sglang`` in the active Python environment and a GPU (or whatever
your SGLang build expects).

Usage
-----
Local checkpoint directory (``--revision`` is omitted, same as validator)::

    python3 tests/host_local_sglang.py /path/to/model --host 0.0.0.0 --port 40035

Hugging Face repo id::

    python3 tests/host_local_sglang.py org/model-name --revision main --port 30000

Extra CLI flags for ``launch_server`` (same idea as ``--sglang-extra-args``)::

    python3 tests/host_local_sglang.py ./my-model \\
      --extra-args "--mem-fraction-static 0.9 --context-length 8192"

After “ready”, open another terminal and e.g.::

    python3 tests/test_model_request.py --image x.png --host 127.0.0.1 --port 40035

To **download** from Hugging Face first, then host the same way the validator uses a local dir, use
``tests/download_and_host_sglang.py``.
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "model_path",
        help="HF repo id or local directory with model weights (validator ``repo``).",
    )
    p.add_argument(
        "--revision",
        default="main",
        help="HF revision (ignored when model_path is a local directory). Default: main",
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address for sglang.launch_server (default 0.0.0.0).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=40035,
        help="Listen port (default 40035).",
    )
    p.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for HTTP readiness (default 600).",
    )
    p.add_argument(
        "--extra-args",
        default="",
        help='Extra arguments for launch_server, shell-style (e.g. \'--mem-fraction-static 0.9\').',
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    extra = shlex.split(args.extra_args.strip()) if args.extra_args.strip() else []

    server = SglangSubprocessServer(
        args.model_path,
        args.revision,
        args.host,
        args.port,
        extra_argv=extra,
        startup_timeout_s=args.startup_timeout,
    )

    async def wait_ready() -> bool:
        async with aiohttp.ClientSession() as session:
            return await server.wait_until_ready(session)

    logging.info("Starting subprocess (listen %s, clients use %s)", server.base_url, server.client_base_url)
    server.start()
    try:
        ok = asyncio.run(wait_ready())
        if not ok:
            logging.error("Server did not become ready; exiting.")
            sys.exit(1)
        logging.info("OpenAI-compatible base for requests: %s", server.client_base_url)
        logging.info("Example: POST %s/v1/chat/completions", server.client_base_url.rstrip("/"))
        logging.info("Ctrl+C to stop.")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logging.info("Shutdown requested.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
