"""
Start and stop a local SGLang OpenAI-compatible server via subprocess.

Uses ``python -m sglang.launch_server`` (same as the CLI) so no in-process
import of ``sglang`` is required. The validator polls until HTTP is ready, then
POSTs to ``{client_base}/v1/chat/completions`` like Chutes.

CLI alignment (example)::

    python3 -m sglang.launch_server \\
      --model-path ./models/MyModel/ --host 0.0.0.0 --port 40035 \\
      --mem-fraction-static 0.9

Validator: commit ``repo`` = model path (or HF id), ``chute_id`` empty,
``--allow-local``, ``--local-sglang-host 0.0.0.0``, ``--local-sglang-port 40035``,
``--sglang-extra-args "--mem-fraction-static 0.9"``. For a **local directory**,
``--revision`` is omitted (HF commits still pass ``--revision`` from chain).

Requires ``sglang`` installed in the same Python environment as the validator.
Run the validator from the same working directory as relative model paths, or
use an absolute ``repo`` in the commitment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from subprocess import Popen
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class SglangSubprocessServer:
    """Manage ``sglang.launch_server`` as a child process (new session / process group)."""

    def __init__(
        self,
        hf_repo: str,
        revision: str,
        host: str,
        port: int,
        *,
        extra_argv: Optional[list[str]] = None,
        startup_timeout_s: float = 600.0,
    ) -> None:
        self.hf_repo = hf_repo
        self.revision = revision
        self.host = host
        self.port = int(port)
        self.extra_argv = list(extra_argv or [])
        self.startup_timeout_s = float(startup_timeout_s)
        self._proc: Optional[Popen] = None

    @property
    def base_url(self) -> str:
        """URL matching ``--host`` (e.g. for logging)."""
        return f"http://{self.host}:{self.port}"

    @property
    def client_base_url(self) -> str:
        """
        Base URL for HTTP clients on the same machine.

        When ``--host`` is ``0.0.0.0`` (listen on all interfaces), clients must
        use ``127.0.0.1`` (or another routable address), not ``0.0.0.0``.
        """
        h = self.host.strip()
        if h in ("0.0.0.0", "::", "[::]"):
            client_host = "127.0.0.1"
        else:
            client_host = h
        return f"http://{client_host}:{self.port}"

    def _model_path_is_local_dir(self) -> bool:
        """Local checkpoint directory → omit ``--revision`` (matches a plain CLI launch)."""
        path = os.path.abspath(os.path.expanduser(self.hf_repo))
        return os.path.isdir(path)

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("SGLang server already started")
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.hf_repo,
        ]
        if not self._model_path_is_local_dir():
            cmd.extend(["--revision", self.revision])
        cmd.extend(
            [
                "--host",
                self.host,
                "--port",
                str(self.port),
            ]
        )
        cmd.extend(self.extra_argv)
        logger.info("Starting SGLang subprocess: %s", " ".join(cmd))
        # New session so we can signal the whole process group on shutdown.
        self._proc = Popen(
            cmd,
            stdin=Popen.DEVNULL,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )

    async def wait_until_ready(self, session: aiohttp.ClientSession) -> bool:
        deadline = time.monotonic() + self.startup_timeout_s
        base = self.client_base_url.rstrip("/")
        probe_urls = [f"{base}/health", f"{base}/v1/models"]
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                logger.error(
                    "SGLang process exited early with code %s",
                    self._proc.returncode,
                )
                return False
            for url in probe_urls:
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status < 500:
                            logger.info("SGLang server ready at %s", base)
                            return True
                except Exception:
                    pass
            await asyncio.sleep(2.0)
        logger.error("SGLang server did not become ready within %.0fs", self.startup_timeout_s)
        return False

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except ProcessLookupError:
                return
        try:
            proc.wait(timeout=60)
        except Exception:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        logger.info("SGLang subprocess stopped")
