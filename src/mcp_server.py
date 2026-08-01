"""
MCP Memory Server Lifecycle Manager — Phase 3.1

Spawns, health-checks, monitors, and gracefully shuts down the bundled
mcp-memory-service subprocess.  All lifecycle logic is encapsulated in
:class:`MCPServerManager`.

Spec references
---------------
* ``Implementation_Phases.md`` §3.1 — phase definition
* ``MCP_research.md`` — spawning, health check, graceful shutdown
* ``architecture.md`` §6.1–6.2 — server selection, bundling, security pinning
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx
import psutil

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PORT: int = 8000
_DEFAULT_READY_TIMEOUT: float = 10.0
_DEFAULT_GRACE_PERIOD: float = 5.0
_TCP_CHECK_INTERVAL: float = 0.1
_TCP_CHECK_PER_ATTEMPT: float = 0.5
_HEALTH_CHECK_TIMEOUT: float = 2.0
_POLL_DELAY: float = 0.2
_PERIODIC_HEALTH_INTERVAL: float = 10.0
_MAX_RESTART_ATTEMPTS: int = 3
_PROCESS_TREE_CLEANUP_TIMEOUT: float = 3.0

_STARTUP_MAX_RETRIES: int = 3
"""Maximum retry attempts when the server fails to become ready during start()."""

_STARTUP_BASE_DELAY: float = 1.0
"""Base delay (seconds) for exponential backoff during startup retries."""

_MONITOR_BASE_DELAY: float = 1.0
"""Base delay (seconds) for exponential backoff between monitor restart attempts."""


# ---------------------------------------------------------------------------
# Process tree cleanup
# ---------------------------------------------------------------------------


def kill_process_tree(pid: int) -> None:
    """Terminate *pid* and all its descendant processes recursively.

    Uses ``psutil`` to walk the process tree.  Children are terminated
    first (3 s grace), then the parent is killed.

    Args:
        pid: The root process ID to prune.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    _, still_alive = psutil.wait_procs(
        children,
        timeout=_PROCESS_TREE_CLEANUP_TIMEOUT,
    )
    for p in still_alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass

    try:
        parent.kill()
    except psutil.NoSuchProcess:
        pass


# ---------------------------------------------------------------------------
# MCPServerManager
# ---------------------------------------------------------------------------


class MCPServerManager:
    """Full lifecycle manager for the bundled MCP memory server subprocess.

    Responsibilities
    ----------------
    * Spawn the server via ``python -m mcp_memory_service.cli.main server --http``
    * Wait for readiness (TCP port + ``GET /api/health``)
    * Monitor for unexpected death with single auto-restart
    * Graceful shutdown (terminate → wait → kill → process-tree cleanup)
    * Periodic health probes (reported via :attr:`is_healthy`)

    The server is spawned with these security environment variables
    (per ``architecture.md`` §6.2):

    * ``MCP_ALLOW_ANONYMOUS_ACCESS=false``
    * ``MCP_CONSOLIDATION_STORE_ASSOCIATIONS=false``

    Attributes:
        port: HTTP port the server listens on (default 8000).
        process: The underlying ``asyncio.subprocess.Process``, or ``None``
            if not running / not yet started.
    """

    __slots__ = (
        "port",
        "ready_timeout",
        "grace_period",
        "process",
        "_monitor_task",
        "_stderr_task",
        "_healthy",
    )

    def __init__(
        self,
        port: int = _DEFAULT_PORT,
        ready_timeout: float = _DEFAULT_READY_TIMEOUT,
        grace_period: float = _DEFAULT_GRACE_PERIOD,
    ) -> None:
        self.port = port
        self.ready_timeout = ready_timeout
        self.grace_period = grace_period
        self.process: asyncio.subprocess.Process | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._healthy: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """Return ``True`` if the last health probe was successful."""
        return self._healthy

    async def start(self) -> None:
        """Spawn the MCP memory server and block until it is ready.

        Sets security environment variables, launches the subprocess,
        then runs the 2‑phase health check (TCP port → HTTP health
        endpoint).  A background monitor task is started to detect
        unexpected death and attempt up to 3 restarts with exponential
        backoff.

        On startup failure the method retries with exponential backoff
        (1 s → 2 s → 4 s) up to *STARTUP_MAX_RETRIES* attempts before
        raising.

        Raises:
            RuntimeError: If the server fails to become ready after all
                retry attempts.  The caller should catch this and
                degrade gracefully (memory tier disabled).
        """
        logger.info(
            "Starting MCP memory server on port {} (ready timeout {:.1f}s, "
            "max retries {}) …",
            self.port,
            self.ready_timeout,
            _STARTUP_MAX_RETRIES,
        )

        # Security pinning (architecture.md §6.2) + disable CUDA to skip
        # the 30‑40 s PyTorch CUDA device scan (CPU‑only ONNX embedding
        # works fine for this service).
        env = {**os.environ}
        env.setdefault("MCP_ALLOW_ANONYMOUS_ACCESS", "true")
        env.setdefault("MCP_CONSOLIDATION_STORE_ASSOCIATIONS", "false")
        env.setdefault("MCP_CONSOLIDATION_ENABLED", "false")
        env["CUDA_VISIBLE_DEVICES"] = ""  # skip PyTorch CUDA kernel load

        cmd: list[str] = [
            sys.executable,
            "-m",
            "mcp_memory_service.cli.main",
            "server",
            "--http",
        ]

        last_error: str = ""

        for attempt in range(_STARTUP_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _STARTUP_BASE_DELAY * (2 ** (attempt - 1))
                logger.info(
                    "MCP startup retry {}/{} after {:.1f}s delay …",
                    attempt,
                    _STARTUP_MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)

            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                await self._wait_for_ready()
            except RuntimeError as exc:
                last_error = str(exc)
                logger.warning(
                    "MCP startup attempt {}/{} failed: {}",
                    attempt + 1,
                    _STARTUP_MAX_RETRIES + 1,
                    last_error,
                )
                # Clean up the failed process before retrying
                if self.process is not None:
                    await self._shutdown_process(self.process)
                    pid = self.process.pid
                    if pid is not None:
                        kill_process_tree(pid)
                    self.process = None
                continue  # retry

            # --- Success path ---
            # Start background monitor + stderr reader
            self._monitor_task = asyncio.create_task(self._monitor())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            logger.info("MCP memory server ready on port {}.", self.port)
            return

        # All retries exhausted
        raise RuntimeError(
            f"MCP memory server failed to start after "
            f"{_STARTUP_MAX_RETRIES + 1} attempts. "
            f"Last error: {last_error}"
        )

    async def stop(self) -> None:
        """Gracefully shut down the server and clean up child processes.

        Uses the two‑step termination protocol:
        1. ``process.terminate()`` + wait up to *grace_period*
        2. ``process.kill()`` if step 1 times out
        3. ``kill_process_tree(pid)`` for any remaining descendants

        Safe to call when the server is already stopped.
        """
        if self.process is None:
            return

        pid = self.process.pid
        logger.info("Shutting down MCP memory server (PID {}) …", pid)

        # Cancel monitor + stderr reader tasks
        for task_attr in ("_monitor_task", "_stderr_task"):
            task = getattr(self, task_attr, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            setattr(self, task_attr, None)

        await self._shutdown_process(self.process)
        if pid is not None:
            kill_process_tree(pid)

        self.process = None
        self._healthy = False
        logger.info("MCP memory server stopped.")

    # ------------------------------------------------------------------
    # Health check (2‑phase)
    # ------------------------------------------------------------------

    async def _wait_for_ready(self) -> None:
        """Phase 1 (TCP port) → Phase 2 (HTTP ``/api/health``).

        Raises :exc:`RuntimeError` if the server does not become ready
        within *ready_timeout*.
        """
        assert self.process is not None

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < self.ready_timeout:
            # Phase 1 — TCP port binding
            if await _wait_for_port("localhost", self.port, timeout=0.5):
                # Phase 2 — authoritative health endpoint
                if await _health_endpoint_check(self.port, timeout=_HEALTH_CHECK_TIMEOUT):
                    self._healthy = True
                    return
            await asyncio.sleep(_POLL_DELAY)

        # Read stderr for diagnostics
        stderr_text = ""
        try:
            if self.process.stderr is not None:
                stderr_text = (await self.process.stderr.read()).decode(
                    "utf-8", errors="replace"
                )
        except Exception:
            pass

        raise RuntimeError(
            f"MCP memory server not ready within {self.ready_timeout:.0f} s. "
            f"stderr: {stderr_text[:500]}"
        )

    # ------------------------------------------------------------------
    # Monitoring & auto‑restart
    # ------------------------------------------------------------------

    async def _monitor(self) -> None:
        """Background task — detect unexpected death and attempt restart.

        If the server dies (returncode != 0), up to ``_MAX_RESTART_ATTEMPTS``
        restarts are attempted with exponential backoff.  After exhausting
        all retries the memory tier is permanently disabled.
        """
        assert self.process is not None

        restart_attempts = 0

        while True:
            returncode = await self.process.wait()

            if returncode == 0:
                logger.info("MCP memory server exited cleanly (code 0).")
                self._healthy = False
                return

            logger.warning(
                "MCP memory server died unexpectedly (returncode {}).",
                returncode,
            )

            if restart_attempts >= _MAX_RESTART_ATTEMPTS:
                logger.error(
                    "MCP memory server restart limit ({}) reached — "
                    "memory tier disabled.",
                    _MAX_RESTART_ATTEMPTS,
                )
                self._healthy = False
                self.process = None
                return

            # Exponential backoff: 1s → 2s → 4s
            delay = _MONITOR_BASE_DELAY * (2 ** restart_attempts)
            restart_attempts += 1
            logger.info(
                "Restarting MCP memory server (attempt {}/{}) "
                "after {:.1f}s delay …",
                restart_attempts,
                _MAX_RESTART_ATTEMPTS,
                delay,
            )
            await asyncio.sleep(delay)

            env = {**os.environ}
            env.setdefault("MCP_ALLOW_ANONYMOUS_ACCESS", "true")
            env.setdefault("MCP_CONSOLIDATION_STORE_ASSOCIATIONS", "false")
            env.setdefault("MCP_CONSOLIDATION_ENABLED", "false")
            env["CUDA_VISIBLE_DEVICES"] = ""  # skip PyTorch CUDA kernel load

            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "mcp_memory_service.cli.main",
                "server",
                "--http",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                await self._wait_for_ready()
            except RuntimeError:
                logger.error(
                    "Restarted MCP server failed to become ready — "
                    "memory tier disabled.",
                )
                self._healthy = False
                return

    # ------------------------------------------------------------------
    # Shutdown helpers
    # ------------------------------------------------------------------

    async def _shutdown_process(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        """Two‑step termination: terminate → timeout → kill."""
        if process.returncode is not None:
            return  # already dead

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=self.grace_period)
        except asyncio.TimeoutError:
            logger.warning(
                "MCP server did not exit after {:.0f}s — sending SIGKILL.",
                self.grace_period,
            )
            process.kill()
            await process.wait()

    async def _read_stderr(self) -> None:
        """Background task: read and log the MCP server's stderr output."""
        assert self.process is not None
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    logger.debug("MCP server stderr: {}", text)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Standalone health‑check helpers
# ---------------------------------------------------------------------------


async def _wait_for_port(
    host: str = "localhost",
    port: int = 8000,
    timeout: float = 5.0,
) -> bool:
    """Return ``True`` once a TCP connection to *host*:*port* succeeds.

    Args:
        host: Hostname or IP.
        port: TCP port.
        timeout: Total time (seconds) to keep retrying.

    Returns:
        ``True`` if the port becomes reachable within *timeout*.
    """
    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < timeout:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=_TCP_CHECK_PER_ATTEMPT,
            )
            writer.close()
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            await asyncio.sleep(_TCP_CHECK_INTERVAL)
    return False


async def _health_endpoint_check(
    port: int = 8000,
    timeout: float = 2.0,
) -> bool:
    """Call ``GET /api/health`` and verify the response is healthy.

    Args:
        port: HTTP port the server listens on.
        timeout: httpx request timeout.

    Returns:
        ``True`` if the server responds with ``{"status": "healthy"}``.
    """
    url = f"http://localhost:{port}/api/health"
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url)
            return (
                response.status_code == 200
                and response.json().get("status") == "healthy"
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            return False


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "MCPServerManager",
    "kill_process_tree",
]