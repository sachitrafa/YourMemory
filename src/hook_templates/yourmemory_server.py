#!/usr/bin/env python3
"""yourmemory_server.py — platform-agnostic, session-scoped HTTP server control.

With MCP disabled, the recall (UserPromptSubmit) and store (Stop) hooks still need
the YourMemory HTTP server on http://127.0.0.1:3033. This controller starts it when a
Claude Code session opens and stops it when the last session closes — no MCP, no bash,
no launchd/systemd. Pure Python stdlib: works on Windows, macOS, and Linux.

Wire-up (settings.json):
  SessionStart → python3 .../yourmemory_server.py start
  SessionEnd   → python3 .../yourmemory_server.py stop

Lifecycle is reference-counted via one marker file per live session, so opening N
sessions starts the server once and it only shuts down when the Nth session ends.
A server we did NOT start (e.g. launched manually or by MCP) is never killed.

Env:
  YOURMEMORY_DASHBOARD_PORT  server port (default 3033)
  YOURMEMORY_PATH            state dir   (default ~/.yourmemory)
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT   = int(os.environ.get("YOURMEMORY_DASHBOARD_PORT", "3033"))
HOST   = "127.0.0.1"
HEALTH = f"http://{HOST}:{PORT}/health"

STATE     = Path(os.environ.get("YOURMEMORY_PATH") or (Path.home() / ".yourmemory"))
SESS_DIR  = STATE / "sessions"
PID_FILE  = STATE / "server.pid"
OWNED     = STATE / "server.owned"      # present only if THIS controller spawned the server
LOCK      = STATE / "server.spawn.lock"
LOG_FILE  = STATE / "server.log"

# Repo/package root so `src.app:app` is importable (works installed or from a checkout).
REPO_ROOT = str(Path(__file__).resolve().parents[2])

# /clear and compaction emit SessionEnd+SessionStart for the SAME session — those are
# not real closes, so we ignore them to avoid needless stop/start churn.
NON_TERMINAL = {"clear", "compact", "resume"}


def _stdin_json() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _server_up(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(HEALTH, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _wait_until_up(deadline: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < deadline:
        if _server_up():
            return True
        time.sleep(0.4)
    return False


def _acquire_spawn_lock():
    """Atomic single-spawner guard so two concurrent SessionStarts don't both boot uvicorn."""
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:  # reclaim a stale lock from a crashed starter
            if time.time() - LOCK.stat().st_mtime > 30:
                LOCK.unlink()
                return _acquire_spawn_lock()
        except Exception:
            pass
        return False


def _release_spawn_lock():
    try:
        LOCK.unlink()
    except Exception:
        pass


def _spawn_server() -> None:
    log = open(LOG_FILE, "ab")
    cmd = [sys.executable, "-m", "uvicorn", "src.app:app",
           "--host", HOST, "--port", str(PORT), "--log-level", "warning"]
    kwargs = dict(cwd=REPO_ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP → survives the hook process exit
        kwargs["creationflags"] = 0x00000008 | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # new session leader → outlives the hook
    proc = subprocess.Popen(cmd, **kwargs)
    PID_FILE.write_text(str(proc.pid))
    OWNED.write_text(str(proc.pid))


def _kill(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def start() -> None:
    SESS_DIR.mkdir(parents=True, exist_ok=True)
    data = _stdin_json()
    sid = (data.get("session_id") or f"pid-{os.getpid()}").replace("/", "_")
    (SESS_DIR / sid).write_text(str(int(time.time())))

    if _server_up():
        return
    if _acquire_spawn_lock():
        try:
            if not _server_up():     # re-check under the lock
                _spawn_server()
                _wait_until_up()
        finally:
            _release_spawn_lock()
    else:
        _wait_until_up()             # another SessionStart is booting it — just wait


def stop() -> None:
    data = _stdin_json()
    reason = (data.get("reason") or data.get("source") or "").strip().lower()
    if reason in NON_TERMINAL:
        return  # /clear, compact, resume — same session continues, leave the server alone

    sid = (data.get("session_id") or "").replace("/", "_")
    if sid:
        try:
            (SESS_DIR / sid).unlink()
        except Exception:
            pass

    remaining = list(SESS_DIR.glob("*")) if SESS_DIR.exists() else []
    if remaining:
        return  # other sessions still open — keep the server up

    # Last session closed. Only stop a server WE started.
    if OWNED.exists() and PID_FILE.exists():
        try:
            _kill(int(PID_FILE.read_text().strip()))
        except Exception:
            pass
        for f in (PID_FILE, OWNED):
            try:
                f.unlink()
            except Exception:
                pass


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "start":
        start()
    elif action == "stop":
        stop()
    # unknown action → no-op (never disrupt the session)


if __name__ == "__main__":
    main()
