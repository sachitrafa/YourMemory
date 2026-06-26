#!/usr/bin/env python3
"""Stop hook: POST new exchanges in the session to /auto-store.

Three behaviours that matter for cost/quality/latency:

1. user_id is resolved per-machine (env → ~/.yourmemory/user_id → system login
   name), never hardcoded — so each OS user gets their own memory namespace.

2. The Stop hook fires after *every* assistant turn, not just at session end.
   Re-extracting the whole transcript each time is O(n^2) LLM calls and produces
   duplicate memories. We persist a per-session cursor in
   ~/.yourmemory/store_state/<session_id>.json and only post exchanges we have
   not processed before.

3. FIRE-AND-FORGET: the actual /auto-store calls run Ollama fact-extraction, which
   is seconds-slow. Blocking the hook on that adds that latency to the *end of every
   turn*. So the foreground does only the fast work (read transcript, diff against
   the cursor, advance it) then hands the slow extraction to a DETACHED background
   worker and returns immediately. The turn completes in milliseconds; storage
   catches up a few seconds later. Pure stdlib; platform-agnostic.
"""

import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

STATE_DIR   = os.path.expanduser("~/.yourmemory/store_state")
PENDING_DIR = os.path.join(STATE_DIR, "pending")


def resolve_user_id():
    """env → ~/.yourmemory/user_id → system login name, lowercased."""
    uid = os.getenv("YOURMEMORY_USER", "").strip()
    if not uid:
        path = os.path.expanduser("~/.yourmemory/user_id")
        if os.path.exists(path):
            try:
                uid = open(path).read().strip()
            except Exception:
                uid = ""
    if not uid:
        try:
            import getpass
            uid = getpass.getuser()
        except Exception:
            uid = "user"
    return uid.lower()


def find_transcript(session_id):
    pattern = os.path.expanduser(f"~/.claude/projects/**/{session_id}.jsonl")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def extract_all_exchanges(transcript_path):
    """Return all (user_text, assistant_text) pairs from the transcript."""
    try:
        with open(transcript_path) as f:
            lines = f.readlines()
    except Exception:
        return []

    messages = []
    for line in lines:
        try:
            d        = json.loads(line)
            msg_type = d.get("type")
            content  = d.get("message", {}).get("content", [])

            if isinstance(content, list):
                text = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ).strip()
            else:
                text = str(content).strip()

            if text and msg_type in ("user", "assistant"):
                messages.append((msg_type, text))
        except Exception:
            pass

    # Pair up user → assistant exchanges
    exchanges = []
    i = 0
    while i < len(messages):
        if messages[i][0] == "user":
            user_text = messages[i][1]
            if i + 1 < len(messages) and messages[i + 1][0] == "assistant":
                assistant_text = messages[i + 1][1]
                # Generous limits — pricing and the deliverable live in the longer
                # assistant output; a tight cut used to truncate them before extraction.
                exchanges.append((user_text[:2000], assistant_text[:4000]))
                i += 2
            else:
                i += 1
        else:
            i += 1

    return exchanges


def load_cursor(session_id):
    """How many exchanges of this session we have already posted."""
    path = os.path.join(STATE_DIR, f"{session_id}.json")
    try:
        return json.load(open(path)).get("processed", 0)
    except Exception:
        return 0


def save_cursor(session_id, processed):
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, f"{session_id}.json")
    try:
        with open(path, "w") as f:
            json.dump({"processed": processed}, f)
    except Exception:
        pass


def call_auto_store(exchange, user_id):
    user_text, assistant_text = exchange
    payload = json.dumps({
        "user_text":      user_text,
        "assistant_text": assistant_text,
        "user_id":        user_id,
    }).encode()

    try:
        req = urllib.request.Request(
            "http://localhost:3033/auto-store",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except Exception:
        return {"stored": 0}


def _spawn_worker(payload_path: str) -> None:
    """Detach a background process to run the slow /auto-store calls, so the Stop
    hook returns immediately and never adds extraction latency to the turn."""
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", payload_path]
    kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                  stdin=subprocess.DEVNULL)
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True   # survives the hook process exit
    subprocess.Popen(cmd, **kwargs)


def worker_main(payload_path: str) -> None:
    """Background entrypoint: read the queued exchanges, store them, delete the file."""
    try:
        with open(payload_path) as f:
            payload = json.load(f)
    except Exception:
        return
    finally:
        try:
            os.remove(payload_path)
        except Exception:
            pass
    user_id   = payload.get("user_id", "user")
    exchanges = payload.get("exchanges", [])
    for ex in exchanges:
        call_auto_store(tuple(ex), user_id=user_id)


def main():
    # Background worker re-invocation (fire-and-forget storage).
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2])
        return

    try:
        data       = json.load(sys.stdin)
        session_id = data.get("session_id", "")
    except Exception:
        return

    if not session_id:
        return

    # Check server is up (fast — milliseconds when running). If it's down, do nothing
    # rather than queue a worker that would only fail.
    try:
        urllib.request.urlopen("http://localhost:3033/health", timeout=2)
    except Exception:
        return

    transcript_path = find_transcript(session_id)
    if not transcript_path:
        return

    exchanges = extract_all_exchanges(transcript_path)
    if not exchanges:
        return

    # Only process exchanges added since the last Stop firing for this session.
    already = load_cursor(session_id)
    new_exchanges = exchanges[already:]
    if not new_exchanges:
        return

    user_id = resolve_user_id()

    # Advance the cursor now (optimistic) so the next Stop doesn't re-queue these,
    # then hand the slow extraction to a detached worker and return immediately.
    save_cursor(session_id, len(exchanges))

    try:
        os.makedirs(PENDING_DIR, exist_ok=True)
        payload_path = os.path.join(PENDING_DIR, f"{session_id}-{int(time.time()*1000)}.json")
        with open(payload_path, "w") as f:
            json.dump({"user_id": user_id, "exchanges": new_exchanges}, f)
        _spawn_worker(payload_path)
    except Exception:
        # If queueing/spawn fails, fall back to inline storage so nothing is lost.
        for exchange in new_exchanges:
            call_auto_store(exchange, user_id=user_id)


if __name__ == "__main__":
    main()
