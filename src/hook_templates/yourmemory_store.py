#!/usr/bin/env python3
"""Stop hook: POST new exchanges in the session to /auto-store.

Two behaviours that matter for cost/quality:

1. user_id is resolved per-machine (env → ~/.yourmemory/user_id → system login
   name), never hardcoded — so each OS user gets their own memory namespace.

2. The Stop hook fires after *every* assistant turn, not just at session end.
   Re-extracting the whole transcript each time is O(n^2) LLM calls and produces
   duplicate memories. We persist a per-session cursor in
   ~/.yourmemory/store_state/<session_id>.json and only post exchanges we have
   not processed before.
"""

import glob
import json
import os
import sys
import urllib.request

STATE_DIR = os.path.expanduser("~/.yourmemory/store_state")


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
        with urllib.request.urlopen(req, timeout=25) as resp:
            return json.loads(resp.read())
    except Exception:
        return {"stored": 0}


def main():
    try:
        data       = json.load(sys.stdin)
        session_id = data.get("session_id", "")
    except Exception:
        return

    if not session_id:
        return

    # Check server is up
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

    total_stored = 0
    for exchange in new_exchanges:
        result = call_auto_store(exchange, user_id=user_id)
        total_stored += result.get("stored", 0)

    save_cursor(session_id, len(exchanges))

    if total_stored > 0:
        print(json.dumps({"systemMessage":
            f"YourMemory: stored {total_stored} fact(s) from "
            f"{len(new_exchanges)} new exchange(s)"}))


if __name__ == "__main__":
    main()
