#!/usr/bin/env python3
"""SessionStart hook: inject a "where we left off" digest.

After a context compaction (or on resume/startup), Claude loses the working context and
would otherwise re-read files to re-orient. This hands the agent YourMemory's recent
memories as additionalContext — so it starts with "what we've been working on" instead of
reconstructing it. This is the post-compaction re-orientation that pairs with the
per-prompt recall hook.

Fires on SessionStart with source in {startup, resume, clear, compact}:
  • compact  → re-inject the work context the compaction dropped
  • resume   → restore where the previous session left off
  • startup  → cross-session continuity ("welcome back")

Pure stdlib; platform-agnostic (Windows/macOS/Linux). No bash/jq/curl.
"""

import json
import os
import sys
import urllib.parse
import urllib.request

URL = os.getenv("YOURMEMORY_URL", "http://localhost:3033")
N   = int(os.getenv("YOURMEMORY_SESSION_CONTEXT_N", "12"))


def resolve_user_id() -> str:
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


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}
    source = (data.get("source") or "").lower()

    # Server must be up (the SessionStart server hook starts it first). /memories is a
    # plain SQL read — no embedding model load — so this stays fast even on a cold server.
    try:
        urllib.request.urlopen(f"{URL}/health", timeout=2)
    except Exception:
        return

    user_id = resolve_user_id()
    try:
        url = f"{URL}/memories?userId={urllib.parse.quote(user_id)}&limit={N}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            mems = json.loads(resp.read()).get("memories", [])
    except Exception:
        return

    lines = [f"- {m['content']}" for m in mems[:N] if m.get("content")]
    if not lines:
        return

    label = {
        "compact": "recalled after compaction",
        "resume":  "from where you left off",
        "clear":   "carried over",
    }.get(source, "from your recent sessions")

    context = (f"[YourMemory — context {label}]\n"
               "Recent memory of what this user/project has been working on:\n"
               + "\n".join(lines))

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }))


if __name__ == "__main__":
    main()
