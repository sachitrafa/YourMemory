#!/usr/bin/env python3
"""UserPromptSubmit hook: recall relevant memories before every Claude prompt.

Reads the user's prompt from stdin JSON, queries YourMemory, and injects
matching memories as additionalContext — no bash, jq, or curl required.
Works on Windows, macOS, and Linux.
"""

import json
import os
import sys
import urllib.request


def resolve_user_id() -> str:
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


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return

    query = (data.get("prompt") or "").strip()
    if len(query) < 30:
        return

    # Check server is up before making a recall request
    try:
        urllib.request.urlopen("http://localhost:3033/health", timeout=2)
    except Exception:
        return

    user_id = resolve_user_id()

    payload = json.dumps({
        "userId": user_id,
        "query":  query,
        "topK":   6,
        "expandK": 4,
    }).encode()

    try:
        req = urllib.request.Request(
            "http://localhost:3033/retrieve",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
    except Exception:
        return

    memories = result.get("memories") or []

    # Direct hits must clear the similarity threshold. Graph-expanded (linked) memories
    # used to be injected unconditionally — that dragged in low-relevance noise from
    # co-occurrence edges (anything stored in the same session). Now a linked memory must
    # also clear a modest score floor, and we cap how many graph hits ride along, so the
    # connected region surfaces useful neighbours without flooding the prompt.
    MIN_GRAPH_SCORE = 0.30
    MAX_GRAPH       = 3

    direct, graph = [], []
    for m in memories:
        content = m.get("content")
        if not content:
            continue
        if m.get("score", 0) >= 0.5 and m.get("similarity", 0) >= 0.50:
            direct.append(content)
        elif m.get("via_graph") and m.get("score", 0) >= MIN_GRAPH_SCORE:
            graph.append((m.get("score", 0), content))

    graph.sort(key=lambda x: x[0], reverse=True)
    lines = [f"- {c}" for c in direct] + [f"- {c}" for _, c in graph[:MAX_GRAPH]]

    if not lines:
        return

    context = "[YourMemory — recalled context]\n" + "\n".join(lines)

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }))


if __name__ == "__main__":
    main()
