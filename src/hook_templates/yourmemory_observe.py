#!/usr/bin/env python3
"""PostToolUse hook: capture substantial tool outputs (file reads, command output, search
results) into YourMemory via /observe — so the model can later answer from memory instead
of re-reading. This is the Claude Code adapter for the proxy's tool-output capture.

Registered with "async": true so it runs in the background and NEVER blocks the tool call
(the /observe extraction uses the local LLM and takes a few seconds).
"""
import getpass
import json
import os
import sys
import urllib.request

CAPTURE_TOOLS = {"Read", "Bash", "Grep", "Glob", "WebFetch"}
MIN_CHARS     = 800



def _ym_base():
    import os
    return os.getenv("YOURMEMORY_URL", "http://localhost:3033").rstrip("/")


def _ym_headers():
    import os
    h = {"Content-Type": "application/json"}
    k = os.getenv("YOURMEMORY_API_KEY", "").strip()
    if k:
        h["Authorization"] = "Bearer " + k
    return h

def resolve_user_id():
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
            uid = getpass.getuser()
        except Exception:
            uid = "user"
    return uid.lower()


def extract_text(tool_response):
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        for k in ("content", "output", "stdout", "result", "text", "stderr"):
            v = tool_response.get(k)
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, list):
                joined = " ".join(b.get("text", "") for b in v if isinstance(b, dict))
                if joined.strip():
                    return joined
        return json.dumps(tool_response)
    return str(tool_response)


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    if data.get("tool_name", "") not in CAPTURE_TOOLS:
        return

    text = extract_text(data.get("tool_response", {}))
    if not text or len(text) < MIN_CHARS:
        return

    src = ""
    ti = data.get("tool_input", {})
    if isinstance(ti, dict):
        src = (ti.get("file_path") or ti.get("command") or ti.get("pattern") or "")[:120]

    try:
        urllib.request.urlopen(_ym_base() + "/health", timeout=2)
    except Exception:
        return

    payload = json.dumps({
        "text":    text[:8000],
        "user_id": resolve_user_id(),
        "source":  src,
    }).encode()
    try:
        req = urllib.request.Request(_ym_base() + "/observe", data=payload,
                                     headers=_ym_headers())
        urllib.request.urlopen(req, timeout=90)
    except Exception:
        pass


if __name__ == "__main__":
    main()
