#!/usr/bin/env python3
"""
yourmemory-run — opt-in headless "lean-window" mode
===================================================
Each turn runs a FRESH `claude -p` (the context window is flushed every time), and
YourMemory supplies continuity instead of carried history:

  • verbatim recent buffer  → injected ALWAYS (the last N exchanges, exact text)
  • distilled recalled facts → injected when they match the prompt (gated ≥0.5)

After each turn it stores the exchange two ways before the next flush:
  • /buffer-store  → keep it verbatim (rolling cap N) so the flush stays lossless
  • /auto-store    → qwen distills it into long-term facts

This does NOT change interactive Claude Code — it's a separate launcher you opt into.
The inner `claude -p` runs under a void namespace so the installed recall/store hooks
fire harmlessly (no double recall/store) without being disabled.

Usage:
  yourmemory-run                 # interactive REPL, lean window
  yourmemory-run "your prompt"   # one-shot
Env: YOURMEMORY_USER, YOURMEMORY_BUFFER (default 3), YOURMEMORY_TOPK (6),
     YOURMEMORY_MODEL (optional), YOURMEMORY_URL (http://localhost:3033)
"""
import getpass
import json
import os
import subprocess
import sys
import urllib.request

URL      = os.getenv("YOURMEMORY_URL", "http://localhost:3033")
N_BUFFER = max(1, min(int(os.getenv("YOURMEMORY_BUFFER", "3")), 20))
TOPK     = int(os.getenv("YOURMEMORY_TOPK", "6"))
MODEL    = os.getenv("YOURMEMORY_MODEL", "").strip()
VOID_NS  = "ym_run_void"   # inner claude -p hooks query this empty ns → no double recall/store


def resolve_user() -> str:
    uid = os.getenv("YOURMEMORY_USER", "").strip()
    if not uid:
        p = os.path.expanduser("~/.yourmemory/user_id")
        if os.path.exists(p):
            try:
                uid = open(p).read().strip()
            except Exception:
                uid = ""
    if not uid:
        try:
            uid = getpass.getuser()
        except Exception:
            uid = "user"
    return uid.lower()


def _post(path, body, timeout=60):
    req = urllib.request.Request(f"{URL}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path, timeout=30):
    with urllib.request.urlopen(f"{URL}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def build_context(user: str, prompt: str) -> str:
    parts = []
    try:
        buf = _get(f"/buffer?userId={user}&n={N_BUFFER}").get("buffer", [])
    except Exception:
        buf = []
    if buf:
        parts.append("[Recent conversation — verbatim]")
        for ex in buf:
            if ex.get("user_text"):
                parts.append(f"User: {ex['user_text']}")
            if ex.get("assistant_text"):
                parts.append(f"Assistant: {ex['assistant_text']}")
    try:
        r = _post("/retrieve", {"userId": user, "query": prompt, "topK": TOPK})
        facts = [m["content"] for m in r.get("memories", [])
                 if m.get("score", 0) >= 0.5 and m.get("similarity", 0) >= 0.5]
    except Exception:
        facts = []
    if facts:
        parts.append("\n[Recalled facts]")
        parts += [f"- {f}" for f in facts]
    return "\n".join(parts)


def run_turn(user: str, prompt: str) -> str:
    ctx = build_context(user, prompt)
    system = ""
    if ctx:
        system = ("You are continuing an ongoing session. The block below is your memory of it — "
                  "recent conversation verbatim plus recalled facts. Use it as if you remember it, "
                  "and don't ask the user to repeat things it already contains.\n\n" + ctx)

    env = dict(os.environ)
    env["YOURMEMORY_USER"] = VOID_NS          # inner hooks no-op; this launcher owns recall/store
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--max-turns", "1"]
    if system:
        cmd += ["--system-prompt", system]
    if MODEL:
        cmd += ["--model", MODEL]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        resp = (json.loads(p.stdout).get("result", "") or "").strip()
    except Exception as e:
        resp = f"[yourmemory-run error: {str(e)[:120]}]"

    # Persist BEFORE the next flush: verbatim buffer (lossless) + distilled facts (long-term)
    try:
        _post("/buffer-store", {"user_id": user, "user_text": prompt,
                                "assistant_text": resp, "keep": N_BUFFER})
    except Exception:
        pass
    try:
        _post("/auto-store", {"user_id": user, "user_text": prompt, "assistant_text": resp})
    except Exception:
        pass
    return resp


def main():
    user = resolve_user()
    try:
        _get("/health", timeout=4)
    except Exception:
        print(f"YourMemory server not reachable at {URL} — start it first.", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:                       # one-shot
        print(run_turn(user, " ".join(sys.argv[1:])))
        return

    print(f"yourmemory-run — lean-window mode (user={user}, buffer={N_BUFFER}, "
          f"flush every turn). Ctrl-D to exit.")
    while True:
        try:
            prompt = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        print(run_turn(user, prompt))


if __name__ == "__main__":
    main()
