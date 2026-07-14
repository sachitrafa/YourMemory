"""
Regression tests for the Stop hook's transcript parser.

Guards the bug where every tool-using turn stored ZERO facts. A turn that calls a
tool emits several assistant text blocks — a short preamble ("I'll fetch that
repository..."), then tool_use, then the real answer. extract_all_exchanges paired
the user prompt with messages[i+1], i.e. the preamble, and dropped the answer. The
exchange posted to /auto-store was therefore ("summarize this repo", "I'll fetch
that repository and summarize it.") — nothing durable in it, so nothing was stored.

Fixed by joining every assistant block up to the next real user message.

Pure parsing — no DB, no server, no network.

Run:
    python tests/test_store_hook.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "hook_templates"))

from yourmemory_store import extract_all_exchanges

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results = []


def ok(name):
    print(f"  {PASS}  {name}")
    results.append((name, True))


def fail(name, reason=""):
    print(f"  {FAIL}  {name}" + (f"  ← {reason}" if reason else ""))
    results.append((name, False))


# ── Transcript builders (mirror Claude Code's .jsonl shape) ────────────────────

def user_msg(text):
    return {"type": "user", "message": {"content": [{"type": "text", "text": text}]}}


def assistant_msg(text):
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def assistant_tool_use(name="WebFetch"):
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": name, "input": {}}]}}


def tool_result(text="<result>...</result>"):
    # Tool results come back as type "user" with tool_result blocks and no text block.
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": text}]}}


def write_transcript(records):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 1. The regression: a tool-using turn must yield the ANSWER, not the preamble
# ═══════════════════════════════════════════════════════════════════════════════
print("\n1. Tool-using turn keeps the final answer")

PREAMBLE = "I'll fetch that repository and summarize it."
ANSWER = "It's a Java/Spring Boot sample: a car sales system split into microservices."

path = write_transcript([
    user_msg("please read and summarize https://github.com/Vilsium/DDD-in-Real-life-Microservices"),
    assistant_msg(PREAMBLE),
    assistant_tool_use(),
    tool_result(),
    assistant_msg(ANSWER),
])
exchanges = extract_all_exchanges(path)
os.remove(path)

if len(exchanges) == 1:
    ok("one exchange parsed from a tool-using turn")
else:
    fail("one exchange parsed from a tool-using turn", f"got {len(exchanges)}")

if exchanges and ANSWER in exchanges[0][1]:
    ok("assistant text contains the final answer")
else:
    got = exchanges[0][1][:60] if exchanges else "<none>"
    fail("assistant text contains the final answer", f"got {got!r}")

if exchanges and exchanges[0][1].strip() != PREAMBLE:
    ok("assistant text is not just the preamble")
else:
    fail("assistant text is not just the preamble", "preamble captured, answer lost")

if exchanges and "please read and summarize" in exchanges[0][0]:
    ok("user text is the prompt")
else:
    fail("user text is the prompt")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Tool results must not be mistaken for user turns
# ═══════════════════════════════════════════════════════════════════════════════
print("\n2. Tool results don't split the turn")

path = write_transcript([
    user_msg("do the thing"),
    assistant_msg("Working on it."),
    assistant_tool_use(),
    tool_result(),
    assistant_tool_use(),
    tool_result(),
    assistant_msg("Done — here is the result."),
])
exchanges = extract_all_exchanges(path)
os.remove(path)

if len(exchanges) == 1:
    ok("multi-tool turn stays a single exchange")
else:
    fail("multi-tool turn stays a single exchange", f"got {len(exchanges)}")

if exchanges and "Done — here is the result." in exchanges[0][1]:
    ok("answer survives multiple tool round-trips")
else:
    fail("answer survives multiple tool round-trips")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Plain turns and multi-turn sessions still pair correctly
# ═══════════════════════════════════════════════════════════════════════════════
print("\n3. Plain and multi-turn sessions")

path = write_transcript([
    user_msg("hi"),
    assistant_msg("hello"),
    user_msg("second question"),
    assistant_msg("second answer"),
])
exchanges = extract_all_exchanges(path)
os.remove(path)

if exchanges == [("hi", "hello"), ("second question", "second answer")]:
    ok("no-tool turns pair one-to-one")
else:
    fail("no-tool turns pair one-to-one", f"got {exchanges}")

# A trailing user message with no reply yet must not produce a half exchange.
path = write_transcript([
    user_msg("q1"),
    assistant_msg("a1"),
    user_msg("q2 still generating"),
])
exchanges = extract_all_exchanges(path)
os.remove(path)

if exchanges == [("q1", "a1")]:
    ok("unanswered trailing prompt is not emitted")
else:
    fail("unanswered trailing prompt is not emitted", f"got {exchanges}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Truncation keeps the tail — the answer, not the progress notes
# ═══════════════════════════════════════════════════════════════════════════════
print("\n4. Over-budget turns keep the answer")

TAIL = "FINAL ANSWER: the deliverable."
path = write_transcript([
    user_msg("long task"),
    assistant_msg("step one " * 900),   # blows past the 4000-char cap on its own
    assistant_tool_use(),
    tool_result(),
    assistant_msg(TAIL),
])
exchanges = extract_all_exchanges(path)
os.remove(path)

if exchanges and exchanges[0][1].endswith(TAIL):
    ok("truncation drops the preamble, not the final answer")
else:
    fail("truncation drops the preamble, not the final answer")

if exchanges and len(exchanges[0][1]) <= 4000:
    ok("assistant text respects the 4000-char cap")
else:
    got = len(exchanges[0][1]) if exchanges else 0
    fail("assistant text respects the 4000-char cap", f"got {got}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("═" * 65)
passed = sum(1 for _, p in results if p)
total = len(results)
print(f"Results: {passed}/{total} passed")

failed = [(n, p) for n, p in results if not p]
if failed:
    print("\nFailed tests:")
    for name, _ in failed:
        print(f"  {FAIL}  {name}")
    sys.exit(1)
else:
    print(f"\n{PASS} All tests passed.")
    sys.exit(0)
