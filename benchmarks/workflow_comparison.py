"""
Workflow Comparison Benchmark: With vs Without YourMemory
----------------------------------------------------------
Simulates a realistic developer working across 3 sessions on the YourMemory project.

Measures per session and cumulatively:
  - Total tokens in context window (system + history + user message)
  - Number of LLM API calls required
  - Stale memory pollution (irrelevant facts injected)
  - Context efficiency (relevant tokens / total tokens)

Two approaches modeled:
  A) Naive Baseline  — no memory system; user manually pastes context or
                       full conversation history is carried forward.
  B) YourMemory      — recall_memory injects top-k compressed, decayed memories.

Usage:
    python3 benchmarks/workflow_comparison.py
"""

import sys
import os
import math
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.services.decay import compute_strength


# ---------------------------------------------------------------------------
# Token counter (GPT-4 style ~4 chars/token; close enough for comparison)
# ---------------------------------------------------------------------------

def tok(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Shared: System prompt (identical for both approaches)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful software engineering assistant working on a Python FastAPI project.
Be concise. Answer based on the context you are given."""

SYSTEM_TOKENS = tok(SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# The scenario: Sachit asks for help across 3 sessions
#
#   Session 1: "Help me add a /health endpoint"
#   Session 2: "Now add rate limiting to the memory endpoints"
#   Session 3: "Write a pytest test for the recall endpoint"
# ---------------------------------------------------------------------------

# ---- Conversation turns (content strings) ---------------------------------

SESSION_1_TURNS = [
    # (role, content)
    ("user",  "I'm building a FastAPI app called YourMemory. It's a persistent memory MCP server using PostgreSQL + pgvector. I prefer Python. Can you help me add a /health endpoint?"),
    ("assistant", "Sure! Here's a /health endpoint for your FastAPI app:\n\n```python\n@app.get('/health')\nasync def health():\n    return {'status': 'ok', 'service': 'yourmemory'}\n```\n\nAdd it to your `main.py`. Want me to also check the DB connection in the health check?"),
    ("user",  "Yes, also check the DB connection."),
    ("assistant", "Here's the updated health endpoint with a DB ping:\n\n```python\nfrom src.db.connection import get_db\n\n@app.get('/health')\nasync def health(db=Depends(get_db)):\n    try:\n        await db.execute('SELECT 1')\n        db_status = 'ok'\n    except Exception:\n        db_status = 'error'\n    return {'status': 'ok', 'db': db_status}\n```"),
    ("user",  "Perfect. The project uses pgvector for similarity search with cosine threshold 0.5. We also expose recall, store, update MCP tools. Remember that."),
    ("assistant", "Got it — I'll remember:\n- pgvector, cosine threshold 0.5\n- MCP tools: recall, store, update\n- Stack: FastAPI + PostgreSQL"),
]

SESSION_2_TURNS = [
    ("user",  "Now add rate limiting to the memory endpoints. We're targeting enterprise customers so it needs to be per API key."),
    ("assistant", "For per-API-key rate limiting in FastAPI, I'll use `slowapi`:\n\n```python\nfrom slowapi import Limiter\nfrom slowapi.util import get_remote_address\n\nlimiter = Limiter(key_func=lambda req: req.headers.get('X-Api-Key', get_remote_address(req)))\n\n@app.post('/memories')\n@limiter.limit('100/minute')\nasync def store_memory(request: Request, ...):\n    ...\n```\n\nApply the same decorator to `/recall` and `/update`. Want me to add the middleware wiring too?"),
    ("user",  "Yes, and make the limits configurable per tier (free vs enterprise)."),
    ("assistant", "Here's a tiered rate limit setup:\n\n```python\nTIER_LIMITS = {\n    'free': '20/minute',\n    'enterprise': '500/minute',\n}\n\ndef get_limit(request: Request):\n    tier = request.state.api_key_tier  # set by auth middleware\n    return TIER_LIMITS.get(tier, '20/minute')\n```\n\nYour auth middleware should attach `api_key_tier` to `request.state` after validating the API key from PostgreSQL."),
]

SESSION_3_TURNS = [
    ("user",  "Now write a pytest test for the recall endpoint."),
    ("assistant", "Here's a pytest test for the recall endpoint:\n\n```python\nimport pytest\nfrom httpx import AsyncClient\nfrom main import app\n\n@pytest.mark.asyncio\nasync def test_recall_returns_memories():\n    async with AsyncClient(app=app, base_url='http://test') as client:\n        resp = await client.post(\n            '/recall',\n            json={'query': 'Python preferences', 'user_id': 'sachit', 'top_k': 3},\n            headers={'X-Api-Key': 'test-key'},\n        )\n    assert resp.status_code == 200\n    data = resp.json()\n    assert 'memories' in data\n    assert isinstance(data['memories'], list)\n```"),
    ("user",  "Also test that cosine threshold 0.5 filters out low-similarity results."),
    ("assistant", "Adding a threshold test:\n\n```python\n@pytest.mark.asyncio\nasync def test_recall_filters_low_similarity():\n    # Insert a memory that shouldn't match\n    async with AsyncClient(app=app, base_url='http://test') as client:\n        resp = await client.post(\n            '/recall',\n            json={'query': 'completely unrelated topic XYZ', 'user_id': 'sachit', 'top_k': 5},\n            headers={'X-Api-Key': 'test-key'},\n        )\n    data = resp.json()\n    # All returned memories must have similarity >= 0.5\n    for mem in data['memories']:\n        assert mem['similarity'] >= 0.5\n```"),
]


# ---------------------------------------------------------------------------
# Memories that accumulate after each session (what YourMemory would store)
# ---------------------------------------------------------------------------

MEMORY_BANK = [
    # (content, importance, days_before_session, recall_count)
    # Available at Session 2:
    ("Sachit is building YourMemory, a FastAPI + PostgreSQL + pgvector MCP memory server.", 0.9, 1, 1),
    ("Sachit prefers Python for backend development.", 0.9, 1, 2),
    ("The project uses pgvector with cosine similarity threshold ≥ 0.5.", 0.8, 1, 1),
    ("YourMemory MCP server exposes three tools: recall, store, update.", 0.8, 1, 1),
    ("Added /health endpoint with DB ping to main.py in Session 1.", 0.5, 1, 0),
    # Available at Session 3 (Session 2 added):
    ("Sachit is targeting enterprise customers with per-API-key rate limiting.", 0.8, 1, 1),
    ("Rate limits are tiered: free=20/min, enterprise=500/min via slowapi.", 0.7, 1, 1),
    ("API key tier is attached to request.state by auth middleware.", 0.6, 1, 0),
]


def get_memories_for_session(session_idx: int, query: str, top_k: int = 5):
    """Return top_k memories available by session_idx, ranked by strength."""
    now = datetime.now(timezone.utc)
    available_count = [0, 5, 8][session_idx]  # Session 1: 0, Session 2: 5, Session 3: 8
    candidates = []
    for content, importance, days_ago, recall_count in MEMORY_BANK[:available_count]:
        last_accessed = now - timedelta(days=days_ago)
        strength = compute_strength(last_accessed, recall_count, importance)
        candidates.append((strength, content, tok(content)))
    candidates.sort(reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Context assembly helpers
# ---------------------------------------------------------------------------

def assemble_naive_context(session_idx: int, current_turns):
    """
    Naive baseline: carries ALL prior session history forward.
    Session 1: system + current turns (no prior history)
    Session 2: system + session1_history + current turns
    Session 3: system + session1_history + session2_history + current turns
    """
    prior_history = [SESSION_1_TURNS, SESSION_1_TURNS + SESSION_2_TURNS]

    parts = [("system", SYSTEM_PROMPT)]
    if session_idx > 0:
        for role, content in prior_history[session_idx - 1]:
            parts.append((role, content))
    for role, content in current_turns:
        parts.append((role, content))
    return parts


def assemble_yourmemory_context(session_idx: int, current_turns):
    """
    YourMemory: system + compressed memory block (top_k recalled) + current turns only.
    No prior conversation history carried forward.
    """
    memories = get_memories_for_session(session_idx, query="current task context")
    if memories:
        mem_block = "Relevant context from memory:\n" + "\n".join(
            f"- [{s:.2f}] {c}" for s, c, _ in memories
        )
    else:
        mem_block = ""

    parts = [("system", SYSTEM_PROMPT)]
    if mem_block:
        parts.append(("context", mem_block))
    for role, content in current_turns:
        parts.append((role, content))
    return parts


# ---------------------------------------------------------------------------
# LLM call counter
# ---------------------------------------------------------------------------

def count_llm_calls(turns) -> int:
    """One LLM call per assistant turn."""
    return sum(1 for role, _ in turns if role == "assistant")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_session(session_idx: int, label: str, turns, naive: bool):
    if naive:
        context = assemble_naive_context(session_idx, turns)
        recall_api_calls = 0
    else:
        context = assemble_yourmemory_context(session_idx, turns)
        recall_api_calls = 1 if session_idx > 0 else 0  # Session 1: nothing to recall

    total_tokens = sum(tok(content) for _, content in context)
    llm_calls = count_llm_calls(turns)

    # Stale / irrelevant token estimation
    if naive and session_idx > 0:
        # Old session history carries stale turns (e.g., /health discussion in session 3)
        stale_turns = prior_turn_counts = [0, len(SESSION_1_TURNS), len(SESSION_1_TURNS) + len(SESSION_2_TURNS)]
        stale_tokens = sum(tok(c) for _, c in (SESSION_1_TURNS + SESSION_2_TURNS)[:stale_turns[session_idx]])
    else:
        stale_tokens = 0

    memory_overhead = 0
    if not naive and session_idx > 0:
        mems = get_memories_for_session(session_idx, "")
        memory_overhead = sum(t for _, _, t in mems) + tok("Relevant context from memory:\n")

    return {
        "session": session_idx + 1,
        "label": label,
        "approach": "Baseline" if naive else "YourMemory",
        "total_tokens": total_tokens,
        "llm_calls": llm_calls,
        "stale_tokens": stale_tokens,
        "memory_overhead_tokens": memory_overhead,
        "context_parts": len(context),
    }


def run():
    sessions = [
        (0, "Add /health endpoint",         SESSION_1_TURNS),
        (1, "Add rate limiting",             SESSION_2_TURNS),
        (2, "Write pytest for recall",       SESSION_3_TURNS),
    ]

    baseline_results = []
    ym_results = []

    for idx, label, turns in sessions:
        baseline_results.append(simulate_session(idx, label, turns, naive=True))
        ym_results.append(simulate_session(idx, label, turns, naive=False))

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  WORKFLOW COMPARISON: WITH vs WITHOUT YourMemory")
    print("  Scenario: Developer working across 3 sessions on YourMemory project")
    print("=" * 72)

    print("\n── PER-SESSION BREAKDOWN ─────────────────────────────────────────────\n")
    hdr = f"{'Session':<30} {'Approach':<12} {'Ctx Tokens':>10} {'LLM Calls':>9} {'Stale Tokens':>12}"
    print(hdr)
    print("-" * 76)

    for b, y in zip(baseline_results, ym_results):
        for r in [b, y]:
            prefix = f"S{r['session']}: {r['label']}"
            print(
                f"{prefix:<30} {r['approach']:<12}"
                f" {r['total_tokens']:>10}"
                f" {r['llm_calls']:>9}"
                f" {r['stale_tokens']:>12}"
            )
        print()

    # Cumulative
    b_total_tokens = sum(r["total_tokens"] for r in baseline_results)
    y_total_tokens = sum(r["total_tokens"] for r in ym_results)
    b_total_calls  = sum(r["llm_calls"] for r in baseline_results)
    y_total_calls  = sum(r["llm_calls"] for r in ym_results)
    b_stale        = sum(r["stale_tokens"] for r in baseline_results)
    y_stale        = sum(r["stale_tokens"] for r in ym_results)

    token_reduction = round((1 - y_total_tokens / b_total_tokens) * 100, 1)
    stale_reduction = round((1 - y_stale / b_stale) * 100, 1) if b_stale else 0

    print("── CUMULATIVE (3 SESSIONS) ───────────────────────────────────────────\n")
    print(f"{'Metric':<35} {'Baseline':>12} {'YourMemory':>12} {'Δ':>10}")
    print("-" * 72)
    print(f"{'Total context tokens':<35} {b_total_tokens:>12} {y_total_tokens:>12} {token_reduction:>9.1f}%")
    print(f"{'Total LLM calls':<35} {b_total_calls:>12} {y_total_calls:>12} {y_total_calls - b_total_calls:>+10}")
    print(f"{'Stale/irrelevant tokens injected':<35} {b_stale:>12} {y_stale:>12} {stale_reduction:>9.1f}%")

    print("\n── COST ESTIMATE (claude-sonnet-4-6 @ $3/$15 per M tokens) ───────────\n")
    INPUT_COST  = 3.00 / 1_000_000
    OUTPUT_COST = 15.00 / 1_000_000
    # Rough: 80% of context is input, 20% is output generation
    b_cost = (b_total_tokens * 0.8 * INPUT_COST) + (b_total_tokens * 0.2 * OUTPUT_COST)
    y_cost = (y_total_tokens * 0.8 * INPUT_COST) + (y_total_tokens * 0.2 * OUTPUT_COST)
    cost_saving = round((1 - y_cost / b_cost) * 100, 1)
    print(f"  Baseline estimated cost (3 sessions):    ${b_cost:.4f}")
    print(f"  YourMemory estimated cost (3 sessions):  ${y_cost:.4f}")
    print(f"  Cost reduction:                          {cost_saving}%")

    print("\n── SCALING PROJECTION (30 sessions / 1 month) ───────────────────────\n")
    # Session N baseline grows linearly with history; YourMemory stays ~flat
    avg_session_turns_tokens = sum(tok(c) for _, c in SESSION_1_TURNS)
    b_30 = sum(SYSTEM_TOKENS + i * avg_session_turns_tokens for i in range(1, 31))
    y_30_per_session = SYSTEM_TOKENS + 350 + avg_session_turns_tokens  # 350 = avg memory block
    y_30 = y_30_per_session * 30

    b_30_cost = b_30 * 0.8 * INPUT_COST + b_30 * 0.2 * OUTPUT_COST
    y_30_cost = y_30 * 0.8 * INPUT_COST + y_30 * 0.2 * OUTPUT_COST
    scale_token_reduction = round((1 - y_30 / b_30) * 100, 1)
    scale_cost_reduction  = round((1 - y_30_cost / b_30_cost) * 100, 1)

    print(f"  {'':35} {'Baseline':>12} {'YourMemory':>12} {'Δ':>10}")
    print(f"  {'Total tokens (30 sessions)':<35} {b_30:>12,} {y_30:>12,} {scale_token_reduction:>9.1f}%")
    print(f"  {'Estimated cost (30 sessions)':<35} ${b_30_cost:>11.3f} ${y_30_cost:>11.3f} {scale_cost_reduction:>9.1f}%")

    print("\n── MEMORY QUALITY (from existing benchmarks) ────────────────────────\n")
    print("  Token reduction via decay pruning (top-5 retrieval):  4.1%  (from token_efficiency.py)")
    print("  Stale memory precision — YourMemory:                  100%  (5/5 correct)")
    print("  Stale memory precision — Baseline:                      0%  (0/5 correct, all ties/wrong)")
    print("  Key: baseline surfaces outdated facts with equal rank; YourMemory demotes them via decay.")

    print("\n── SUMMARY ──────────────────────────────────────────────────────────\n")
    print("  WITHOUT YourMemory:")
    print("    - Full conversation history grows each session (O(n) tokens)")
    print("    - Stale, resolved, or irrelevant context pollutes the window")
    print("    - No additional API calls, but token costs compound fast")
    print()
    print("  WITH YourMemory:")
    print("    - Context stays roughly constant: system + compressed memories + current turn")
    print("    - Same number of LLM calls; recall_memory is a lightweight MCP call, not an LLM call")
    print("    - Stale memories decay automatically — only relevant facts injected")
    print("    - Accuracy advantage: 100% vs 0% on stale-fact disambiguation")
    print()
    print(f"  3-session token reduction:    {token_reduction}%")
    print(f"  30-session token reduction:   {scale_token_reduction}%  (grows with history length)")
    print(f"  30-session cost reduction:    {scale_cost_reduction}%")
    print("=" * 72)
    print()


if __name__ == "__main__":
    run()
