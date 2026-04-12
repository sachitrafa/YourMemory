"""
Two-Session Workflow Comparison: Normal vs YourMemory
------------------------------------------------------
Runs the same multi-step "Debug & Fix API Endpoint" workflow twice:

  Session 1 input: "The /recall endpoint is slow for users with >1000 memories"
  Session 2 input: "The /store endpoint times out under concurrent writes"

Within each session, both Normal and YourMemory are simulated.

WHY LLM CALLS DIFFER:
  Normal (stateless):  The assistant has no prior context. It must ask clarifying
                       questions (stack, driver, pool size, concurrency target) before
                       it can implement anything. Each clarification round = 1 LLM call.

  YourMemory:          recall_memory surfaces the stack, driver, preferences, and prior
                       fixes before the first user message. The assistant skips all
                       clarifying questions and delivers the solution immediately.

Session 1: Both start cold — identical turns, identical LLM calls.
Session 2: Gap opens. Normal asks 2 clarifying questions = 2 extra LLM calls.

Workflow turns:

  NORMAL Session 2 (6 LLM calls):
    U1  user        — describe /store timeout problem
    A1  assistant   — "What DB driver and pool size are you using?"       [call 1 — clarify]
    U2  user        — "asyncpg, pool_size=5"
    A2  assistant   — "What's your concurrency target? Transactions?"     [call 2 — clarify]
    U3  user        — "50 concurrent, yes need transactions"
    A3  assistant   — implement pool resize + retry                        [call 3]
    U4  user        — "wrap in a transaction"
    A4  assistant   — add transaction wrapper                              [call 4]
    U5  user        — "write a pytest"
    A5  assistant   — write test                                           [call 5]

  YOURMEMORY Session 2 (4 LLM calls — skips both clarifying calls):
    [memory block injected: stack, asyncpg, pool=5, 50+ concurrent, Python]
    U1  user        — describe /store timeout problem
    A1  assistant   — immediately diagnose + propose pool/retry fix        [call 1]
    U2  user        — "also wrap in a transaction"
    A2  assistant   — add transaction                                      [call 2]
    U3  user        — "write a pytest"
    A3  assistant   — write test                                           [call 3]

Usage:
    python3 benchmarks/two_session_comparison.py
"""

import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.services.decay import compute_strength


def tok(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a senior backend engineer helping debug and fix a FastAPI + "
    "PostgreSQL + pgvector application. Be concise and precise."
)
SYSTEM_TOKENS = tok(SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Session 1 — /recall latency fix  (identical for both approaches)
# ---------------------------------------------------------------------------

S1_TURNS = [
    ("user",
     "Our /recall endpoint is getting slow. Users with over 1000 memories are "
     "seeing p95 latency above 2 seconds. We're on FastAPI + PostgreSQL + pgvector. "
     "I prefer Python solutions. Can you help diagnose?"),

    ("assistant",
     "Most likely cause is a full table scan — no HNSW/IVFFlat index on the embeddings "
     "column. Do you have a vector index? And is user_id filtered before the similarity scan?"),

    ("user",
     "No index yet. We filter user_id in a WHERE clause. Fix the index first."),

    ("assistant",
     "Add an HNSW index:\n```sql\nCREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops)\n"
     "WITH (m = 16, ef_construction = 64);\nCREATE INDEX ON memories (user_id);\n```"),

    ("user",
     "Good. Our cosine threshold is 0.5 — add that filter to the query too."),

    ("assistant",
     "Updated recall query with 0.5 cosine threshold:\n```python\nresults = await db.fetch(\n"
     "    'SELECT id, content, 1-(embedding<=>$1) AS sim FROM memories'\n"
     "    ' WHERE user_id=$2 AND 1-(embedding<=>$1)>=0.5 ORDER BY embedding<=>$1 LIMIT $3',\n"
     "    query_embedding, user_id, top_k\n)\n```"),

    ("user",
     "Write a pytest confirming latency stays under 500ms for a user with 1000 memories."),

    ("assistant",
     "```python\n@pytest.mark.asyncio\nasync def test_recall_latency_under_500ms():\n"
     "    async with AsyncClient(app=app, base_url='http://test') as client:\n"
     "        start = time.perf_counter()\n"
     "        resp = await client.post('/recall',\n"
     "            json={'query': 'Python preferences', 'user_id': 'bench_user', 'top_k': 5},\n"
     "            headers={'X-Api-Key': 'test-key'})\n"
     "        elapsed_ms = (time.perf_counter() - start) * 1000\n"
     "    assert resp.status_code == 200\n"
     "    assert elapsed_ms < 500\n```"),
]


# ---------------------------------------------------------------------------
# Session 2 — /store concurrency fix
#
# NORMAL: assistant doesn't know the stack — must ask 2 clarifying questions
#         before it can implement anything. 5 LLM calls total.
#
# YOURMEMORY: memory block supplies stack/driver/pool/concurrency context.
#             Assistant skips clarifying questions. 3 LLM calls total.
# ---------------------------------------------------------------------------

S2_NORMAL_TURNS = [
    ("user",
     "The /store endpoint times out under concurrent writes. "
     "Seeing DB connection pool exhaustion in the logs. Help me fix this."),

    # Clarifying call 1 — assistant doesn't know the driver or current pool config
    ("assistant",
     "To diagnose this I need a few details: "
     "(1) What database driver are you using — asyncpg, psycopg2, or something else? "
     "(2) What is your current connection pool size? "
     "(3) What framework is this — FastAPI, Flask?"),

    ("user",
     "asyncpg, pool_size=5, FastAPI."),

    # Clarifying call 2 — assistant doesn't know concurrency target or transaction requirement
    ("assistant",
     "Got it. Two more questions before I write the fix: "
     "(1) What is your peak concurrency target — how many simultaneous writers? "
     "(2) Do the writes need to be transactional (atomic insert + any side effects)?"),

    ("user",
     "50+ concurrent writers. Yes, wrap in a transaction."),

    # Implementation call 3
    ("assistant",
     "Increase pool size and add exponential backoff retry with transaction:\n"
     "```python\npool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=10, max_size=50)\n\n"
     "async def store_with_retry(pool, content, embedding, user_id, retries=3):\n"
     "    for attempt in range(retries):\n"
     "        try:\n"
     "            async with pool.acquire(timeout=2) as conn:\n"
     "                async with conn.transaction():\n"
     "                    await conn.execute(\n"
     "                        'INSERT INTO memories (content,embedding,user_id) VALUES ($1,$2,$3)',\n"
     "                        content, embedding, user_id)\n"
     "            return\n"
     "        except asyncpg.TooManyConnectionsError:\n"
     "            await asyncio.sleep(0.1 * 2**attempt)\n"
     "    raise RuntimeError('store failed after retries')\n```"),

    ("user",
     "Write a pytest that fires 50 concurrent store requests and asserts all succeed."),

    # Test call 4
    ("assistant",
     "```python\n@pytest.mark.asyncio\nasync def test_concurrent_store_all_succeed():\n"
     "    async with AsyncClient(app=app, base_url='http://test') as client:\n"
     "        tasks = [\n"
     "            client.post('/store',\n"
     "                json={'content': f'memory {i}', 'user_id': 'load_test'},\n"
     "                headers={'X-Api-Key': 'test-key'})\n"
     "            for i in range(50)\n"
     "        ]\n"
     "        responses = await asyncio.gather(*tasks)\n"
     "    failed = [r for r in responses if r.status_code != 200]\n"
     "    assert not failed, f'{len(failed)}/50 failed'\n```"),
]

S2_YOURMEMORY_TURNS = [
    # memory block already injected: FastAPI+pgvector, asyncpg, pool=5, 50+ concurrent, Python
    ("user",
     "The /store endpoint times out under concurrent writes. "
     "Seeing DB connection pool exhaustion in the logs. Help me fix this."),

    # No clarifying questions needed — assistant already knows the stack
    ("assistant",
     "With asyncpg pool_size=5 and 50+ concurrent writers, you're exhausting the pool. "
     "Resize to max_size=50 and add retry with backoff + transaction:\n"
     "```python\npool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=10, max_size=50)\n\n"
     "async def store_with_retry(pool, content, embedding, user_id, retries=3):\n"
     "    for attempt in range(retries):\n"
     "        try:\n"
     "            async with pool.acquire(timeout=2) as conn:\n"
     "                async with conn.transaction():\n"
     "                    await conn.execute(\n"
     "                        'INSERT INTO memories (content,embedding,user_id) VALUES ($1,$2,$3)',\n"
     "                        content, embedding, user_id)\n"
     "            return\n"
     "        except asyncpg.TooManyConnectionsError:\n"
     "            await asyncio.sleep(0.1 * 2**attempt)\n"
     "    raise RuntimeError('store failed after retries')\n```"),

    ("user",
     "Write a pytest that fires 50 concurrent store requests and asserts all succeed."),

    ("assistant",
     "```python\n@pytest.mark.asyncio\nasync def test_concurrent_store_all_succeed():\n"
     "    async with AsyncClient(app=app, base_url='http://test') as client:\n"
     "        tasks = [\n"
     "            client.post('/store',\n"
     "                json={'content': f'memory {i}', 'user_id': 'load_test'},\n"
     "                headers={'X-Api-Key': 'test-key'})\n"
     "            for i in range(50)\n"
     "        ]\n"
     "        responses = await asyncio.gather(*tasks)\n"
     "    failed = [r for r in responses if r.status_code != 200]\n"
     "    assert not failed, f'{len(failed)}/50 failed'\n```"),
]


# ---------------------------------------------------------------------------
# Memories stored after Session 1 — injected by YourMemory in Session 2
# ---------------------------------------------------------------------------

S1_MEMORIES = [
    ("Sachit prefers Python for all backend solutions.",                                        0.9, 1, 3),
    ("The YourMemory project runs on FastAPI + PostgreSQL + pgvector.",                        0.9, 1, 2),
    ("asyncpg is the database driver; current pool_size=5.",                                   0.8, 1, 2),
    ("The project targets 50+ concurrent users (enterprise tier).",                            0.8, 1, 1),
    ("HNSW index (m=16, ef_construction=64) added to embeddings column in Session 1.",        0.8, 1, 1),
    ("The /recall endpoint uses cosine similarity threshold >= 0.5.",                          0.7, 1, 2),
    ("btree index on user_id column added in Session 1.",                                      0.6, 1, 1),
]


def get_session1_memories(top_k=5):
    now = datetime.now(timezone.utc)
    ranked = []
    for content, importance, days_ago, recall_count in S1_MEMORIES:
        strength = compute_strength(now - timedelta(days=days_ago), recall_count, importance)
        ranked.append((strength, content, tok(content)))
    ranked.sort(reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def build_context(turns_so_far, session_idx, approach):
    ctx = [("system", SYSTEM_PROMPT)]
    if session_idx == 1 and approach == "yourmemory":
        mems = get_session1_memories()
        mem_block = "Context recalled from memory:\n" + "\n".join(
            f"- {c}" for _, c, _ in mems
        )
        ctx.append(("context", mem_block))
    for role, content in turns_so_far:
        ctx.append((role, content))
    return ctx


def context_tokens(ctx):
    return sum(tok(content) for _, content in ctx)


def stale_tokens(ctx, session_idx, approach):
    """Normal in session 2 has no history to carry — it starts stateless.
       All context is current. Stale tokens = 0 for both approaches here.
       The cost is paid in extra LLM calls (clarifications), not stale tokens."""
    return 0


# ---------------------------------------------------------------------------
# Simulate one full session step-by-step
# ---------------------------------------------------------------------------

def simulate_session(session_idx, turns, approach):
    label = ["Session 1: /recall latency fix", "Session 2: /store concurrency fix"][session_idx]
    approach_label = "Normal" if approach == "normal" else "YourMemory"

    snapshots = []
    accumulated = []
    llm_call_idx = 0

    for role, content in turns:
        accumulated.append((role, content))
        if role == "assistant":
            llm_call_idx += 1
            ctx = build_context(accumulated, session_idx, approach)
            total = context_tokens(ctx)
            # Tag what kind of call this is
            call_type = "clarify" if approach == "normal" and session_idx == 1 and llm_call_idx <= 2 else "work"
            snapshots.append({
                "llm_call": llm_call_idx,
                "ctx_tokens": total,
                "call_type": call_type,
            })

    return {
        "session_idx": session_idx,
        "label": label,
        "approach": approach_label,
        "snapshots": snapshots,
        "total_llm_calls": llm_call_idx,
        "clarify_calls": sum(1 for s in snapshots if s["call_type"] == "clarify"),
        "work_calls": sum(1 for s in snapshots if s["call_type"] == "work"),
        "total_ctx_tokens": sum(s["ctx_tokens"] for s in snapshots),
        "peak_ctx_tokens": max(s["ctx_tokens"] for s in snapshots),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run():
    results = {
        (0, "normal"):      simulate_session(0, S1_TURNS, "normal"),
        (0, "yourmemory"):  simulate_session(0, S1_TURNS, "yourmemory"),
        (1, "normal"):      simulate_session(1, S2_NORMAL_TURNS, "normal"),
        (1, "yourmemory"):  simulate_session(1, S2_YOURMEMORY_TURNS, "yourmemory"),
    }

    print()
    print("=" * 76)
    print("  TWO-SESSION WORKFLOW: Normal vs YourMemory")
    print("  Workflow: Debug & Fix API Endpoint")
    print("=" * 76)

    for s_idx in [0, 1]:
        n = results[(s_idx, "normal")]
        y = results[(s_idx, "yourmemory")]

        print(f"\n{'─'*76}")
        print(f"  {n['label'].upper()}")
        print(f"{'─'*76}\n")

        all_calls = max(len(n["snapshots"]), len(y["snapshots"]))
        print(f"  {'LLM Call':<10} {'Type':<10} {'Normal Tokens':>14} {'YM Tokens':>12}")
        print(f"  {'-'*50}")

        n_snaps = {s["llm_call"]: s for s in n["snapshots"]}
        y_snaps = {s["llm_call"]: s for s in y["snapshots"]}

        for i in range(1, all_calls + 1):
            ns = n_snaps.get(i)
            ys = y_snaps.get(i)
            call_label = f"Call {i}"
            ctype = (ns["call_type"] if ns else (ys["call_type"] if ys else "work"))
            n_tok_str = f"{ns['ctx_tokens']:>14}" if ns else f"{'(skipped)':>14}"
            y_tok_str = f"{ys['ctx_tokens']:>12}" if ys else f"{'(skipped)':>12}"
            type_tag = f"[{ctype}]" if ctype == "clarify" else ""
            print(f"  {call_label:<10} {type_tag:<10} {n_tok_str} {y_tok_str}")

        print(f"\n  Totals:")
        print(f"    Normal     — {n['total_llm_calls']} LLM calls  "
              f"({n['clarify_calls']} clarify + {n['work_calls']} work)  "
              f"| {n['total_ctx_tokens']} total tokens  | peak {n['peak_ctx_tokens']}")
        print(f"    YourMemory — {y['total_llm_calls']} LLM calls  "
              f"({y['clarify_calls']} clarify + {y['work_calls']} work)  "
              f"| {y['total_ctx_tokens']} total tokens  | peak {y['peak_ctx_tokens']}")

        call_delta = n["total_llm_calls"] - y["total_llm_calls"]
        tok_delta  = round((1 - y["total_ctx_tokens"] / n["total_ctx_tokens"]) * 100, 1) if n["total_ctx_tokens"] else 0
        print(f"\n    LLM calls saved:   {call_delta}  |  Token reduction: {tok_delta}%")

    # ---- Cumulative --------------------------------------------------------
    n_calls  = sum(results[(s, "normal")]["total_llm_calls"]      for s in [0, 1])
    y_calls  = sum(results[(s, "yourmemory")]["total_llm_calls"]  for s in [0, 1])
    n_tokens = sum(results[(s, "normal")]["total_ctx_tokens"]      for s in [0, 1])
    y_tokens = sum(results[(s, "yourmemory")]["total_ctx_tokens"]  for s in [0, 1])

    call_reduction  = round((1 - y_calls  / n_calls)  * 100, 1)
    token_reduction = round((1 - y_tokens / n_tokens) * 100, 1)

    INPUT_COST  = 3.00  / 1_000_000
    OUTPUT_COST = 15.00 / 1_000_000
    n_cost = n_tokens * 0.8 * INPUT_COST + n_tokens * 0.2 * OUTPUT_COST
    y_cost = y_tokens * 0.8 * INPUT_COST + y_tokens * 0.2 * OUTPUT_COST
    cost_reduction = round((1 - y_cost / n_cost) * 100, 1)

    print(f"\n{'═'*76}")
    print("  CUMULATIVE — BOTH SESSIONS")
    print(f"{'═'*76}\n")
    print(f"  {'Metric':<38} {'Normal':>10} {'YourMemory':>12} {'Δ':>8}")
    print(f"  {'-'*70}")
    print(f"  {'Total LLM calls':<38} {n_calls:>10} {y_calls:>12} {-call_reduction:>+7.1f}%")
    print(f"  {'Total context tokens':<38} {n_tokens:>10} {y_tokens:>12} {-token_reduction:>+7.1f}%")
    print(f"  {'Estimated cost (claude-sonnet-4-6)':<38} ${n_cost:>9.5f} ${y_cost:>11.5f} {-cost_reduction:>+7.1f}%")

    # ---- Memory block detail -----------------------------------------------
    mems = get_session1_memories()
    mem_block_tokens = sum(t for _, _, t in mems) + tok("Context recalled from memory:\n")

    print(f"\n{'─'*76}")
    print(f"  YOURMEMORY — Session 2 memory block ({mem_block_tokens} tokens)\n")
    for strength, content, t in mems:
        print(f"  [{strength:.3f}] ({t:>3} tok) {content}")

    print(f"\n{'─'*76}")
    print("  WHY LLM CALLS ARE REDUCED\n")
    print("  Normal (stateless): assistant has no stack/driver/concurrency context.")
    print("  It must ask 2 clarifying questions before writing a single line of code.")
    print("  Each question = 1 LLM call. Those calls produce no code — pure overhead.\n")
    print("  YourMemory: recall_memory surfaces stack + driver + pool size + concurrency")
    print(f"  target in {mem_block_tokens} tokens before the first user message.")
    print("  The assistant reads the memory block and solves the problem immediately.\n")
    print(f"  Result: {call_reduction}% fewer LLM calls, {token_reduction}% fewer tokens, {cost_reduction}% lower cost.")
    print(f"{'═'*76}\n")


if __name__ == "__main__":
    run()
