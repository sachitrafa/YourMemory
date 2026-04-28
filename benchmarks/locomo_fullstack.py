"""
LoCoMo Recall@5 Benchmark — YourMemory FULL STACK
--------------------------------------------------
Full production pipeline: BM25 + vector + graph BFS + Ebbinghaus decay.
Replicates the 59% result methodology from locomo_4way.py without
needing external APIs or a running HTTP server.

Pipeline per sample:
  1. Store all session summaries directly into a dedicated DuckDB
     (with graph indexing via index_memory)
  2. Query via retrieve() — full BM25 + cosine + graph BFS
  3. Evaluate with answer_hit
  4. DELETE user memories between samples

Usage:
    python benchmarks/locomo_fullstack.py
"""

import sys
import os
import json
import math
import time
import numpy as np
from datetime import datetime, timezone
from dateutil import parser as dateparser

# Use a dedicated benchmark DB — never pollute the real one
BENCH_DB = os.path.expanduser("/tmp/locomo_bench.duckdb")
os.environ["YOURMEMORY_DB"] = BENCH_DB

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.migrate import migrate
from src.db.connection import get_backend, get_conn, emb_to_db, duckdb_rows
from src.services.embed import embed, DEFAULT_MODEL
from src.graph.graph_store import index_memory
from src.services.retrieve import retrieve
from src.services.decay import compute_strength

LOCOMO_PATH  = os.path.expanduser("~/Desktop/locomo/data/locomo10.json")
TOP_K        = 5
IMPORTANCE   = 0.7


def parse_date(s: str) -> datetime:
    try:
        return dateparser.parse(s, dayfirst=True).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def answer_hit(answer: str, chunks: list) -> bool:
    al  = str(answer).lower().strip()
    ctx = " ".join(str(c) for c in chunks).lower()
    if al in ctx:
        return True
    toks = [t for t in al.split() if len(t) > 3]
    if not toks:
        return al in ctx
    return sum(1 for t in toks if t in ctx) / len(toks) >= 0.5


def store_session(user_id: str, text: str, stored_at: datetime) -> int | None:
    """Insert one memory directly into DuckDB and index it in the graph."""
    emb     = embed(text)
    backend = get_backend()
    emb_str = emb_to_db(emb, backend)
    conn    = get_conn()

    memory_id = None
    try:
        if backend == "duckdb":
            conn.execute("""
                INSERT INTO memories (user_id, content, category, importance, embedding, last_accessed_at)
                VALUES (?, ?, 'fact', ?, ?, ?)
                ON CONFLICT (user_id, content) DO UPDATE
                    SET recall_count = recall_count + 1, last_accessed_at = excluded.last_accessed_at
            """, [user_id, text, IMPORTANCE, emb_str, stored_at])
            row = conn.execute(
                "SELECT id FROM memories WHERE user_id = ? AND content = ?", [user_id, text]
            ).fetchone()
            memory_id = row[0] if row else None
        else:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO memories (user_id, content, category, importance, embedding, last_accessed_at)
                VALUES (?, ?, 'fact', ?, ?, ?)
                ON CONFLICT (user_id, content) DO UPDATE
                    SET recall_count = recall_count + 1, last_accessed_at = excluded.last_accessed_at
            """, (user_id, text, IMPORTANCE, emb_str, stored_at))
            memory_id = cur.lastrowid
            cur.close()
            conn.commit()
    except Exception as e:
        print(f"  [store error: {e}]")
    finally:
        conn.close()

    if memory_id is not None:
        strength = compute_strength(
            last_accessed_at=stored_at,
            recall_count=0,
            importance=IMPORTANCE,
            category="fact",
        )
        index_memory(memory_id, user_id, text, strength, IMPORTANCE, "fact", embedding=emb)

    return memory_id


def delete_user(user_id: str) -> None:
    """Remove all memories for this user from DB (graph nodes persist but are scoped by user_id)."""
    conn = get_conn()
    try:
        if get_backend() == "duckdb":
            conn.execute("DELETE FROM memories WHERE user_id = ?", [user_id])
        else:
            cur = conn.cursor()
            cur.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            conn.commit()
            cur.close()
    finally:
        conn.close()


def run():
    # Initialise benchmark DB schema
    migrate()
    print(f"YourMemory FULL STACK LoCoMo — model: {DEFAULT_MODEL}")
    print(f"Pipeline: vector + BM25 + graph BFS + decay")
    print(f"DB: {BENCH_DB}")
    print("=" * 65)

    with open(LOCOMO_PATH) as f:
        data = json.load(f)

    results    = []
    total_hits = total_qa = 0
    start      = time.time()

    for idx, sample in enumerate(data):
        conv = sample["conversation"]
        sa   = conv.get("speaker_a", "A")
        sb   = conv.get("speaker_b", "B")
        user_id = f"lme_locomo_{idx}"

        all_qa = [
            q for q in sample["qa"]
            if q.get("category") in (1, 2, 3, 4)
            and isinstance(q.get("answer", ""), str)
        ]

        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1])
        )
        summaries = sample.get("session_summary", {})
        now       = datetime.now(timezone.utc)

        # Store all session summaries with current timestamp (mirrors live server
        # methodology from locomo_4way.py — HTTP API uses CURRENT_TIMESTAMP,
        # so decay is ~0 days at query time, keeping graph node scores live).
        stored = 0
        for sk in session_keys:
            summary = summaries.get(sk + "_summary", "")
            if summary:
                mid = store_session(user_id, summary, stored_at=now)
                if mid:
                    stored += 1

        # Query
        hits = 0
        for qa in all_qa:
            result  = retrieve(user_id, qa["question"], top_k=TOP_K)
            chunks  = [m["content"] for m in result.get("memories", [])]
            if answer_hit(qa["answer"], chunks):
                hits += 1

        # Cleanup
        delete_user(user_id)

        pct = round(hits / len(all_qa) * 100) if all_qa else 0
        total_hits += hits
        total_qa   += len(all_qa)
        elapsed     = time.time() - start

        print(f"Sample {idx+1:2d} | {sa} & {sb:<20} | {hits:3d}/{len(all_qa):3d} = {pct:3d}%"
              f"  ({stored} sessions, {elapsed:.0f}s)")
        results.append({"sample": idx+1, "speakers": f"{sa} & {sb}",
                        "hits": hits, "total": len(all_qa), "pct": pct})

    overall = round(total_hits / total_qa * 100) if total_qa else 0

    print()
    print("=" * 65)
    print(f"{'Sample':<8} {'Speakers':<28} {'Recall@5':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['sample']:<8} {r['speakers']:<28} {r['pct']:>9}%")
    print("-" * 50)
    print(f"{'TOTAL':<8} {str(total_qa)+' QA pairs':<28} {overall:>9}%")
    print("=" * 65)
    print(f"\nYourMemory Recall@5 (full stack, {DEFAULT_MODEL}): {overall}%")
    print(f"Previous benchmark (all-mpnet-base-v2, full stack):          59%")
    print(f"Delta: {'+' if overall >= 59 else ''}{overall - 59}pp")
    print(f"\nTotal time: {time.time()-start:.0f}s")


if __name__ == "__main__":
    run()
