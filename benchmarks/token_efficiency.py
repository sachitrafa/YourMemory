"""
Token Efficiency Benchmark
--------------------------
Compares how many tokens YourMemory injects into context vs a naive "keep everything" baseline.

YourMemory prunes decayed memories, so retrieval returns fewer, more relevant chunks.
A baseline that never decays returns the full top_k every time.

No external service needed — runs purely on the decay math + a synthetic memory set.

Usage:
    python benchmarks/token_efficiency.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.decay import compute_strength
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Rough token counter (GPT-style: ~4 chars per token)
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Synthetic memory set
# ---------------------------------------------------------------------------

MEMORIES = [
    # (content, importance, days_since_last_access, recall_count)
    ("Sachit prefers Python over JavaScript for backend services.", 0.9, 0, 5),
    ("Sachit is building YourMemory, a persistent MCP memory server.", 0.9, 1, 8),
    ("The project uses PostgreSQL + pgvector for vector search.", 0.8, 2, 3),
    ("Sachit's GitHub username is sachitrafa.", 0.5, 7, 1),
    ("Sachit asked about Docker networking last Tuesday.", 0.3, 14, 0),
    ("The decay rate λ=0.16 was chosen after testing on synthetic data.", 0.7, 5, 2),
    ("Sachit mentioned he enjoys hiking on weekends.", 0.2, 30, 0),
    ("Sachit uses Ollama locally for embeddings.", 0.6, 10, 1),
    ("The retrieval threshold is cosine similarity ≥ 0.50.", 0.7, 3, 2),
    ("Sachit explored FastAPI vs Flask and chose FastAPI.", 0.5, 20, 0),
    ("Sachit asked how to configure pgvector indexes.", 0.3, 25, 0),
    ("The YourMemory MCP server exposes recall, store, update tools.", 0.8, 2, 4),
    ("Sachit is targeting enterprise licensing as a revenue model.", 0.7, 8, 1),
    ("Sachit mentioned interest in benchmarking against Mem0.", 0.6, 6, 2),
    ("Sachit's timezone is IST (UTC+5:30).", 0.2, 60, 0),
]

PRUNE_THRESHOLD = 0.05  # memories below this strength are excluded


def build_memory_rows():
    now = datetime.now(timezone.utc)
    rows = []
    for content, importance, days_ago, recall_count in MEMORIES:
        last_accessed = now - timedelta(days=days_ago)
        strength = compute_strength(last_accessed, recall_count, importance)
        rows.append({
            "content": content,
            "importance": importance,
            "days_ago": days_ago,
            "recall_count": recall_count,
            "strength": strength,
            "tokens": count_tokens(content),
        })
    return rows


def run():
    rows = build_memory_rows()
    top_k = 5

    # --- Baseline: no decay, return top_k by importance only ---
    baseline = sorted(rows, key=lambda r: r["importance"], reverse=True)[:top_k]
    baseline_tokens = sum(r["tokens"] for r in baseline)

    # --- YourMemory: prune below threshold, then rank by strength ---
    active = [r for r in rows if r["strength"] >= PRUNE_THRESHOLD]
    yourmemory = sorted(active, key=lambda r: r["strength"], reverse=True)[:top_k]
    yourmemory_tokens = sum(r["tokens"] for r in yourmemory)

    pruned_count = len(rows) - len(active)
    saving_pct = round((1 - yourmemory_tokens / baseline_tokens) * 100, 1) if baseline_tokens else 0

    print("=" * 60)
    print("TOKEN EFFICIENCY BENCHMARK")
    print("=" * 60)

    print(f"\n{'Memory':<52} {'Days':>4} {'Strength':>8} {'Tokens':>6} {'Active':>7}")
    print("-" * 82)
    for r in sorted(rows, key=lambda x: x["days_ago"]):
        active_flag = "yes" if r["strength"] >= PRUNE_THRESHOLD else "pruned"
        print(f"{r['content'][:50]:<52} {r['days_ago']:>4} {r['strength']:>8.4f} {r['tokens']:>6} {active_flag:>7}")

    print("\n" + "=" * 60)
    print(f"Total memories:          {len(rows)}")
    print(f"Pruned (strength<{PRUNE_THRESHOLD}):   {pruned_count}")
    print(f"Active memories:         {len(active)}")
    print()
    print(f"Baseline tokens (top {top_k}):  {baseline_tokens}")
    print(f"YourMemory tokens (top {top_k}): {yourmemory_tokens}")
    print(f"Token reduction:         {saving_pct}%")
    print()
    print("YourMemory injects (top 5 by strength):")
    for r in yourmemory:
        print(f"  [{r['strength']:.4f}] {r['content'][:70]}")
    print()
    print("Baseline injects (top 5 by importance):")
    for r in baseline:
        print(f"  [{r['importance']:.2f} imp] {r['content'][:70]}")
    print("=" * 60)

    return {
        "total": len(rows),
        "pruned": pruned_count,
        "active": len(active),
        "baseline_tokens": baseline_tokens,
        "yourmemory_tokens": yourmemory_tokens,
        "token_reduction_pct": saving_pct,
    }


if __name__ == "__main__":
    results = run()
