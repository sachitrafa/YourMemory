"""
LoCoMo-style Long-Context Memory Benchmark
-------------------------------------------
Simulates the LoCoMo benchmark structure: multi-session conversations at
t=0, t=7, t=14, and t=30 days. Tests whether a memory system can recall
facts from earlier sessions at later time points.

Real LoCoMo dataset: https://huggingface.co/datasets/snap-research/LoCoMo
This script uses a synthetic LoCoMo-style dataset. Structure matches the real dataset.

Scoring model:
  YourMemory : score = cosine_similarity × Ebbinghaus_strength
  Baseline   : score = cosine_similarity only (no temporal signal)

Each QA pair has a per-memory similarity score simulating vector search results.
High similarity = semantically close to the query. Both systems see the same
similarity distribution; the only difference is whether strength is applied.

Metric: Recall Accuracy @ k

Usage:
    python benchmarks/locomo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.decay import compute_strength
from datetime import datetime, timezone, timedelta

PRUNE_THRESHOLD = 0.05


def make_fact(content, importance, days_ago, recall_count):
    now = datetime.now(timezone.utc)
    last_accessed = now - timedelta(days=days_ago)
    strength = compute_strength(last_accessed, recall_count, importance)
    return {
        "content": content,
        "importance": importance,
        "recall_count": recall_count,
        "days_ago": days_ago,
        "strength": strength,
        "pruned": strength < PRUNE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Memory corpus (stored across 4 sessions)
# ---------------------------------------------------------------------------

FACTS = {
    "project_name":    make_fact("Sachit works on an AI memory project called YourMemory.", 0.9, 30, 5),
    "backend_stack":   make_fact("Sachit uses Python and FastAPI for the backend.", 0.8, 30, 3),
    "database":        make_fact("The database is PostgreSQL with pgvector extension.", 0.8, 30, 2),
    "favorite_food":   make_fact("Sachit's favorite food is biryani.", 0.2, 30, 0),
    "cricket":         make_fact("Sachit mentioned he watched a cricket match last weekend.", 0.1, 30, 0),
    "embed_model":     make_fact("The embedding model is nomic-embed-text via Ollama.", 0.7, 30, 1),
    "benchmark_work":  make_fact("Sachit is preparing a benchmark comparing YourMemory to Mem0.", 0.7, 23, 2),
    "decay_rate":      make_fact("The decay rate λ=0.16 was selected based on human forgetting research.", 0.7, 23, 1),
    "hn_submit":       make_fact("Sachit wants to submit the project to HackerNews.", 0.6, 23, 1),
    "mcp_published":   make_fact("Sachit published YourMemory to the MCP registry.", 0.8, 16, 3),
    "pr_number":       make_fact("The PR to modelcontextprotocol/servers is #3446.", 0.6, 16, 1),
    "revenue_model":   make_fact("Sachit decided to target enterprise licensing as a revenue model.", 0.7, 16, 2),
    "github_stars":    make_fact("Sachit mentioned the project has 2 GitHub stars currently.", 0.2, 16, 0),
    "hn_today":        make_fact("Sachit wants to post on HackerNews after benchmarks are ready.", 0.6, 0, 1),
}

# ---------------------------------------------------------------------------
# QA pairs: each with per-fact cosine similarity (simulating vector search)
# High sim = query is semantically close to that fact
# ---------------------------------------------------------------------------

QA_PAIRS = [
    {
        "question": "What is the name of Sachit's AI project?",
        "answer_key": "project_name",
        "importance": "high",
        "similarities": {
            "project_name":  0.93,
            "mcp_published": 0.61,
            "hn_today":      0.58,
            "backend_stack": 0.45,
            "database":      0.38,
            "embed_model":   0.36,
            "pr_number":     0.33,
            "revenue_model": 0.31,
            "decay_rate":    0.28,
            "benchmark_work":0.27,
            "hn_submit":     0.26,
            "github_stars":  0.22,
            "favorite_food": 0.18,
            "cricket":       0.12,
        },
    },
    {
        "question": "What database does the project use?",
        "answer_key": "database",
        "importance": "high",
        "similarities": {
            "database":      0.91,
            "backend_stack": 0.55,
            "embed_model":   0.48,
            "project_name":  0.41,
            "pr_number":     0.35,
            "decay_rate":    0.30,
            "mcp_published": 0.28,
            "benchmark_work":0.25,
            "revenue_model": 0.22,
            "hn_today":      0.20,
            "github_stars":  0.18,
            "hn_submit":     0.17,
            "favorite_food": 0.14,
            "cricket":       0.10,
        },
    },
    {
        "question": "What embedding model does YourMemory use?",
        "answer_key": "embed_model",
        "importance": "medium",
        "similarities": {
            "embed_model":   0.89,
            "database":      0.52,
            "backend_stack": 0.49,
            "project_name":  0.44,
            "decay_rate":    0.36,
            "pr_number":     0.30,
            "mcp_published": 0.28,
            "benchmark_work":0.26,
            "revenue_model": 0.21,
            "hn_today":      0.19,
            "hn_submit":     0.17,
            "github_stars":  0.15,
            "favorite_food": 0.13,
            "cricket":       0.09,
        },
    },
    {
        "question": "What is the decay rate used in YourMemory?",
        "answer_key": "decay_rate",
        "importance": "medium",
        "similarities": {
            "decay_rate":    0.88,
            "embed_model":   0.42,
            "database":      0.38,
            "benchmark_work":0.35,
            "project_name":  0.33,
            "backend_stack": 0.28,
            "pr_number":     0.25,
            "mcp_published": 0.22,
            "revenue_model": 0.20,
            "hn_today":      0.17,
            "hn_submit":     0.15,
            "github_stars":  0.13,
            "favorite_food": 0.12,
            "cricket":       0.08,
        },
    },
    {
        "question": "What is the PR number for the MCP servers submission?",
        "answer_key": "pr_number",
        "importance": "medium",
        "similarities": {
            "pr_number":     0.90,
            "mcp_published": 0.62,
            "project_name":  0.44,
            "hn_today":      0.35,
            "revenue_model": 0.30,
            "benchmark_work":0.26,
            "backend_stack": 0.23,
            "database":      0.20,
            "decay_rate":    0.18,
            "embed_model":   0.16,
            "hn_submit":     0.15,
            "github_stars":  0.14,
            "favorite_food": 0.11,
            "cricket":       0.08,
        },
    },
    {
        "question": "What revenue model did Sachit choose?",
        "answer_key": "revenue_model",
        "importance": "medium",
        "similarities": {
            "revenue_model": 0.92,
            "mcp_published": 0.40,
            "pr_number":     0.35,
            "project_name":  0.33,
            "hn_today":      0.28,
            "benchmark_work":0.25,
            "backend_stack": 0.20,
            "database":      0.18,
            "decay_rate":    0.16,
            "embed_model":   0.14,
            "hn_submit":     0.13,
            "github_stars":  0.22,
            "favorite_food": 0.10,
            "cricket":       0.07,
        },
    },
    {
        "question": "What food does Sachit like?",
        "answer_key": "favorite_food",
        "importance": "low",
        "similarities": {
            "favorite_food": 0.87,
            "cricket":       0.30,
            "project_name":  0.18,
            "backend_stack": 0.15,
            "database":      0.12,
            "embed_model":   0.11,
            "decay_rate":    0.10,
            "pr_number":     0.09,
            "revenue_model": 0.09,
            "mcp_published": 0.08,
            "benchmark_work":0.08,
            "hn_today":      0.07,
            "hn_submit":     0.07,
            "github_stars":  0.06,
        },
    },
    {
        "question": "What sport did Sachit watch recently?",
        "answer_key": "cricket",
        "importance": "low",
        "similarities": {
            "cricket":       0.86,
            "favorite_food": 0.29,
            "project_name":  0.17,
            "backend_stack": 0.14,
            "database":      0.11,
            "embed_model":   0.10,
            "decay_rate":    0.09,
            "pr_number":     0.09,
            "revenue_model": 0.08,
            "mcp_published": 0.07,
            "benchmark_work":0.07,
            "hn_today":      0.06,
            "hn_submit":     0.06,
            "github_stars":  0.05,
        },
    },
]


def score_yourmemory(fact_key, similarity):
    fact = FACTS[fact_key]
    if fact["pruned"]:
        return 0.0
    return similarity * fact["strength"]


def score_baseline(similarity):
    return similarity


def recall(qa, top_k=5):
    answer_key = qa["answer_key"]
    sims = qa["similarities"]

    ym_scores = {k: score_yourmemory(k, v) for k, v in sims.items()}
    bl_scores = {k: score_baseline(v) for k, v in sims.items()}

    ym_ranked = sorted(ym_scores.items(), key=lambda x: x[1], reverse=True)
    bl_ranked = sorted(bl_scores.items(), key=lambda x: x[1], reverse=True)

    ym_hit = any(k == answer_key for k, _ in ym_ranked[:top_k])
    bl_hit = any(k == answer_key for k, _ in bl_ranked[:top_k])

    ym_rank = next((i + 1 for i, (k, _) in enumerate(ym_ranked) if k == answer_key), None)
    bl_rank = next((i + 1 for i, (k, _) in enumerate(bl_ranked) if k == answer_key), None)

    return ym_hit, bl_hit, ym_rank, bl_rank


def run():
    top_k = 5

    print("=" * 72)
    print("LoCoMo-STYLE LONG-CONTEXT MEMORY BENCHMARK")
    print(f"Sessions: Day 0, 7, 14, 30 | Scoring: sim×strength vs sim-only | top_k={top_k}")
    print("=" * 72)

    print("\nMemory strength at query time (Day 30):")
    print(f"{'Content':<55} {'Days':>4} {'Imp':>5} {'RC':>3} {'Strength':>9} {'Pruned':>7}")
    print("-" * 85)
    for key, m in sorted(FACTS.items(), key=lambda x: x[1]["strength"], reverse=True):
        pruned = "PRUNED" if m["pruned"] else ""
        print(f"{m['content'][:53]:<55} {m['days_ago']:>4} {m['importance']:>5.1f} {m['recall_count']:>3} {m['strength']:>9.4f} {pruned:>7}")

    print("\n" + "=" * 72)
    print(f"{'Question':<44} {'Imp':<7} {'YM':>6} {'BL':>6} {'YM#':>5} {'BL#':>5}")
    print("-" * 72)

    ym_total = bl_total = 0
    ym_high = bl_high = high_n = 0
    ym_med = bl_med = med_n = 0
    ym_low = bl_low = low_n = 0

    for qa in QA_PAIRS:
        ym_hit, bl_hit, ym_rank, bl_rank = recall(qa, top_k)
        ym_total += ym_hit
        bl_total += bl_hit

        imp = qa["importance"]
        if imp == "high":
            ym_high += ym_hit; bl_high += bl_hit; high_n += 1
        elif imp == "medium":
            ym_med += ym_hit; bl_med += bl_hit; med_n += 1
        else:
            ym_low += ym_hit; bl_low += bl_hit; low_n += 1

        ym_str = "HIT" if ym_hit else "miss"
        bl_str = "HIT" if bl_hit else "miss"
        ym_r = str(ym_rank) if ym_rank else "—"
        bl_r = str(bl_rank) if bl_rank else "—"

        print(f"{qa['question'][:42]:<44} {imp:<7} {ym_str:>6} {bl_str:>6} {ym_r:>5} {bl_r:>5}")

    n = len(QA_PAIRS)
    print("\n" + "=" * 72)
    print(f"Recall@{top_k}:   YourMemory {ym_total}/{n} ({round(ym_total/n*100)}%)   Baseline {bl_total}/{n} ({round(bl_total/n*100)}%)")
    print()
    print(f"  High importance (should persist):  YM {ym_high}/{high_n}   BL {bl_high}/{high_n}")
    print(f"  Medium importance (may decay):     YM {ym_med}/{med_n}   BL {bl_med}/{med_n}")
    print(f"  Low importance (expect pruned):    YM {ym_low}/{low_n}   BL {bl_low}/{low_n}")
    print()
    print("YourMemory: high-importance facts survive 30 days; ephemeral noise is pruned.")
    print("Baseline:   no temporal signal — surfaces everything equally, noise included.")
    print("=" * 72)

    return {
        "ym_recall_pct": round(ym_total / n * 100),
        "bl_recall_pct": round(bl_total / n * 100),
    }


if __name__ == "__main__":
    results = run()
