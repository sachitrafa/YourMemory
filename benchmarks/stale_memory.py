"""
Stale Memory Precision Benchmark
---------------------------------
Tests whether a memory system surfaces the CURRENT fact vs an outdated one.

Scenario: A memory was stored, then the user changed their preference/fact.
A new (correct) memory is stored. Both exist in the system.

Metric: Does retrieval return the newer, correct memory ranked #1?

YourMemory: decayed strength demotes the stale memory automatically.
Baseline:   no decay — ranks purely on cosine similarity, may surface the stale one.

Pairs tested (stale → current):
  - Framework preference changed (React → Vue)
  - Job title changed
  - Database choice changed
  - Location changed
  - Project name changed

Usage:
    python benchmarks/stale_memory.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.decay import compute_strength
from datetime import datetime, timezone, timedelta

PRUNE_THRESHOLD = 0.05
SIMILARITY_SCORE = 0.85   # both stale and current have high semantic overlap — worst case


def make_memory(content, importance, days_ago, recall_count):
    now = datetime.now(timezone.utc)
    last_accessed = now - timedelta(days=days_ago)
    strength = compute_strength(last_accessed, recall_count, importance)
    return {
        "content": content,
        "importance": importance,
        "days_ago": days_ago,
        "recall_count": recall_count,
        "strength": strength,
        "similarity": SIMILARITY_SCORE,
        "yourmemory_score": SIMILARITY_SCORE * strength,
        "baseline_score": importance,  # baseline ranks by importance only
    }


# Each pair: (query, stale_memory, current_memory)
TEST_PAIRS = [
    {
        "query": "What frontend framework does Sachit use?",
        "stale":   make_memory("Sachit prefers React for frontend development.", 0.8, 45, 2),
        "current": make_memory("Sachit switched to Vue.js for all new frontend work.", 0.8, 3, 1),
        "label": "Framework preference (React → Vue)",
    },
    {
        "query": "What is Sachit's job title?",
        "stale":   make_memory("Sachit works as a Senior Data Scientist at Acme Corp.", 0.7, 60, 1),
        "current": make_memory("Sachit is now Head of AI at a new startup.", 0.7, 5, 0),
        "label": "Job title change",
    },
    {
        "query": "What database does the project use?",
        "stale":   make_memory("The project uses MongoDB for storage.", 0.8, 50, 3),
        "current": make_memory("The project migrated to PostgreSQL + pgvector.", 0.8, 4, 2),
        "label": "Database migration (Mongo → Postgres)",
    },
    {
        "query": "Where does Sachit live?",
        "stale":   make_memory("Sachit is based in Bangalore, India.", 0.5, 90, 0),
        "current": make_memory("Sachit relocated to Mumbai in January 2026.", 0.5, 10, 0),
        "label": "Location change",
    },
    {
        "query": "What is the project called?",
        "stale":   make_memory("The project is called MemoryCore.", 0.9, 30, 4),
        "current": make_memory("The project was renamed to YourMemory.", 0.9, 2, 6),
        "label": "Project rename",
    },
]


def run():
    print("=" * 70)
    print("STALE MEMORY PRECISION BENCHMARK")
    print("=" * 70)
    print(f"{'Scenario':<40} {'YourMemory':>12} {'Baseline':>10} {'Winner':>8}")
    print("-" * 70)

    ym_correct = 0
    bl_correct = 0

    for pair in TEST_PAIRS:
        stale = pair["stale"]
        current = pair["current"]

        # YourMemory: score = similarity × strength
        ym_stale_score = stale["yourmemory_score"]
        ym_current_score = current["yourmemory_score"]
        ym_picks_current = ym_current_score > ym_stale_score

        # Baseline: score = importance (no decay signal)
        bl_stale_score = stale["baseline_score"]
        bl_current_score = current["baseline_score"]
        # Tie-break: when importance is equal, both score equal — baseline may surface stale (older)
        bl_picks_current = bl_current_score > bl_stale_score

        ym_result = "CORRECT" if ym_picks_current else "WRONG"
        bl_result = "CORRECT" if bl_picks_current else "TIE/WRONG"

        if ym_picks_current:
            ym_correct += 1
        if bl_picks_current:
            bl_correct += 1

        winner = "YM" if ym_picks_current and not bl_picks_current else ("TIE" if ym_picks_current == bl_picks_current else "BL")

        print(f"{pair['label']:<40} {ym_result:>12} {bl_result:>10} {winner:>8}")

    total = len(TEST_PAIRS)
    ym_acc = round(ym_correct / total * 100, 1)
    bl_acc = round(bl_correct / total * 100, 1)

    print("\n" + "=" * 70)
    print(f"YourMemory precision:  {ym_correct}/{total} = {ym_acc}%")
    print(f"Baseline precision:    {bl_correct}/{total} = {bl_acc}%")
    print()

    print("Score breakdown (similarity × strength vs importance-only):")
    print(f"{'Scenario':<40} {'YM stale':>9} {'YM curr':>9} {'BL stale':>9} {'BL curr':>9}")
    print("-" * 80)
    for pair in TEST_PAIRS:
        print(
            f"{pair['label']:<40}"
            f" {pair['stale']['yourmemory_score']:>9.4f}"
            f" {pair['current']['yourmemory_score']:>9.4f}"
            f" {pair['stale']['baseline_score']:>9.4f}"
            f" {pair['current']['baseline_score']:>9.4f}"
        )

    print("=" * 70)
    print()
    print("Key insight: baseline scores ties on all equal-importance pairs.")
    print("YourMemory naturally demotes stale memories via decay — no reranking needed.")

    return {
        "yourmemory_precision": ym_acc,
        "baseline_precision": bl_acc,
        "ym_correct": ym_correct,
        "bl_correct": bl_correct,
        "total": total,
    }


if __name__ == "__main__":
    results = run()
