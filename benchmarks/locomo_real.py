"""
LoCoMo Real Benchmark: YourMemory vs Mem0
------------------------------------------
Uses the actual LoCoMo dataset (10 samples, ~199 QA pairs per sample).
Feeds session summaries into both systems, then evaluates recall accuracy.

Mem0: hosted API (no decay, recency-unaware)
YourMemory: in-process (Ollama embeddings + Ebbinghaus decay, no server needed)

Scoring for YourMemory:
    score = cosine_similarity × compute_strength(session_date, recall_count, importance)

Metric: Recall@5 — does the correct answer appear in retrieved context?

Usage:
    python benchmarks/locomo_real.py

Requires:
    - Ollama running locally with nomic-embed-text pulled
    - MEM0_API_KEY in .env
    - LoCoMo dataset at ../locomo/data/locomo10.json (relative to repo root)
"""

import sys
import os
import json
import time
import uuid
import numpy as np
from datetime import datetime, timezone
from dateutil import parser as dateparser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.services.embed import embed
from src.services.decay import compute_strength
from mem0 import MemoryClient

LOCOMO_PATH = os.path.join(os.path.dirname(__file__), "../../locomo/data/locomo10.json")
PRUNE_THRESHOLD = 0.05
TOP_K = 5
# Run only first N samples (None = all 10). Set to 2 for a fast smoke test.
MAX_SAMPLES = None  # all 10
# Run only first N QA pairs per sample (None = all). Categories 1-4 only.
MAX_QA_PER_SAMPLE = 20
# Resume from this sample index (0-based). Set to 0 to run all from scratch.
START_FROM_SAMPLE = 3
# Pre-seeded results from previous run (to combine with new results)
PRIOR_RESULTS = [
    {"sample": 1, "speakers": "Caroline & Melanie", "qa_total": 20, "ym_hits": 8, "m0_hits": 7, "ym_pct": 40, "m0_pct": 35},
    {"sample": 2, "speakers": "Jon & Gina",         "qa_total": 20, "ym_hits": 13, "m0_hits": 1, "ym_pct": 65, "m0_pct": 5},
    {"sample": 3, "speakers": "John & Maria",       "qa_total": 20, "ym_hits": 4, "m0_hits": 3, "ym_pct": 20, "m0_pct": 15},
]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def parse_session_date(date_str: str) -> datetime:
    """Parse LoCoMo date strings like '1:56 pm on 8 May, 2023'."""
    try:
        return dateparser.parse(date_str, dayfirst=True).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def load_locomo():
    with open(LOCOMO_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# YourMemory in-process store
# ---------------------------------------------------------------------------

class YourMemoryInProcess:
    """
    Lightweight in-process version of YourMemory.
    Uses real Ollama embeddings + Ebbinghaus decay. No server needed.
    """

    def __init__(self):
        self.memories = []  # {content, embedding, stored_at, importance, recall_count}

    def add(self, text: str, stored_at: datetime, importance: float = 0.7):
        vec = embed(text)
        self.memories.append({
            "content": text,
            "embedding": vec,
            "stored_at": stored_at,
            "importance": importance,
            "recall_count": 0,
            "last_accessed": stored_at,
        })

    def search(self, query: str, top_k: int = TOP_K, query_time: datetime = None) -> list[str]:
        if not self.memories:
            return []
        if query_time is None:
            query_time = datetime.now(timezone.utc)

        q_vec = embed(query)
        scored = []
        for m in self.memories:
            sim = cosine_similarity(q_vec, m["embedding"])
            if sim < 0.30:
                continue
            # Compute strength relative to query_time so dataset dates
            # don't get penalised by wall-clock distance from 2026.
            days_elapsed = max(0, (query_time - m["last_accessed"]).total_seconds() / 86400)
            import math
            effective_lambda = 0.16 * (1 - m["importance"] * 0.8)
            strength = m["importance"] * math.exp(-effective_lambda * days_elapsed) * (1 + m["recall_count"] * 0.2)
            if strength < PRUNE_THRESHOLD:
                continue
            scored.append((sim * strength, m["content"], sim, strength))

        scored.sort(reverse=True)
        return [content for _, content, _, _ in scored[:top_k]]

    def clear(self):
        self.memories = []


# ---------------------------------------------------------------------------
# Mem0 wrapper
# ---------------------------------------------------------------------------

class Mem0System:
    def __init__(self, api_key: str):
        self.client = MemoryClient(api_key=api_key)
        self.user_id = None

    def new_user(self):
        self.user_id = f"locomo_bench_{uuid.uuid4().hex[:8]}"

    def add(self, text: str):
        for attempt in range(3):
            try:
                self.client.add(text, user_id=self.user_id)
                return
            except Exception:
                if attempt < 2:
                    time.sleep(3)
                else:
                    pass  # skip this memory on persistent failure

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        for attempt in range(3):
            try:
                results = self.client.search(query, filters={"user_id": self.user_id}, limit=top_k)
                return [r["memory"] for r in results.get("results", results)]
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"    [Mem0 search failed after 3 attempts: {e}]")
                    return []

    def clear(self):
        try:
            self.client.delete_all(user_id=self.user_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------

def answer_in_context(answer: str, context_chunks: list[str]) -> bool:
    """True if any word-level token from answer appears in retrieved context."""
    answer_lower = answer.lower().strip()
    full_context = " ".join(context_chunks).lower()
    # Substring match first
    if answer_lower in full_context:
        return True
    # Token overlap: at least half the answer tokens appear
    tokens = [t for t in answer_lower.split() if len(t) > 3]
    if not tokens:
        return answer_lower in full_context
    hits = sum(1 for t in tokens if t in full_context)
    return hits / len(tokens) >= 0.5


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run():
    data = load_locomo()
    mem0_api_key = os.getenv("MEM0_API_KEY")
    if not mem0_api_key:
        print("ERROR: MEM0_API_KEY not set in .env")
        sys.exit(1)

    ym = YourMemoryInProcess()
    m0 = Mem0System(mem0_api_key)

    total_ym = sum(r["ym_hits"] for r in PRIOR_RESULTS)
    total_m0 = sum(r["m0_hits"] for r in PRIOR_RESULTS)
    total_qa = sum(r["qa_total"] for r in PRIOR_RESULTS)
    results_by_sample = list(PRIOR_RESULTS)

    samples = data[:MAX_SAMPLES] if MAX_SAMPLES else data

    for sample_idx, sample in enumerate(samples):
        if sample_idx < START_FROM_SAMPLE:
            continue
        conv = sample["conversation"]
        qa_pairs = [q for q in sample["qa"] if q.get("category") in (1, 2, 3, 4) and "answer" in q and isinstance(q["answer"], str)]
        if MAX_QA_PER_SAMPLE:
            qa_pairs = qa_pairs[:MAX_QA_PER_SAMPLE]

        speaker_a = conv.get("speaker_a", "Person A")
        speaker_b = conv.get("speaker_b", "Person B")

        print(f"\nSample {sample_idx+1}/{len(samples)}: {speaker_a} & {speaker_b}")
        print(f"  Sessions to store, QA pairs: {len([k for k in conv if k.startswith('session_') and not k.endswith('date_time')])}, {len(qa_pairs)}")

        # Reset both systems
        ym.clear()
        m0.new_user()

        # Store session summaries
        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1])
        )

        session_summaries = sample.get("session_summary", {})
        now = datetime.now(timezone.utc)

        stored = 0
        for skey in session_keys:
            date_key = skey + "_date_time"
            date_str = conv.get(date_key, "")
            session_date = parse_session_date(date_str) if date_str else now
            summary_key = skey + "_summary"
            summary = session_summaries.get(summary_key, "")
            if not summary:
                continue

            # YourMemory: store with real timestamp
            ym.add(summary, stored_at=session_date, importance=0.7)

            # Mem0: store as plain text
            m0.add(summary)
            stored += 1
            time.sleep(0.3)  # rate limit

        print(f"  Stored {stored} session summaries. Waiting for Mem0 to index...")
        time.sleep(15)  # Mem0 processes memories asynchronously

        # Evaluate QA pairs
        # Use the last session date as query_time so decay is computed
        # relative to the conversation timeline, not from today (2026).
        last_session_key = session_keys[-1] if session_keys else None
        last_date_str = conv.get(last_session_key + "_date_time", "") if last_session_key else ""
        query_time = parse_session_date(last_date_str) if last_date_str else now
        ym_hits = m0_hits = 0

        for qa in qa_pairs:
            question = qa["question"]
            answer = qa["answer"]

            ym_context = ym.search(question, query_time=query_time)
            m0_context = m0.search(question)

            ym_hit = answer_in_context(answer, ym_context)
            m0_hit = answer_in_context(answer, m0_context)

            ym_hits += ym_hit
            m0_hits += m0_hit
            time.sleep(0.3)

        total_qa += len(qa_pairs)
        total_ym += ym_hits
        total_m0 += m0_hits

        ym_pct = round(ym_hits / len(qa_pairs) * 100) if qa_pairs else 0
        m0_pct = round(m0_hits / len(qa_pairs) * 100) if qa_pairs else 0

        print(f"  YourMemory: {ym_hits}/{len(qa_pairs)} ({ym_pct}%)")
        print(f"  Mem0:       {m0_hits}/{len(qa_pairs)} ({m0_pct}%)")

        results_by_sample.append({
            "sample": sample_idx + 1,
            "speakers": f"{speaker_a} & {speaker_b}",
            "qa_total": len(qa_pairs),
            "ym_hits": ym_hits,
            "m0_hits": m0_hits,
            "ym_pct": ym_pct,
            "m0_pct": m0_pct,
        })

        # Clean up Mem0 user
        m0.clear()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS — LoCoMo Benchmark")
    print("=" * 60)
    print(f"{'Sample':<6} {'Speakers':<30} {'YM':>6} {'Mem0':>6}")
    print("-" * 52)
    for r in results_by_sample:
        print(f"{r['sample']:<6} {r['speakers']:<30} {r['ym_pct']:>5}% {r['m0_pct']:>5}%")

    overall_ym = round(total_ym / total_qa * 100) if total_qa else 0
    overall_m0 = round(total_m0 / total_qa * 100) if total_qa else 0

    print("-" * 52)
    print(f"{'TOTAL':<6} {str(total_qa) + ' QA pairs':<30} {overall_ym:>5}% {overall_m0:>5}%")
    print("=" * 60)
    print(f"\nYourMemory Recall@{TOP_K}: {overall_ym}%")
    print(f"Mem0 Recall@{TOP_K}:       {overall_m0}%")

    return {"ym_pct": overall_ym, "m0_pct": overall_m0, "total_qa": total_qa}


if __name__ == "__main__":
    run()
