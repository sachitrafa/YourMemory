"""
LoCoMo Real Benchmark: YourMemory vs Zep Cloud
-----------------------------------------------
Uses the full LoCoMo dataset (10 samples, all QA pairs, categories 1-4).
Feeds session summaries into both systems, then evaluates recall accuracy.

YourMemory: in-process (Ollama embeddings + Ebbinghaus decay)
Zep:        Zep Cloud (hosted, no Docker needed) — extracts facts via LLM

Metric: Recall@5 — does the correct answer appear in retrieved context?

Setup:
    1. Set ZEP_API_KEY in .env (get from app.getzep.com)
    2. Ollama running locally with nomic-embed-text pulled
    3. LoCoMo dataset at ~/Desktop/locomo/data/locomo10.json

Usage:
    python benchmarks/locomo_zep.py
"""

import sys
import os
import json
import time
import math
import uuid
import numpy as np
from datetime import datetime, timezone
from dateutil import parser as dateparser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.services.embed import embed

LOCOMO_PATH = os.path.expanduser("~/Desktop/locomo/data/locomo10.json")
PRUNE_THRESHOLD = 0.05
TOP_K = 5
SIMILARITY_THRESHOLD = 0.30
ZEP_API_KEY = os.getenv("ZEP_API_KEY", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def parse_session_date(date_str: str) -> datetime:
    try:
        return dateparser.parse(date_str, dayfirst=True).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def load_locomo():
    with open(LOCOMO_PATH) as f:
        return json.load(f)


def answer_in_context(answer: str, context_chunks: list) -> bool:
    answer_lower = answer.lower().strip()
    full_context = " ".join(str(c) for c in context_chunks).lower()
    if answer_lower in full_context:
        return True
    tokens = [t for t in answer_lower.split() if len(t) > 3]
    if not tokens:
        return answer_lower in full_context
    hits = sum(1 for t in tokens if t in full_context)
    return hits / len(tokens) >= 0.5


# ---------------------------------------------------------------------------
# YourMemory in-process store
# ---------------------------------------------------------------------------

class YourMemoryInProcess:
    def __init__(self):
        self.memories = []

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

    def search(self, query: str, top_k: int = TOP_K, query_time: datetime = None) -> list:
        if not self.memories:
            return []
        if query_time is None:
            query_time = datetime.now(timezone.utc)

        q_vec = embed(query)
        scored = []
        for m in self.memories:
            sim = cosine_similarity(q_vec, m["embedding"])
            if sim < SIMILARITY_THRESHOLD:
                continue
            days_elapsed = max(0, (query_time - m["last_accessed"]).total_seconds() / 86400)
            effective_lambda = 0.16 * (1 - m["importance"] * 0.8)
            strength = m["importance"] * math.exp(-effective_lambda * days_elapsed) * (1 + m["recall_count"] * 0.2)
            if strength < PRUNE_THRESHOLD:
                continue
            scored.append((sim * strength, m["content"]))

        scored.sort(reverse=True)
        return [content for _, content in scored[:top_k]]

    def clear(self):
        self.memories = []


# ---------------------------------------------------------------------------
# Zep wrapper
# ---------------------------------------------------------------------------

class ZepSystem:
    """
    Wraps Zep Cloud for the benchmark.
    Each sample gets its own user + thread to isolate memory.
    Zep extracts facts from messages and returns context per query.
    """

    def __init__(self):
        from zep_cloud.client import Zep
        self.client = Zep(api_key=ZEP_API_KEY)
        self.thread_id = None
        self.user_id = None

    def new_session(self):
        from zep_cloud.types import Message
        self.user_id = f"bench_{uuid.uuid4().hex[:8]}"
        self.thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        try:
            self.client.user.add(user_id=self.user_id)
            self.client.thread.create(thread_id=self.thread_id, user_id=self.user_id)
        except Exception as e:
            print(f"    [Zep session setup error: {e}]")

    def add(self, text: str):
        from zep_cloud.types import Message
        try:
            self.client.thread.add_messages(
                thread_id=self.thread_id,
                messages=[Message(role="user", role_type="user", content=text)]
            )
        except Exception as e:
            print(f"    [Zep add error: {e}]")

    def search(self, query: str, top_k: int = TOP_K) -> list:
        try:
            ctx = self.client.thread.get_user_context(thread_id=self.thread_id)
            if ctx and ctx.context:
                return [ctx.context]
            return []
        except Exception as e:
            print(f"    [Zep search error: {e}]")
            return []

    def clear(self):
        try:
            self.client.user.delete(user_id=self.user_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run():
    if not ZEP_API_KEY:
        print("ERROR: ZEP_API_KEY not set in .env")
        sys.exit(1)

    data = load_locomo()

    ym = YourMemoryInProcess()
    zp = ZepSystem()

    total_ym = total_zp = total_qa = 0
    results_by_sample = []

    for sample_idx, sample in enumerate(data):
        conv = sample["conversation"]
        qa_pairs = [
            q for q in sample["qa"]
            if q.get("category") in (1, 2, 3, 4)
            and "answer" in q
            and isinstance(q["answer"], str)
        ]

        speaker_a = conv.get("speaker_a", "Person A")
        speaker_b = conv.get("speaker_b", "Person B")

        print(f"\nSample {sample_idx + 1}/10: {speaker_a} & {speaker_b} — {len(qa_pairs)} QA pairs")

        ym.clear()
        zp.new_session()

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
            summary = session_summaries.get(skey + "_summary", "")
            if not summary:
                continue

            ym.add(summary, stored_at=session_date, importance=0.7)
            zp.add(summary)
            stored += 1
            time.sleep(0.1)

        print(f"  Stored {stored} session summaries. Waiting for Zep to extract facts...")
        time.sleep(25)

        last_session_key = session_keys[-1] if session_keys else None
        last_date_str = conv.get(last_session_key + "_date_time", "") if last_session_key else ""
        query_time = parse_session_date(last_date_str) if last_date_str else now

        ym_hits = zp_hits = 0

        for i, qa in enumerate(qa_pairs):
            question = qa["question"]
            answer = qa["answer"]

            ym_context = ym.search(question, query_time=query_time)
            zp_context = zp.search(question)

            ym_hits += answer_in_context(answer, ym_context)
            zp_hits += answer_in_context(answer, zp_context)

            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(qa_pairs)} evaluated...")

        total_qa += len(qa_pairs)
        total_ym += ym_hits
        total_zp += zp_hits

        ym_pct = round(ym_hits / len(qa_pairs) * 100) if qa_pairs else 0
        zp_pct = round(zp_hits / len(qa_pairs) * 100) if qa_pairs else 0

        print(f"  YourMemory: {ym_hits}/{len(qa_pairs)} ({ym_pct}%)")
        print(f"  Zep:        {zp_hits}/{len(qa_pairs)} ({zp_pct}%)")

        results_by_sample.append({
            "sample": sample_idx + 1,
            "speakers": f"{speaker_a} & {speaker_b}",
            "qa_total": len(qa_pairs),
            "ym_hits": ym_hits,
            "zp_hits": zp_hits,
            "ym_pct": ym_pct,
            "zp_pct": zp_pct,
        })

        zp.clear()

    overall_ym = round(total_ym / total_qa * 100) if total_qa else 0
    overall_zp = round(total_zp / total_qa * 100) if total_qa else 0

    print("\n" + "=" * 65)
    print("FINAL RESULTS — LoCoMo Full Benchmark")
    print(f"{'Sample':<6} {'Speakers':<30} {'YourMemory':>12} {'Zep':>6}")
    print("-" * 58)
    for r in results_by_sample:
        print(f"{r['sample']:<6} {r['speakers']:<30} {r['ym_pct']:>10}% {r['zp_pct']:>4}%")

    print("-" * 58)
    print(f"{'TOTAL':<6} {str(total_qa) + ' QA pairs':<30} {overall_ym:>10}% {overall_zp:>4}%")
    print("=" * 65)
    print(f"\nYourMemory Recall@{TOP_K}: {overall_ym}%")
    print(f"Zep        Recall@{TOP_K}: {overall_zp}%")

    return {"ym_pct": overall_ym, "zp_pct": overall_zp, "total_qa": total_qa}


if __name__ == "__main__":
    run()
