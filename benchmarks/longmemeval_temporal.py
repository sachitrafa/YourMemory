"""
LongMemEval — Temporal Boost Ablation
======================================

Compares Recall@5 WITH vs WITHOUT the temporal boost from src/services/temporal.py
across all LongMemEval question types, spotlighting the "temporal-reasoning" subset.

Key difference from locomo_temporal.py:
  - LongMemEval has a dedicated `temporal-reasoning` question type — these questions
    are specifically designed to require time-window reasoning ("last week", "recently",
    "last month"), making them the natural target for the temporal boost.
  - Temporal range is anchored to `question_date` (not datetime.now) so that
    "last week" resolves correctly relative to when the question was asked.

Scoring:
  base  = cosine(q, memory) × ebbinghaus_strength
  boost = base + TEMPORAL_BOOST (0.25) if session_date falls in resolved window

USAGE:
  python benchmarks/longmemeval_temporal.py
  python benchmarks/longmemeval_temporal.py --limit 100  # quick smoke test

DATA: ~/Desktop/longmemeval/longmemeval_oracle.json
RESULTS: benchmarks/results/longmemeval_temporal.json
"""

import sys, os, json, re, time, argparse, math
import numpy as np
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("YOURMEMORY_DB", os.path.expanduser("~/.yourmemory/memories.duckdb"))

from src.services.embed import embed
from src.services.decay import compute_strength
from src.services.temporal import TEMPORAL_BOOST, _PATTERNS

DEFAULT_DATA = os.path.expanduser("~/Desktop/longmemeval/longmemeval_oracle.json")
TOP_K        = 5
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "longmemeval_temporal.json")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def parse_date(s: str) -> datetime:
    try:
        return dateparser.parse(str(s)).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def session_to_text(session: list) -> str:
    parts = []
    for turn in session:
        role    = turn.get("role", "")
        content = turn.get("content", "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def detect_temporal_range_anchored(
    query: str, anchor: datetime
) -> tuple[datetime, datetime] | None:
    """
    Like detect_temporal_range but anchored to a specific date instead of now.
    This is critical for LongMemEval: "last week" must resolve relative to
    question_date, not today.
    """
    q_lower = query.lower()

    # spaCy pass — try to detect entity text first, then match patterns
    try:
        from src.services.extract import _nlp
        if _nlp is not None:
            doc = _nlp(query)
            for ent in doc.ents:
                if ent.label_ in ("DATE", "TIME"):
                    for pattern, days_ago, _ in _PATTERNS:
                        if re.search(pattern, ent.text.lower()):
                            return (anchor - timedelta(days=days_ago), anchor)
    except Exception:
        pass

    # Regex fallback directly on query
    for pattern, days_ago, _ in _PATTERNS:
        if re.search(pattern, q_lower):
            return (anchor - timedelta(days=days_ago), anchor)
    return None


def temporal_boost_for(session_date: datetime, date_range) -> float:
    if not date_range:
        return 0.0
    start, end = date_range
    if session_date.tzinfo is None:
        session_date = session_date.replace(tzinfo=timezone.utc)
    return TEMPORAL_BOOST if start <= session_date <= end else 0.0


def ci95(h, n):
    if n == 0:
        return (0, 0)
    p = h / n
    z = 1.96
    lo = (p + z*z/(2*n) - z*((p*(1-p)+z*z/(4*n))/n)**0.5) / (1+z*z/n)
    hi = (p + z*z/(2*n) + z*((p*(1-p)+z*z/(4*n))/n)**0.5) / (1+z*z/n)
    return round(lo*100, 1), round(hi*100, 1)


# ── Core scoring ──────────────────────────────────────────────────────────────

def run_instance(instance: dict) -> tuple[bool, bool]:
    """
    Returns (hit_base, hit_boost) — whether the answer session appears in top-K
    with and without the temporal boost.
    """
    sessions    = instance["haystack_sessions"]
    session_ids = instance["haystack_session_ids"]
    dates       = instance["haystack_dates"]
    answer_ids  = set(instance["answer_session_ids"])
    question    = instance["question"]
    q_date      = parse_date(instance.get("question_date", ""))

    # Resolve temporal window anchored to question_date
    date_range = detect_temporal_range_anchored(question, q_date)

    # Build memory bank
    memories = []
    for sess, sid, date_str in zip(sessions, session_ids, dates):
        text = session_to_text(sess)
        if not text.strip():
            continue
        emb      = embed(text[:2000])
        sess_dt  = parse_date(date_str)
        memories.append({
            "session_id":    sid,
            "embedding":     emb,
            "last_accessed": sess_dt,
            "recall_count":  0,
            "importance":    0.7,
            "category":      "fact",
            "boost":         temporal_boost_for(sess_dt, date_range),
        })

    if not memories:
        return False, False

    q_emb = embed(question)

    scored_base  = []
    scored_boost = []
    for m in memories:
        sim      = cosine(q_emb, m["embedding"])
        strength = compute_strength(
            last_accessed_at=m["last_accessed"],
            recall_count=m["recall_count"],
            importance=m["importance"],
            category=m["category"],
        )
        base_score  = sim * strength
        boost_score = base_score + m["boost"]
        scored_base.append((base_score,  m["session_id"]))
        scored_boost.append((boost_score, m["session_id"]))

    scored_base.sort(reverse=True)
    scored_boost.sort(reverse=True)

    top_base  = {sid for _, sid in scored_base[:TOP_K]}
    top_boost = {sid for _, sid in scored_boost[:TOP_K]}

    return bool(top_base & answer_ids), bool(top_boost & answer_ids)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: dataset not found at {args.data}")
        print("Download with:")
        print("  curl -L https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json \\")
        print("       -o ~/Desktop/longmemeval/longmemeval_oracle.json")
        sys.exit(1)

    data = json.load(open(args.data))
    if args.limit:
        data = data[:args.limit]

    print()
    print("═" * 72)
    print("  LongMemEval — Temporal Boost Ablation   Recall@5")
    print(f"  Base: cosine × decay   |   Boost: base + {TEMPORAL_BOOST} on window match")
    print(f"  Temporal window anchored to question_date (not today)")
    print("═" * 72)

    # Per-type tracking
    type_base  = {}  # qtype → list of 0/1
    type_boost = {}

    total_base  = 0
    total_boost = 0
    n_boosted_q = 0   # questions where temporal range was detected

    start = time.time()

    for i, instance in enumerate(data):
        qtype = instance["question_type"]
        hit_b, hit_k = run_instance(instance)

        type_base.setdefault(qtype,  []).append(int(hit_b))
        type_boost.setdefault(qtype, []).append(int(hit_k))
        total_base  += int(hit_b)
        total_boost += int(hit_k)

        # Check if temporal range was triggered for this question
        q_date = parse_date(instance.get("question_date", ""))
        if detect_temporal_range_anchored(instance["question"], q_date):
            n_boosted_q += 1

        if (i + 1) % 25 == 0 or i == len(data) - 1:
            elapsed = time.time() - start
            eta     = (elapsed / (i + 1)) * (len(data) - i - 1)
            print(f"  [{i+1:>3}/{len(data)}]  base {total_base/(i+1):.1%}  boost {total_boost/(i+1):.1%}  |  {elapsed:.0f}s  eta {eta:.0f}s")

    n = len(data)
    print()
    print("═" * 72)
    print("  FINAL RESULTS — LongMemEval Temporal Boost Ablation")
    print("═" * 72)
    print(f"  Questions with temporal range triggered: {n_boosted_q}/{n} ({n_boosted_q/n:.0%})")
    print()

    col = 20
    header = f"  {'Question Type':<36}  {'Base':>{col}}  {'+ Boost':>{col}}  {'Δpp':>6}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    by_type_out = {}
    for qtype in sorted(type_base.keys()):
        b_hits = sum(type_base[qtype])
        k_hits = sum(type_boost[qtype])
        n_t    = len(type_base[qtype])
        b_pct  = b_hits / n_t * 100
        k_pct  = k_hits / n_t * 100
        delta  = k_pct - b_pct
        marker = " ◀" if qtype == "temporal-reasoning" else ""
        print(f"  {qtype:<36}  {b_pct:>{col-1}.1f}%  {k_pct:>{col-1}.1f}%  {delta:>+5.1f}pp{marker}")
        by_type_out[qtype] = {
            "n":         n_t,
            "base_pct":  round(b_pct, 2),
            "boost_pct": round(k_pct, 2),
            "delta_pp":  round(delta, 2),
            "ci95_base":  ci95(b_hits, n_t),
            "ci95_boost": ci95(k_hits, n_t),
        }

    print("  " + "─" * (len(header) - 2))
    overall_b  = total_base  / n * 100
    overall_k  = total_boost / n * 100
    overall_d  = overall_k - overall_b
    print(f"  {'OVERALL':<36}  {overall_b:>{col-1}.1f}%  {overall_k:>{col-1}.1f}%  {overall_d:>+5.1f}pp")
    print()

    # Temporal-reasoning deep dive
    if "temporal-reasoning" in type_base:
        tr_b = sum(type_base["temporal-reasoning"])
        tr_k = sum(type_boost["temporal-reasoning"])
        tr_n = len(type_base["temporal-reasoning"])
        print(f"  Temporal-reasoning CI95 base : {ci95(tr_b, tr_n)}")
        print(f"  Temporal-reasoning CI95 boost: {ci95(tr_k, tr_n)}")
        print()

    print(f"  Overall CI95 base : {ci95(total_base,  n)}")
    print(f"  Overall CI95 boost: {ci95(total_boost, n)}")
    print()
    print(f"  Total time: {time.time()-start:.1f}s")

    output = {
        "benchmark":            "LongMemEval — Temporal Boost Ablation",
        "date":                 __import__("datetime").datetime.now(timezone.utc).isoformat(),
        "top_k":                TOP_K,
        "total_questions":      n,
        "temporal_boost_value": TEMPORAL_BOOST,
        "questions_with_temporal_range": n_boosted_q,
        "overall": {
            "base_pct":  round(overall_b, 2),
            "boost_pct": round(overall_k, 2),
            "delta_pp":  round(overall_d, 2),
            "ci95_base":  ci95(total_base,  n),
            "ci95_boost": ci95(total_boost, n),
        },
        "by_type": by_type_out,
        "methodology": {
            "base":  "cosine(q, memory) × ebbinghaus_strength",
            "boost": "base + 0.25 if session_date within resolved temporal window",
            "anchor": "question_date (not datetime.now)",
            "temporal_detection": "spaCy DATE/TIME entities + regex patterns",
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
