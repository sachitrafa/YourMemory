"""
YourMemory adapter for the official LongMemEval evaluation framework.
https://github.com/xiaowu0162/LongMemEval

Outputs a JSONL file in the exact format expected by:
  python /tmp/LongMemEval/src/evaluation/print_retrieval_metrics.py <outfile>

Usage:
    # Full run (~15-30 min on CPU, 500 questions × ~40 sessions):
    python benchmarks/longmemeval_official.py

    # Quick smoke test (first 50 questions):
    python benchmarks/longmemeval_official.py --limit 50

    # Use oracle dataset (trivially easy — only evidence sessions):
    python benchmarks/longmemeval_official.py --data ~/Desktop/longmemeval/longmemeval_oracle.json

    # Print metrics after run:
    python /tmp/LongMemEval/src/evaluation/print_retrieval_metrics.py benchmarks/longmemeval_official_results.jsonl
"""

import sys
import os
import json
import argparse
import time
import numpy as np
from datetime import datetime, timezone
from dateutil import parser as dateparser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("YOURMEMORY_DB", os.path.expanduser("~/.yourmemory/memories.duckdb"))

from src.services.embed import embed


# Inlined from /tmp/LongMemEval/src/retrieval/eval_utils.py to avoid src/ namespace collision
def _dcg(relevances, k):
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.0


def evaluate_retrieval(rankings, correct_docs, corpus_ids, k=10):
    relevances = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    sorted_relevances = [relevances[idx] for idx in rankings[:k]]
    ideal_relevance = sorted(relevances, reverse=True)
    ideal_dcg = _dcg(ideal_relevance, k)
    actual_dcg = _dcg(sorted_relevances, k)
    ndcg_score = (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

    recalled_docs = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(doc in recalled_docs for doc in correct_docs))
    recall_all = float(all(doc in recalled_docs for doc in correct_docs))
    return recall_any, recall_all, ndcg_score

DEFAULT_DATA = os.path.expanduser("~/Desktop/longmemeval/longmemeval_s_cleaned.json")
OUT_FILE     = os.path.join(os.path.dirname(__file__), "longmemeval_official_results.jsonl")


def cosine(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def parse_date(s) -> datetime:
    try:
        return dateparser.parse(str(s)).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def session_to_text(session: list) -> str:
    """Flatten session to user-turn text only (matches LongMemEval's flat-index convention)."""
    return " ".join(
        turn["content"]
        for turn in session
        if turn.get("role") == "user" and turn.get("content", "").strip()
    )


def process_instance(entry: dict) -> dict:
    """Run YourMemory retrieval on one LongMemEval instance and return the result record."""
    sessions    = entry["haystack_sessions"]
    sess_ids    = entry["haystack_session_ids"]
    dates       = entry["haystack_dates"]
    question    = entry["question"]
    q_date      = parse_date(entry.get("question_date", ""))

    # Build corpus (session-level, user turns only — same as flat-bm25 baseline)
    corpus, corpus_ids, corpus_timestamps = [], [], []
    for sess, sid, date_str in zip(sessions, sess_ids, dates):
        text = session_to_text(sess)
        if not text.strip():
            # empty session: keep as placeholder so IDs stay aligned
            text = " "
        corpus.append(text)
        corpus_ids.append(sid)
        corpus_timestamps.append(date_str)

    corpus_ids_arr = np.array(corpus_ids)

    # Correct docs: sessions whose ID contains "answer"
    correct_docs = list(set(sid for sid in corpus_ids if "answer" in sid))

    # Embed question
    q_emb = embed(question)

    # Score each session: pure cosine similarity.
    # Decay is a production pruning signal, not a retrieval ranking signal —
    # applying it here would penalise answer sessions that happen to be older
    # than filler sessions, distorting recall.
    scores = []
    for text, sid, date_str in zip(corpus, corpus_ids, corpus_timestamps):
        emb = embed(text[:2000])
        scores.append(cosine(q_emb, emb))

    scores = np.array(scores)
    rankings = np.argsort(scores)[::-1].tolist()   # descending, list of int indices

    # Evaluate at all k values required by the framework
    metrics_session = {}
    for k in [1, 3, 5, 10, 30, 50]:
        recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, corpus_ids_arr, k=k)
        metrics_session[f"recall_any@{k}"] = recall_any
        metrics_session[f"recall_all@{k}"] = recall_all
        metrics_session[f"ndcg_any@{k}"]   = ndcg_any

    return {
        "question_id":          entry["question_id"],
        "question_type":        entry["question_type"],
        "question":             question,
        "answer":               entry.get("answer", ""),
        "question_date":        entry.get("question_date", ""),
        "haystack_dates":       dates,
        "haystack_sessions":    sessions,
        "haystack_session_ids": sess_ids,
        "answer_session_ids":   entry.get("answer_session_ids", []),
        "retrieval_results": {
            "query": question,
            "ranked_items": [
                {
                    "corpus_id": corpus_ids[rid],
                    "text":      corpus[rid],
                    "timestamp": corpus_timestamps[rid],
                }
                for rid in rankings
            ],
            "metrics": {
                "session": metrics_session,
                "turn":    {},
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DEFAULT_DATA)
    parser.add_argument("--out",   default=OUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: dataset not found at {args.data}")
        print("Download with:")
        print("  mkdir -p ~/Desktop/longmemeval")
        print("  curl -L https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s_cleaned.json \\")
        print("       -o ~/Desktop/longmemeval/longmemeval_s_cleaned.json")
        sys.exit(1)

    data = json.load(open(args.data))
    if args.limit:
        data = data[:args.limit]

    print(f"YourMemory × LongMemEval — {len(data)} questions | session-level retrieval")
    print(f"Dataset : {args.data}")
    print(f"Output  : {args.out}")
    print("=" * 70)

    out_f   = open(args.out, "w")
    start   = time.time()
    n_done  = 0

    recall5_running = []

    for i, entry in enumerate(data):
        result = process_instance(entry)
        print(json.dumps(result), file=out_f)
        out_f.flush()

        r5 = result["retrieval_results"]["metrics"]["session"].get("recall_all@5", 0.0)
        recall5_running.append(r5)
        n_done += 1

        if (i + 1) % 10 == 0 or i == len(data) - 1:
            elapsed = time.time() - start
            eta     = (elapsed / n_done) * (len(data) - n_done)
            avg_r5  = float(np.mean(recall5_running))
            print(f"  [{i+1:>3}/{len(data)}]  recall_all@5: {avg_r5:.3f}  |  "
                  f"elapsed {elapsed:.0f}s  eta {eta:.0f}s")

    out_f.close()

    print()
    print("=" * 70)
    print(f"Done. Results written to: {args.out}")
    print()
    print("Print official metrics with:")
    print(f"  python /tmp/LongMemEval/src/evaluation/print_retrieval_metrics.py {args.out}")
    print()
    print(f"Total time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
