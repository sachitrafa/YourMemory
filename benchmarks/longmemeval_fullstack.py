"""
YourMemory FULL STACK LongMemEval benchmark.

Runs the complete production retrieval pipeline in-memory per question:

  Round 1 — vector cosine filtered at threshold (0.50 primary, 0.20 fallback)
             + BM25 re-rank (additive, same as production hybrid scoring)

  Round 2 — in-memory graph BFS expansion from Round 1 seeds
             edges: cosine × verb_weight (0.5 default, same as spaCy fallback)
             score: W_VECTOR × edge_weight × strength (decay on graph nodes)

  Merge   — combined pool re-ranked by score, top-k taken

This faithfully reproduces retrieve() logic post all fixes:
  - Decay excluded from direct match ranking (removed from _score_candidates)
  - Graph nodes scored with edge_weight × strength (not flat 0.5)
  - Graph nodes capped below REINFORCE_THRESHOLD to prevent propagate_recall
  - W_VECTOR applied to graph scores so they can't outrank strong direct hits

Usage:
    python benchmarks/longmemeval_fullstack.py
    python benchmarks/longmemeval_fullstack.py --limit 50
    python benchmarks/longmemeval_fullstack.py --data ~/Desktop/longmemeval/longmemeval_oracle.json

    # Print official metrics:
    python /tmp/LongMemEval/src/evaluation/print_retrieval_metrics.py \
        benchmarks/longmemeval_fullstack_results.jsonl
"""

import sys
import os
import json
import argparse
import time
import hashlib
from collections import deque
from datetime import datetime, timezone

import numpy as np
from dateutil import parser as dateparser
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("YOURMEMORY_DB", os.path.expanduser("~/.yourmemory/memories.duckdb"))

from src.services.embed import embed
from src.services.decay import compute_strength

# ── Production constants (mirror retrieve.py) ─────────────────────────────────
SIMILARITY_THRESHOLD          = 0.50
SIMILARITY_THRESHOLD_FALLBACK = 0.20
REINFORCE_THRESHOLD           = 0.75
W_BM25                        = 0.4
W_VECTOR                      = 0.6
GRAPH_MIN_SIM                 = 0.4    # min cosine to create a graph edge
GRAPH_VERB_WEIGHT             = 0.5    # default when spaCy unavailable
GRAPH_MAX_DEPTH               = 2

DEFAULT_DATA = os.path.expanduser("~/Desktop/longmemeval/longmemeval_s_cleaned.json")
OUT_FILE     = os.path.join(os.path.dirname(__file__), "longmemeval_fullstack_results.jsonl")


# ── Embedding cache (avoids re-encoding duplicate session texts) ───────────────

_embed_cache: dict[str, list] = {}


def cached_embed(text: str) -> list:
    key = hashlib.md5(text.encode()).hexdigest()
    if key not in _embed_cache:
        _embed_cache[key] = embed(text)
    return _embed_cache[key]


# ── Eval utils (inlined to avoid src/ namespace collision) ────────────────────

def _dcg(relevances, k):
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.0


def evaluate_retrieval(rankings, correct_docs, corpus_ids, k=10):
    relevances        = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    sorted_relevances = [relevances[idx] for idx in rankings[:k]]
    ideal_relevance   = sorted(relevances, reverse=True)
    ideal_dcg         = _dcg(ideal_relevance, k)
    actual_dcg        = _dcg(sorted_relevances, k)
    ndcg_score        = (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0
    recalled          = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any        = float(any(d in recalled for d in correct_docs))
    recall_all        = float(all(d in recalled for d in correct_docs))
    return recall_any, recall_all, ndcg_score


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    return " ".join(
        t["content"] for t in session
        if t.get("role") == "user" and t.get("content", "").strip()
    )


def normalize_bm25(score: float) -> float:
    """Match production: score / (score + 1)."""
    return score / (score + 1.0) if score > 0 else 0.0


# ── Graph (in-memory, per question) ──────────────────────────────────────────

def build_adjacency(sim_matrix: np.ndarray) -> dict[int, list]:
    """
    Build adjacency list from pairwise cosine sim matrix.
    Edge weight = cosine × GRAPH_VERB_WEIGHT (spaCy fallback).
    Only edges with cosine >= GRAPH_MIN_SIM are created.
    """
    n = sim_matrix.shape[0]
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and sim_matrix[i, j] >= GRAPH_MIN_SIM:
                ew = round(float(sim_matrix[i, j]) * GRAPH_VERB_WEIGHT, 4)
                adj[i].append((j, ew))
    return adj


def bfs_expand(seeds: list[int], adj: dict, max_depth: int = GRAPH_MAX_DEPTH) -> dict[int, float]:
    """
    BFS from seed indices; return {idx: best_edge_weight} for new nodes.
    Mirrors expand_with_graph() logic.
    """
    visited    = set(seeds)
    candidates: dict[int, float] = {}
    queue      = deque((s, 0) for s in seeds)
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nbr, ew in adj.get(node, []):
            if nbr not in visited:
                visited.add(nbr)
                if nbr not in candidates or candidates[nbr] < ew:
                    candidates[nbr] = ew
                queue.append((nbr, depth + 1))
    return candidates


# ── Core retrieval per question ───────────────────────────────────────────────

def run_instance(entry: dict) -> dict:
    sessions    = entry["haystack_sessions"]
    sess_ids    = entry["haystack_session_ids"]
    dates       = entry["haystack_dates"]
    question    = entry["question"]
    q_date      = parse_date(entry.get("question_date", ""))

    n = len(sessions)

    # ── Build corpus ──────────────────────────────────────────────────────
    texts = [session_to_text(s) or " " for s in sessions]

    # ── Embed everything ──────────────────────────────────────────────────
    q_emb   = cached_embed(question)
    s_embs  = [cached_embed(t[:2000]) for t in texts]

    # ── Pairwise cosine matrix (for graph) ────────────────────────────────
    emb_mat = np.array(s_embs, dtype=np.float32)
    norms   = np.linalg.norm(emb_mat, axis=1, keepdims=True)
    normed  = emb_mat / np.where(norms > 0, norms, 1.0)
    sim_mat = normed @ normed.T                        # shape (n, n)

    # ── Query cosine similarities ─────────────────────────────────────────
    q_arr = np.array(q_emb, dtype=np.float32)
    q_arr /= max(np.linalg.norm(q_arr), 1e-10)
    q_sims = (normed @ q_arr).tolist()                 # shape (n,)

    # ── BM25 scores ───────────────────────────────────────────────────────
    tokenized_corpus = [t.split() for t in texts]
    bm25             = BM25Okapi(tokenized_corpus)
    raw_bm25         = bm25.get_scores(question.split())
    bm25_norm        = np.array([normalize_bm25(float(s)) for s in raw_bm25])

    # ── Decay (for graph nodes, relative to question date) ───────────────
    # Use q_date as reference so sessions from 2023 aren't penalised by
    # ~1000 wall-clock days from datetime.now() — simulates the system
    # as it would run at the time the question was asked.
    strengths = []
    for date_str in dates:
        accessed  = parse_date(date_str)
        days_old  = max(0.0, (q_date - accessed).total_seconds() / 86400)
        strengths.append(compute_strength(
            last_accessed_at=accessed,
            recall_count=0,
            importance=0.7,
            category="fact",
            active_days=days_old,
        ))
    strengths = np.array(strengths)

    # ── Round 1: cosine filter + BM25 hybrid ─────────────────────────────
    q_sims_arr = np.array(q_sims)
    mask_primary  = q_sims_arr >= SIMILARITY_THRESHOLD
    mask_fallback = q_sims_arr >= SIMILARITY_THRESHOLD_FALLBACK

    mask = mask_primary if mask_primary.any() else mask_fallback
    r1_indices = np.where(mask)[0].tolist()

    # Hybrid score for Round 1 candidates (no decay — direct match)
    r1_scored = []
    for i in r1_indices:
        score = W_BM25 * bm25_norm[i] + W_VECTOR * q_sims_arr[i]
        r1_scored.append((i, score))
    r1_scored.sort(key=lambda x: x[1], reverse=True)

    # Top-k seeds for graph expansion
    top_k_seeds = [i for i, _ in r1_scored[:5]]

    # ── Round 2: graph BFS expansion ─────────────────────────────────────
    adj        = build_adjacency(sim_mat)
    graph_hits = bfs_expand(top_k_seeds, adj)

    # Remove any indices already in Round 1
    r1_set = set(r1_indices)
    r2_scored = []
    for i, ew in graph_hits.items():
        if i in r1_set:
            continue
        # Mirror production: cap below REINFORCE_THRESHOLD, apply W_VECTOR × decay
        sim_capped = min(ew, REINFORCE_THRESHOLD - 0.01)
        score      = W_VECTOR * sim_capped * strengths[i]
        r2_scored.append((i, score, sim_capped))
    r2_scored.sort(key=lambda x: x[1], reverse=True)

    # ── Merge + re-rank ───────────────────────────────────────────────────
    merged: list[tuple[int, float]] = r1_scored + [(i, s) for i, s, _ in r2_scored]
    merged.sort(key=lambda x: x[1], reverse=True)

    TOP_K = 5
    top_indices = [i for i, _ in merged[:TOP_K]]

    # ── Build full ranking (top retrieved first, then remaining by original order) ─
    top_set    = set(top_indices)
    remaining  = [i for i in range(n) if i not in top_set]
    rankings   = top_indices + remaining

    # ── Evaluate ──────────────────────────────────────────────────────────
    corpus_ids   = np.array(sess_ids)
    correct_docs = list(entry.get("answer_session_ids", []))

    metrics_session = {}
    for k in [1, 3, 5, 10, 30, 50]:
        ra, ra_all, ndcg = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
        metrics_session[f"recall_any@{k}"] = ra
        metrics_session[f"recall_all@{k}"] = ra_all
        metrics_session[f"ndcg_any@{k}"]   = ndcg

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
                {"corpus_id": sess_ids[i], "text": texts[i], "timestamp": dates[i]}
                for i in rankings
            ],
            "metrics": {"session": metrics_session, "turn": {}},
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DEFAULT_DATA)
    parser.add_argument("--out",   default=OUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: dataset not found at {args.data}")
        sys.exit(1)

    data = json.load(open(args.data))
    if args.limit:
        data = data[:args.limit]

    print(f"YourMemory FULL STACK × LongMemEval — {len(data)} questions")
    print(f"Pipeline: cosine (thr={SIMILARITY_THRESHOLD}/{SIMILARITY_THRESHOLD_FALLBACK})"
          f" + BM25 (w={W_BM25}) + graph BFS (depth={GRAPH_MAX_DEPTH})")
    print(f"Dataset : {args.data}")
    print(f"Output  : {args.out}")
    print("=" * 70)

    out_f  = open(args.out, "w")
    start  = time.time()
    r5_all = []

    for i, entry in enumerate(data):
        result = run_instance(entry)
        print(json.dumps(result), file=out_f)
        out_f.flush()

        r5 = result["retrieval_results"]["metrics"]["session"].get("recall_all@5", 0.0)
        r5_all.append(r5)

        if (i + 1) % 10 == 0 or i == len(data) - 1:
            elapsed = time.time() - start
            eta     = (elapsed / (i + 1)) * (len(data) - i - 1)
            print(f"  [{i+1:>3}/{len(data)}]  recall_all@5: {np.mean(r5_all):.3f}"
                  f"  |  elapsed {elapsed:.0f}s  eta {eta:.0f}s")

    out_f.close()

    print()
    print("=" * 70)
    print(f"Done. {len(data)} questions in {time.time()-start:.1f}s")
    print()
    print("Official metrics:")
    print(f"  python /tmp/LongMemEval/src/evaluation/print_retrieval_metrics.py {args.out}")


if __name__ == "__main__":
    main()
