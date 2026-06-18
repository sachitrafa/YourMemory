import json
import math
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone

from .embed import embed
from .decay import compute_strength
from .utils import parse_dt
from .temporal import detect_temporal_range, score_boost as temporal_score_boost
from src.db.connection import get_backend, get_conn
from src.graph.graph_store import expand_with_graph, propagate_recall

SIMILARITY_THRESHOLD          = 0.50
SIMILARITY_THRESHOLD_FALLBACK = 0.20
REINFORCE_THRESHOLD           = 0.75

# Hybrid scoring weights: BM25 keyword + vector×decay
W_BM25   = 0.5
W_VECTOR = 0.5


def _user_has_memories(user_id: str) -> bool:
    """Return False when the user has no stored memories — prevents cold-start cross-user leakage."""
    backend = get_backend()
    conn = get_conn()
    try:
        if backend == "postgres":
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM memories WHERE user_id = %s LIMIT 1", (user_id,))
            found = cur.fetchone() is not None
            cur.close()
        elif backend == "duckdb":
            result = conn.execute("SELECT 1 FROM memories WHERE user_id = ? LIMIT 1", [user_id])
            found = result.fetchone() is not None
        else:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM memories WHERE user_id = ? LIMIT 1", (user_id,))
            found = cur.fetchone() is not None
            cur.close()
        return found
    except Exception:
        return True  # on error, let retrieval proceed normally
    finally:
        conn.close()


def _scope_fragment(scope: list[str] | None, param_style: str = "qmark") -> tuple[str, list]:
    """
    Build a SQL WHERE fragment that hard-filters candidates by context_paths scope.
    Memories with context_paths IS NULL or containing 'user:universal' always pass.
    param_style: 'qmark' for SQLite/DuckDB (?), 'format' for Postgres (%s).
    """
    if not scope:
        return "", []
    ph   = "?" if param_style == "qmark" else "%s"
    func = "instr" if param_style == "qmark" else "strpos"
    parts  = ["context_paths IS NULL", f"{func}(context_paths, {ph}) > 0"]
    params = ['"user:universal"']
    for s in scope:
        parts.append(f"{func}(context_paths, {ph}) > 0")
        params.append(f'"{s}"')
    return "AND (" + " OR ".join(parts) + ")", params


# ── BM25 / FTS helpers ───────────────────────────────────────────────────────

def _normalize_bm25_sqlite(raw: float) -> float:
    """
    SQLite FTS5 bm25() returns negative values — more negative = better match.
    Convert to [0, 1] via sigmoid of the negated score.
    """
    return 1.0 / (1.0 + math.exp(raw))   # raw is negative, so exp(raw) < 1


def _fts_search_sqlite(conn, user_id: str, agent_id: str | None,
                       query: str, limit: int) -> dict[int, float]:
    """Return {memory_id: bm25_norm} for FTS5 matches."""
    try:
        cur = conn.cursor()
        if agent_id:
            cur.execute("""
                SELECT m.id, bm25(memories_fts) AS bm25_score
                FROM memories_fts
                JOIN memories m ON memories_fts.rowid = m.id
                WHERE memories_fts MATCH ?
                  AND m.user_id = ?
                  AND (m.visibility = 'shared'
                       OR (m.visibility = 'private' AND m.agent_id = ?))
                ORDER BY bm25_score
                LIMIT ?
            """, (query, user_id, agent_id, limit))
        else:
            cur.execute("""
                SELECT m.id, bm25(memories_fts) AS bm25_score
                FROM memories_fts
                JOIN memories m ON memories_fts.rowid = m.id
                WHERE memories_fts MATCH ?
                  AND m.user_id = ?
                  AND m.visibility = 'shared'
                ORDER BY bm25_score
                LIMIT ?
            """, (query, user_id, limit))
        rows = cur.fetchall()
        cur.close()
        return {r[0]: _normalize_bm25_sqlite(r[1]) for r in rows}
    except Exception as exc:
        print(f"[retrieve] FTS search failed (sqlite): {exc}", file=sys.stderr)
        return {}


def _fts_search_duckdb(conn, user_id: str, agent_id: str | None,
                       query: str, limit: int) -> dict[int, float]:
    """Return {memory_id: bm25_norm} using DuckDB FTS extension."""
    try:
        conn.execute("LOAD fts;")
        conn.execute(
            "PRAGMA create_fts_index('memories', 'id', 'content', overwrite=1);"
        )
        if agent_id:
            result = conn.execute("""
                SELECT id, fts_main_memories.match_bm25(id, ?) AS score
                FROM memories
                WHERE score IS NOT NULL
                  AND user_id = ?
                  AND (visibility = 'shared'
                       OR (visibility = 'private' AND agent_id = ?))
                ORDER BY score DESC
                LIMIT ?
            """, [query, user_id, agent_id, limit])
        else:
            result = conn.execute("""
                SELECT id, fts_main_memories.match_bm25(id, ?) AS score
                FROM memories
                WHERE score IS NOT NULL
                  AND user_id = ?
                  AND visibility = 'shared'
                ORDER BY score DESC
                LIMIT ?
            """, [query, user_id, limit])
        from src.db.connection import duckdb_rows
        rows = duckdb_rows(result)
        # DuckDB FTS returns positive scores; normalize with score/(score+1)
        return {r["id"]: r["score"] / (r["score"] + 1.0) for r in rows if r["score"]}
    except Exception as exc:
        print(f"[retrieve] FTS search failed (duckdb): {exc}", file=sys.stderr)
        return {}


def _fts_search_postgres(conn, user_id: str, agent_id: str | None,
                         query: str, limit: int) -> dict[int, float]:
    """Return {memory_id: bm25_norm} using Postgres tsvector + ts_rank."""
    try:
        from psycopg2.extras import RealDictCursor
        # OR the sanitized content terms — any keyword match contributes (FTS is one
        # leg of the hybrid score), and the 'english' config strips stopwords. The old
        # code AND-joined the raw query incl. stopwords, so it never matched anything.
        terms = [t for t in re.findall(r'[a-z0-9]+', query.lower()) if len(t) > 2]
        if not terms:
            return {}
        tsq = " | ".join(terms)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if agent_id:
            cur.execute("""
                SELECT id, ts_rank_cd(content_tsv, to_tsquery('english', %s)) AS score
                FROM memories
                WHERE content_tsv @@ to_tsquery('english', %s)
                  AND user_id = %s
                  AND (visibility = 'shared'
                       OR (visibility = 'private' AND agent_id = %s))
                ORDER BY score DESC
                LIMIT %s
            """, (tsq, tsq, user_id, agent_id, limit))
        else:
            cur.execute("""
                SELECT id, ts_rank_cd(content_tsv, to_tsquery('english', %s)) AS score
                FROM memories
                WHERE content_tsv @@ to_tsquery('english', %s)
                  AND user_id = %s
                  AND visibility = 'shared'
                ORDER BY score DESC
                LIMIT %s
            """, (tsq, tsq, user_id, limit))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        if not rows:
            return {}
        # Max-normalize ts_rank_cd (small absolute values) into a 0-1 signal so the
        # keyword leg is comparable in magnitude to cosine in the hybrid score.
        mx = max((r["score"] for r in rows if r["score"]), default=0.0) or 1.0
        return {r["id"]: float(r["score"]) / mx for r in rows if r["score"]}
    except Exception as exc:
        print(f"[retrieve] FTS search failed (postgres): {exc}", file=sys.stderr)
        return {}


SPATIAL_BOOST = 0.08   # score bonus when memory context_paths overlap current path


def _apply_scope_filter(candidates: list, scope: list[str] | None) -> list:
    """Hard-filter candidates to those matching requested scope.
    Beliefs with no context_paths or with user:universal are always included.
    """
    if not scope:
        return candidates
    result = []
    for m in candidates:
        raw = m.get("context_paths")
        if not raw:
            result.append(m)
            continue
        try:
            paths = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except Exception:
            result.append(m)
            continue
        if "user:universal" in paths or any(s in paths for s in scope):
            result.append(m)
    return result


def retrieve(user_id: str, query: str, top_k: int = 5, agent_id: str = None,
             current_path: str | None = None, no_graph: bool = False,
             score_threshold: float | None = None,
             scope: list[str] | None = None, expand_k: int = 0) -> dict:
    """
    Round 1 — vector search (cosine similarity):
       - DuckDB:   native array_cosine_similarity
       - Postgres: pgvector operator
       - SQLite:   Python numpy cosine

    Round 2 — graph expansion (multi-hop BFS from Round 1 seeds).
    Recall propagation: high-similarity memories boost graph neighbours.
    Temporal boost: memories within a resolved date range score higher.

    no_graph=True: skip graph expansion — used for BEAM ablation benchmark.
    """
    if not _user_has_memories(user_id):
        return {"memoriesFound": 0, "context": "", "memories": []}

    query_embedding = embed(query)
    backend = get_backend()
    temporal_range = detect_temporal_range(query)

    if backend == "postgres":
        result = _retrieve_postgres(user_id, query, query_embedding, top_k, agent_id, temporal_range, scope=scope)
    elif backend == "duckdb":
        result = _retrieve_duckdb(user_id, query, query_embedding, top_k, agent_id, temporal_range, scope=scope)
    else:
        result = _retrieve_sqlite(user_id, query, query_embedding, top_k, agent_id, temporal_range, scope=scope)

    if current_path and result.get("memories"):
        result = _apply_spatial_boost(result, current_path, top_k)

    if score_threshold is not None:
        result["memories"] = [m for m in result.get("memories", []) if m.get("score", 0) >= score_threshold]

    seed_ids = [m["id"] for m in result.get("memories", [])]
    if seed_ids and not no_graph:
        # expand_k > 0 makes graph expansion ADDITIVE: keep the top_k direct hits and
        # enrich with up to expand_k connected neighbours (the "connected region" that
        # lets the model answer without re-reading). expand_k=0 = legacy behaviour.
        cap = top_k + max(0, expand_k)
        graph_hits = expand_with_graph(seed_ids, user_id, top_k=cap)
        new_hits = [(nid, ew) for nid, ew in graph_hits if nid not in set(seed_ids)]
        if new_hits:
            extra = _fetch_by_ids(new_hits, user_id, backend)
            result = _merge_graph_results(result, extra, cap)

        reinforced = [m for m in result.get("memories", [])
                      if m.get("similarity", 0) >= REINFORCE_THRESHOLD]
        for m in reinforced:
            boosted_ids = propagate_recall(m["id"], user_id)
            if boosted_ids:
                _bump_recall_count(boosted_ids, backend)

    # Optional LLM relevance gate — see _relevance_filter. Off unless
    # YOURMEMORY_RELEVANCE_JUDGE=1. Runs at most one batched call, and only when
    # there are candidates, so an empty/off-topic store costs zero LLM calls.
    result = _relevance_filter(query, result)

    return result


def _relevance_filter(query: str, result: dict) -> dict:
    """Ask the local model which candidate memories are actually relevant to the
    query — a SINGLE batched call — and drop the rest.

    Purpose: prevent "worse than nothing" injection. Vector recall always returns
    the nearest memories even when none are on-topic (e.g. session 1 of a new
    deal surfaces unrelated identity/preference memories). This gate lets recall
    stay SILENT when nothing is genuinely relevant.

    Cost controls (matters: recall runs on every prompt via the hook):
      - No-op unless YOURMEMORY_RELEVANCE_JUDGE=1.
      - Skips the LLM entirely when there are no candidates.
      - One call total, not one per memory.
      - Strict timeout; fails OPEN (returns candidates unchanged) so a slow or
        down Ollama degrades to threshold-only behaviour instead of breaking recall.
      - Judge model is configurable (YOURMEMORY_JUDGE_MODEL) so the hot path can
        use a faster/smaller model than the extractor if latency demands it.
    """
    if os.getenv("YOURMEMORY_RELEVANCE_JUDGE", "0") != "1":
        return result

    memories = result.get("memories", [])
    if not memories:
        return result

    listing = "\n".join(f"{i + 1}. {m['content']}" for i, m in enumerate(memories))
    prompt = (
        "You filter a memory store before injecting context into an assistant.\n"
        "Given a QUERY and a numbered list of candidate MEMORIES, return the numbers "
        "of ONLY the memories that are directly relevant as context for this query. "
        "Be strict — exclude memories about a different topic, person, or project. "
        "If none are relevant, reply NONE.\n"
        "Reply with comma-separated numbers (or NONE) and nothing else.\n\n"
        f"QUERY: {query}\n\nMEMORIES:\n{listing}\n\nRelevant numbers:"
    )

    url   = os.getenv("YOURMEMORY_OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("YOURMEMORY_JUDGE_MODEL",
                      os.getenv("YOURMEMORY_OLLAMA_MODEL", "qwen2.5:7b"))
    payload = json.dumps({
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "keep_alive": os.getenv("YOURMEMORY_OLLAMA_KEEPALIVE", "30m"),
        "options": {"temperature": 0, "num_predict": 24},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{url}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        timeout = float(os.getenv("YOURMEMORY_JUDGE_TIMEOUT", "5"))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            answer = json.loads(resp.read()).get("response", "")
    except Exception:
        return result  # fail open — degrade to threshold-only behaviour

    ans = answer.strip().upper()
    keep_idx = {int(n) for n in re.findall(r"\d+", ans)}
    if not keep_idx:
        if "NONE" in ans:
            kept = []                 # judge explicitly found nothing relevant
        else:
            return result             # unparseable answer → fail open
    else:
        kept = [m for i, m in enumerate(memories) if (i + 1) in keep_idx]

    result["memories"]      = kept
    result["memoriesFound"] = len(kept)
    result["context"]       = _build_context(kept)
    return result


# ── Shared helpers ────────────────────────────────────────────────────────────

def _apply_spatial_boost(result: dict, current_path: str, top_k: int) -> dict:
    """Boost memories whose context_paths overlap with current_path, then re-rank."""
    import json as _json
    boosted = []
    for m in result["memories"]:
        extra = 0.0
        raw = m.get("context_paths")
        if raw:
            try:
                paths = _json.loads(raw) if isinstance(raw, str) else raw
                if any(current_path.startswith(p) or p.startswith(current_path) for p in paths):
                    extra = SPATIAL_BOOST
            except Exception:
                pass
        boosted.append({**m, "score": round(m["score"] + extra, 4)})
    boosted.sort(key=lambda x: x["score"], reverse=True)
    return _format_result(boosted[:top_k])


def _score_candidates(candidates: list, fts_scores: dict[int, float] | None = None,
                      temporal_range=None) -> list:
    """
    Add strength + hybrid score to each candidate.

    Hybrid formula:
        score = W_BM25 × bm25_norm + W_VECTOR × cosine [+ temporal_boost]

    Decay (strength) is computed for display and pruning purposes only — it is
    intentionally excluded from the ranking formula. Multiplying cosine by strength
    causes old-but-still-valid memories to rank below newer but irrelevant ones,
    which is wrong: decay handles staleness via the 24h pruning job, not ranking.
    Knowledge-update conflicts are resolved at store time via contradiction detection.

    temporal_range: optional (start, end) UTC datetime tuple from detect_temporal_range().
    When present, memories whose created_at falls within the range receive a boost.
    Has zero effect on non-temporal queries (temporal_range is None).
    """
    fts = fts_scores or {}
    scored = []
    for m in candidates:
        strength = compute_strength(
            last_accessed_at=parse_dt(m["last_accessed_at"]),
            recall_count=m["recall_count"],
            importance=m["importance"],
            category=m["category"],
        )
        bm25_norm    = fts.get(m["id"], 0.0)
        hybrid_score = W_BM25 * bm25_norm + W_VECTOR * m["similarity"]
        hybrid_score += temporal_score_boost(m.get("created_at"), temporal_range)
        scored.append({
            **m,
            "strength":   strength,
            "bm25":       round(bm25_norm, 4),
            "score":      hybrid_score,
            "context_paths": m.get("context_paths"),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _build_context(top: list) -> str:
    """Build the categorized context string from a ranked memory list."""
    categories = ["fact", "assumption", "strategy", "failure"]
    labels     = {"fact": "[Facts]", "assumption": "[Assumptions]",
                  "strategy": "[Strategies]", "failure": "[Failures]"}
    parts = []
    for cat in categories:
        group = [m for m in top if m["category"] == cat]
        if group:
            parts.append(labels[cat] + "\n" + "\n".join(m["content"] for m in group))
    return "\n\n".join(parts)


def _format_result(top: list) -> dict:
    """Turn a ranked memory list into the standard API response dict."""
    return {
        "memoriesFound": len(top),
        "context": _build_context(top),
        "memories": [
            {
                "id":            m["id"],
                "content":       m["content"],
                "category":      m["category"],
                "agent_id":      m.get("agent_id"),
                "visibility":    m.get("visibility"),
                "importance":    round(m["importance"], 4),
                "strength":      round(m["strength"], 4),
                "similarity":    round(m["similarity"], 4),
                "bm25":          round(m.get("bm25", 0.0), 4),
                "score":         round(m["score"], 4),
                "context_paths": m.get("context_paths"),
                "via_graph":     bool(m.get("via_graph", False)),
            }
            for m in top
        ],
    }


def _reinforce(top: list, conn, backend: str) -> None:
    """Bump recall_count + last_accessed_at for high-similarity memories."""
    relevant_ids = [m["id"] for m in top if m["similarity"] >= REINFORCE_THRESHOLD]
    if not relevant_ids:
        return
    if backend == "postgres":
        cur = conn.cursor()
        cur.execute(
            "UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW()"
            " WHERE id = ANY(%s)",
            (relevant_ids,),
        )
        cur.close()
    elif backend == "duckdb":
        for mid in relevant_ids:
            conn.execute(
                "UPDATE memories SET recall_count = recall_count + 1,"
                " last_accessed_at = CURRENT_TIMESTAMP WHERE id = ?",
                [mid],
            )
    else:
        cur = conn.cursor()
        for mid in relevant_ids:
            cur.execute(
                "UPDATE memories SET recall_count = recall_count + 1,"
                " last_accessed_at = datetime('now') WHERE id = ?",
                (mid,),
            )
        cur.close()


def _finish(candidates: list, top_k: int, conn, backend: str,
            fts_scores: dict[int, float] | None = None,
            temporal_range=None) -> dict:
    """Score → rank → reinforce → format → close connection."""
    scored = _score_candidates(candidates, fts_scores, temporal_range=temporal_range)
    top = scored[:top_k]
    if not top:
        conn.close()
        return {"memoriesFound": 0, "context": "", "memories": []}
    _reinforce(top, conn, backend)
    if backend != "duckdb":
        conn.commit()
    conn.close()
    return _format_result(top)


# ── Graph expansion helpers ───────────────────────────────────────────────────

def _fetch_by_ids(hits: list[tuple[int, float]], user_id: str, backend: str) -> list:
    """
    Fetch memory rows for graph-expanded (id, edge_weight) pairs.

    edge_weight (cosine × verb_weight) is used as the similarity proxy —
    it encodes how strongly each node is connected to its seed, which was
    itself relevant to the query.  This replaces the previous flat 0.5.
    """
    if not hits:
        return []
    ids       = [nid for nid, _ in hits]
    ew_by_id  = {nid: ew for nid, ew in hits}

    conn = get_conn()
    rows = []
    try:
        if backend == "postgres":
            from psycopg2.extras import RealDictCursor
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT id, content, category, importance, recall_count, last_accessed_at, context_paths,"
                "       agent_id, visibility FROM memories WHERE id = ANY(%s) AND user_id = %s",
                (ids, user_id),
            )
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
        elif backend == "duckdb":
            from src.db.connection import duckdb_rows
            placeholders = ", ".join("?" * len(ids))
            result = conn.execute(
                f"SELECT id, content, category, importance, recall_count, last_accessed_at, context_paths,"
                f"       agent_id, visibility FROM memories WHERE id IN ({placeholders}) AND user_id = ?",
                ids + [user_id],
            )
            rows = duckdb_rows(result)
        else:
            cur = conn.cursor()
            placeholders = ", ".join("?" * len(ids))
            cur.execute(
                f"SELECT id, content, category, importance, recall_count, last_accessed_at, context_paths,"
                f"       agent_id, visibility FROM memories WHERE id IN ({placeholders}) AND user_id = ?",
                ids + [user_id],
            )
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
    except Exception as exc:
        print(f"[retrieve] _fetch_by_ids failed: {exc}", file=sys.stderr)
    finally:
        conn.close()

    scored = []
    for m in rows:
        strength   = compute_strength(
            last_accessed_at=parse_dt(m["last_accessed_at"]),
            recall_count=m["recall_count"],
            importance=m["importance"],
            category=m["category"],
        )
        edge_weight = ew_by_id.get(m["id"], 0.4)
        # Cap similarity below REINFORCE_THRESHOLD so graph-expanded nodes
        # never accidentally trigger propagate_recall — their edge_weight
        # measures connection to a seed, not direct similarity to the query.
        # Do NOT multiply by strength: Round 1 scoring (W_BM25×bm25 + W_VECTOR×cosine)
        # also excludes strength, so graph nodes must be scored consistently
        # or they can never compete for top-k slots.
        sim = min(edge_weight, REINFORCE_THRESHOLD - 0.01)
        scored.append({
            **m,
            "similarity": sim,
            "strength":   strength,
            "score":      W_VECTOR * sim,
            "via_graph":  True,
        })
    return scored


def _merge_graph_results(result: dict, extra: list, top_k: int) -> dict:
    """Merge graph-expanded memories into vector results, re-rank, trim to top_k."""
    if not extra:
        return result
    existing = result.get("memories", [])
    existing_ids = {m["id"] for m in existing}
    new_entries = [m for m in extra if m["id"] not in existing_ids]
    if not new_entries:
        return result
    merged = existing + new_entries
    merged.sort(key=lambda x: x["score"], reverse=True)
    return _format_result(merged[:top_k])


def _bump_recall_count(ids: list, backend: str) -> None:
    """Increment recall_count for a list of memory IDs (graph propagation)."""
    if not ids:
        return
    conn = get_conn()
    try:
        if backend == "postgres":
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET recall_count = recall_count + 1 WHERE id = ANY(%s)",
                (ids,),
            )
            conn.commit()
            cur.close()
        elif backend == "duckdb":
            placeholders = ", ".join("?" * len(ids))
            conn.execute(
                f"UPDATE memories SET recall_count = recall_count + 1 WHERE id IN ({placeholders})",
                ids,
            )
        else:
            cur = conn.cursor()
            for mid in ids:
                cur.execute(
                    "UPDATE memories SET recall_count = recall_count + 1 WHERE id = ?", (mid,)
                )
            conn.commit()
            cur.close()
    except Exception as exc:
        print(f"[retrieve] _bump_recall_count failed: {exc}", file=sys.stderr)
    finally:
        conn.close()


# ── Postgres path ─────────────────────────────────────────────────────────────

def _retrieve_postgres(user_id, query, query_embedding, top_k, agent_id, temporal_range=None, scope=None):
    from psycopg2.extras import RealDictCursor
    embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
    scope_sql, scope_params = _scope_fragment(scope, param_style="format")
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    if agent_id:
        cur.execute(f"""
            SELECT id, content, category, importance, recall_count, created_at, last_accessed_at,
                   context_paths, agent_id, visibility,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM memories
            WHERE user_id = %s
              AND (visibility = 'shared' OR (visibility = 'private' AND agent_id = %s))
              AND 1 - (embedding <=> %s::vector) >= %s
              {scope_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding_str, user_id, agent_id, embedding_str,
              SIMILARITY_THRESHOLD, *scope_params, embedding_str, top_k * 2))
    else:
        cur.execute(f"""
            SELECT id, content, category, importance, recall_count, created_at, last_accessed_at,
                   context_paths, agent_id, visibility,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM memories
            WHERE user_id = %s
              AND visibility = 'shared'
              AND 1 - (embedding <=> %s::vector) >= %s
              {scope_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding_str, user_id, embedding_str,
              SIMILARITY_THRESHOLD, *scope_params, embedding_str, top_k * 2))
    candidates = [dict(row) for row in cur.fetchall()]
    cur.close()
    fts_scores = _fts_search_postgres(conn, user_id, agent_id, query=query, limit=top_k * 2)
    candidates = _apply_scope_filter(candidates, scope)
    return _finish(candidates, top_k, conn, backend="postgres", fts_scores=fts_scores,
                   temporal_range=temporal_range)


# ── DuckDB path ───────────────────────────────────────────────────────────────

def _retrieve_duckdb(user_id, query, query_embedding, top_k, agent_id, temporal_range=None, scope=None):
    from src.db.connection import duckdb_rows
    scope_sql, scope_params = _scope_fragment(scope, param_style="qmark")
    conn = get_conn()

    def _query(threshold):
        if agent_id:
            return duckdb_rows(conn.execute(f"""
                SELECT id, content, category, importance, recall_count, created_at, last_accessed_at,
                       context_paths, agent_id, visibility,
                       array_cosine_similarity(embedding, ?::FLOAT[768]) AS similarity
                FROM memories
                WHERE user_id = ?
                  AND (visibility = 'shared' OR (visibility = 'private' AND agent_id = ?))
                  AND array_cosine_similarity(embedding, ?::FLOAT[768]) >= ?
                  {scope_sql}
                ORDER BY similarity DESC LIMIT ?
            """, [query_embedding, user_id, agent_id, query_embedding, threshold] + scope_params + [top_k * 2]))
        return duckdb_rows(conn.execute(f"""
            SELECT id, content, category, importance, recall_count, created_at, last_accessed_at,
                   context_paths, agent_id, visibility,
                   array_cosine_similarity(embedding, ?::FLOAT[768]) AS similarity
            FROM memories
            WHERE user_id = ?
              AND visibility = 'shared'
              AND array_cosine_similarity(embedding, ?::FLOAT[768]) >= ?
              {scope_sql}
            ORDER BY similarity DESC LIMIT ?
        """, [query_embedding, user_id, query_embedding, threshold] + scope_params + [top_k * 2]))

    candidates = _query(SIMILARITY_THRESHOLD)
    if not candidates:
        candidates = _query(SIMILARITY_THRESHOLD_FALLBACK)

    fts_scores = _fts_search_duckdb(conn, user_id, agent_id, query=query, limit=top_k * 2)
    candidates = _apply_scope_filter(candidates, scope)
    return _finish(candidates, top_k, conn, backend="duckdb", fts_scores=fts_scores,
                   temporal_range=temporal_range)


# ── SQLite path ───────────────────────────────────────────────────────────────

def _retrieve_sqlite(user_id, query, query_embedding, top_k, agent_id, temporal_range=None, scope=None):
    from .utils import cosine
    scope_sql, scope_params = _scope_fragment(scope, param_style="qmark")
    conn = get_conn()
    cur  = conn.cursor()
    if agent_id:
        cur.execute(f"""
            SELECT id, content, category, importance, recall_count, created_at, last_accessed_at,
                   context_paths, agent_id, visibility, embedding
            FROM memories
            WHERE user_id = ? AND (visibility = 'shared' OR (visibility = 'private' AND agent_id = ?))
            {scope_sql}
        """, (user_id, agent_id) + tuple(scope_params))
    else:
        cur.execute(f"""
            SELECT id, content, category, importance, recall_count, created_at, last_accessed_at,
                   context_paths, agent_id, visibility, embedding
            FROM memories WHERE user_id = ? AND visibility = 'shared'
            {scope_sql}
        """, (user_id,) + tuple(scope_params))
    rows = cur.fetchall()
    cur.close()

    def _filter(threshold):
        result = []
        for row in rows:
            raw_emb = row["embedding"]
            if raw_emb is None:
                continue
            sim = cosine(query_embedding, json.loads(raw_emb))
            if sim >= threshold:
                d = dict(row)
                d["similarity"] = sim
                result.append(d)
        return result

    candidates = _filter(SIMILARITY_THRESHOLD)
    if not candidates:
        candidates = _filter(SIMILARITY_THRESHOLD_FALLBACK)

    fts_scores = _fts_search_sqlite(conn, user_id, agent_id, query=query, limit=top_k * 2)
    candidates = _apply_scope_filter(candidates, scope)
    return _finish(candidates, top_k, conn, backend="sqlite", fts_scores=fts_scores,
                   temporal_range=temporal_range)
