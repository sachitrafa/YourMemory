"""
Memory compaction — auto-compress clusters of related memories into one structured
summary and archive the originals.

Why: a memory store that only ever grows bloats and its signal-to-noise degrades. After
several memories accumulate about the same topic/entity, we summarize them into a single
consolidated memory and move the originals to `memory_archive`. The live `memories` table
stays lean (recall stays fast and clean), the summary preserves every distinct detail,
and the originals remain recoverable and auditable.

Flow per user:
  1. Embed all live memories, greedily cluster by cosine similarity.
  2. For each cluster of >= MIN_CLUSTER members, LLM-summarize into one memory.
  3. Insert the summary, copy originals to memory_archive, delete originals from memories.
  4. Re-index the graph (drop original nodes, index the summary). Audit the compaction.

Conservative by design: the summary prompt is instructed to preserve all facts, the
threshold groups clearly-related memories, and originals are archived (never lost).
"""

import json
import math
import os
import urllib.request
from datetime import datetime, timezone

from src.services.embed import embed
from src.services.extract import categorize
from src.db.connection import get_backend, get_conn, emb_to_db, duckdb_rows
from src.services.audit import log_event

MIN_CLUSTER     = int(os.getenv("YOURMEMORY_COMPACT_MIN", "5"))     # min memories to compress
SIM_THRESHOLD   = float(os.getenv("YOURMEMORY_COMPACT_SIM", "0.62")) # cosine to group as "related"
MAX_SCAN        = int(os.getenv("YOURMEMORY_COMPACT_MAX", "2000"))   # cap per run (O(n^2) guard)


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _summarize(contents: list[str]) -> str | None:
    """LLM-compress related facts into ONE structured memory, preserving every detail."""
    joined = "\n".join(f"- {c}" for c in contents)
    prompt = (
        "You are compressing several related memory facts about the same topic into ONE "
        "consolidated memory. Preserve EVERY distinct detail — names, numbers, dates, "
        "preferences, decisions. Do not drop or invent information. Merge overlaps, keep "
        "specifics. Output a single self-contained declarative summary (1–3 sentences), "
        "no preamble, no markdown.\n\n"
        f"Facts:\n{joined}\n\nConsolidated memory:"
    )
    backend = os.getenv("YOURMEMORY_EXTRACT_BACKEND", "ollama").lower()
    try:
        if backend == "anthropic":
            from src.services.extract import _anthropic_complete  # type: ignore
            return _anthropic_complete(prompt).strip() or None  # best-effort if present
    except Exception:
        pass
    # Default: Ollama
    url   = os.getenv("YOURMEMORY_OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("YOURMEMORY_OLLAMA_MODEL", "qwen2.5:7b")
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "keep_alive": os.getenv("YOURMEMORY_OLLAMA_KEEPALIVE", "30m"),
        "options": {"temperature": 0, "num_predict": 220},
    }).encode()
    try:
        req = urllib.request.Request(f"{url}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read()).get("response", "").strip()
        return out or None
    except Exception:
        return None


def _live_memories(conn, backend: str, user_id: str) -> list[dict]:
    cols = "id, content, category, importance, agent_id, visibility, created_at"
    if backend == "postgres":
        from psycopg2.extras import RealDictCursor
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"SELECT {cols} FROM memories WHERE user_id = %s ORDER BY id LIMIT %s",
                    (user_id, MAX_SCAN))
        rows = [dict(r) for r in cur.fetchall()]; cur.close(); return rows
    if backend == "duckdb":
        return duckdb_rows(conn.execute(
            f"SELECT {cols} FROM memories WHERE user_id = ? ORDER BY id LIMIT ?", [user_id, MAX_SCAN]))
    cur = conn.cursor()
    cur.execute(f"SELECT {cols} FROM memories WHERE user_id = ? ORDER BY id LIMIT ?", (user_id, MAX_SCAN))
    cn = [d[0] for d in cur.description]
    rows = [dict(zip(cn, r)) for r in cur.fetchall()]; cur.close(); return rows


def compact_user(user_id: str, min_cluster: int = None, sim_threshold: float = None) -> dict:
    """Compress related-memory clusters for one user. Returns stats."""
    user_id = (user_id or "").strip().lower()
    min_cluster = min_cluster or MIN_CLUSTER
    sim_threshold = sim_threshold if sim_threshold is not None else SIM_THRESHOLD
    backend = get_backend()

    conn = get_conn()
    try:
        mems = _live_memories(conn, backend, user_id)
    finally:
        conn.close()
    if len(mems) < min_cluster:
        return {"clusters": 0, "archived": 0, "summaries": 0, "scanned": len(mems)}

    # Embed + greedy cluster by cosine similarity.
    vecs = [embed(m["content"]) for m in mems]
    used, clusters = set(), []
    for i in range(len(mems)):
        if i in used:
            continue
        group = [i]; used.add(i)
        for j in range(i + 1, len(mems)):
            if j in used:
                continue
            if _cosine(vecs[i], vecs[j]) >= sim_threshold:
                group.append(j); used.add(j)
        if len(group) >= min_cluster:
            clusters.append(group)

    if not clusters:
        return {"clusters": 0, "archived": 0, "summaries": 0, "scanned": len(mems)}

    archived_total, summaries = 0, 0
    for group in clusters:
        members = [mems[k] for k in group]
        summary = _summarize([m["content"] for m in members])
        if not summary or len(summary) < 12:
            continue   # summarization failed → leave the cluster untouched
        # Inherit the strongest signal from the cluster.
        importance = max(float(m["importance"] or 0.5) for m in members)
        category   = categorize(summary)
        agent_id   = members[0].get("agent_id")
        visibility = members[0].get("visibility") or "shared"
        summary_id = _apply_compaction(backend, user_id, summary, importance, category,
                                       agent_id, visibility, members)
        if summary_id is not None:
            archived_total += len(members)
            summaries += 1

    if summaries:
        log_event("write", "compact", user_id,
                  detail={"clusters": summaries, "archived": archived_total, "scanned": len(mems)})
    return {"clusters": len(clusters), "archived": archived_total,
            "summaries": summaries, "scanned": len(mems)}


def _apply_compaction(backend, user_id, summary, importance, category, agent_id,
                      visibility, members) -> int | None:
    """Insert the summary memory, archive the originals, delete them. Returns summary id."""
    emb_str = emb_to_db(embed(summary), backend)
    ids = [m["id"] for m in members]
    conn = get_conn()
    cur = conn.cursor() if backend != "duckdb" else None
    summary_id = None
    try:
        # 1. Insert the consolidated summary.
        if backend == "postgres":
            cur.execute(
                "INSERT INTO memories (user_id, content, embedding, importance, category, agent_id, visibility) "
                "VALUES (%s,%s,%s::vector,%s,%s,%s,%s) "
                "ON CONFLICT (user_id, content) DO UPDATE SET importance = EXCLUDED.importance RETURNING id",
                (user_id, summary, emb_str, importance, category, agent_id, visibility))
            summary_id = cur.fetchone()[0]
        elif backend == "duckdb":
            conn.execute(
                "INSERT INTO memories (user_id, content, embedding, importance, category, agent_id, visibility) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT (user_id, content) DO UPDATE SET importance = excluded.importance",
                [user_id, summary, emb_str, importance, category, agent_id, visibility])
            summary_id = conn.execute("SELECT id FROM memories WHERE user_id=? AND content=?",
                                      [user_id, summary]).fetchone()[0]
        else:
            cur.execute(
                "INSERT INTO memories (user_id, content, embedding, importance, category, agent_id, visibility) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT (user_id, content) DO UPDATE SET importance = excluded.importance",
                (user_id, summary, emb_str, importance, category, agent_id, visibility))
            cur.execute("SELECT id FROM memories WHERE user_id=? AND content=?", (user_id, summary))
            summary_id = cur.fetchone()[0]

        # 2. Archive originals + 3. delete them (skip the summary row if it collided with one).
        for m in members:
            if m["id"] == summary_id:
                continue
            vals = (m["id"], user_id, m["content"], m.get("category"), m.get("importance"),
                    m.get("agent_id"), m.get("visibility"), m.get("created_at"), summary_id)
            if backend == "postgres":
                cur.execute(
                    "INSERT INTO memory_archive (orig_id,user_id,content,category,importance,agent_id,visibility,created_at,summary_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", vals)
                cur.execute("DELETE FROM memories WHERE id = %s", (m["id"],))
            elif backend == "duckdb":
                conn.execute(
                    "INSERT INTO memory_archive (orig_id,user_id,content,category,importance,agent_id,visibility,created_at,summary_id) "
                    "VALUES (?,?,?,?,?,?,?,?,?)", list(vals))
                conn.execute("DELETE FROM memories WHERE id = ?", [m["id"]])
            else:
                cur.execute(
                    "INSERT INTO memory_archive (orig_id,user_id,content,category,importance,agent_id,visibility,created_at,summary_id) "
                    "VALUES (?,?,?,?,?,?,?,?,?)", vals)
                cur.execute("DELETE FROM memories WHERE id = ?", (m["id"],))
        if backend != "duckdb":
            conn.commit()
    except Exception:
        if backend == "postgres":
            try: conn.rollback()
            except Exception: pass
        summary_id = None
    finally:
        if cur: cur.close()
        conn.close()

    # Best-effort graph upkeep: drop original nodes, index the summary.
    if summary_id is not None:
        try:
            from src.graph import get_graph_backend
            gb = get_graph_backend()
            for m in members:
                if m["id"] != summary_id:
                    try: gb.delete_node(m["id"])
                    except Exception: pass
        except Exception:
            pass
        try:
            from src.graph.graph_store import index_memory
            index_memory(memory_id=summary_id, user_id=user_id, content=summary,
                         strength=importance, importance=importance, category=category,
                         embedding=list(embed(summary)))
        except Exception:
            pass
    return summary_id
