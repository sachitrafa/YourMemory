"""
DSAR / data-portability endpoints — export, right-to-forget purge, and bulk import.

  GET    /users/{user_id}/export    → full JSON of a user's memories (DSAR export / backup)
  DELETE /users/{user_id}/memories  → purge ALL of a user's data (right-to-forget)
  POST   /users/{user_id}/import    → bulk restore/seed memories from an export

Every operation is recorded in the audit trail. In the default local deployment the
path `user_id` identifies the data subject (trust boundary is the OS user); a hosted
deployment must additionally authenticate the caller before these are exposed.

See docs/policies/06-data-retention-deletion-policy.md.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from src.services.embed import embed
from src.services.extract import categorize
from src.db.connection import get_backend, get_conn, emb_to_db, duckdb_rows
from src.services.audit import log_event

router = APIRouter()

_EXPORT_COLS = ("id, content, category, importance, recall_count, "
                "last_accessed_at, created_at, agent_id, visibility, context_paths")
_EXPORT_KEYS = ["id", "content", "category", "importance", "recall_count",
                "last_accessed_at", "created_at", "agent_id", "visibility", "context_paths"]


def _rows_for_user(conn, backend: str, user_id: str) -> list[dict]:
    sql_pg    = f"SELECT {_EXPORT_COLS} FROM memories WHERE user_id = %s ORDER BY id"
    sql_other = f"SELECT {_EXPORT_COLS} FROM memories WHERE user_id = ? ORDER BY id"
    if backend == "postgres":
        from psycopg2.extras import RealDictCursor
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql_pg, (user_id,))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows
    if backend == "duckdb":
        return duckdb_rows(conn.execute(sql_other, [user_id]))
    cur = conn.cursor()
    cur.execute(sql_other, (user_id,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    return rows


# ── GET /users/{user_id}/export ────────────────────────────────────────────────

@router.get("/users/{user_id}/export")
def export_user(user_id: str):
    """Return all of a user's memories as JSON (DSAR access request / backup)."""
    user_id = user_id.strip().lower()
    backend = get_backend()
    conn = get_conn()
    try:
        rows = _rows_for_user(conn, backend, user_id)
    finally:
        conn.close()

    # Stringify timestamps for clean JSON.
    for r in rows:
        for k in ("last_accessed_at", "created_at"):
            if r.get(k) is not None:
                r[k] = str(r[k])

    from datetime import datetime, timezone
    log_event("read", "export", user_id, detail={"count": len(rows)})
    return {
        "user_id":     user_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "count":       len(rows),
        "memories":    rows,
    }


# ── DELETE /users/{user_id}/memories  (right-to-forget) ─────────────────────────

@router.delete("/users/{user_id}/memories")
def purge_user(user_id: str):
    """Delete ALL of a user's memories, graph nodes, and conversation buffer."""
    user_id = user_id.strip().lower()
    backend = get_backend()
    conn = get_conn()
    deleted = 0
    try:
        # Count first (for the audit detail + response).
        if backend == "postgres":
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memories WHERE user_id = %s", (user_id,))
            deleted = cur.fetchone()[0]
            cur.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM conversation_buffer WHERE user_id = %s", (user_id,))
            conn.commit()
            cur.close()
        elif backend == "duckdb":
            deleted = conn.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", [user_id]).fetchone()[0]
            conn.execute("DELETE FROM memories WHERE user_id = ?", [user_id])
            try:
                conn.execute("DELETE FROM conversation_buffer WHERE user_id = ?", [user_id])
            except Exception:
                pass
        else:  # sqlite
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,))
            deleted = cur.fetchone()[0]
            cur.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            try:
                cur.execute("DELETE FROM conversation_buffer WHERE user_id = ?", (user_id,))
            except Exception:
                pass
            conn.commit()
            cur.close()
    finally:
        conn.close()

    # Best-effort: drop the user's graph nodes so recall can't resurface them.
    try:
        from src.graph import get_graph_backend
        gb = get_graph_backend()
        for node in gb.get_all_nodes_for_user(user_id):
            try:
                gb.delete_node(node["memory_id"])
            except Exception:
                pass
    except Exception:
        pass

    # The audit entry is retained (immutability / accountability) — it holds only the
    # user id + count, never memory content, so it is not personal content.
    log_event("delete", "purge", user_id, detail={"count": deleted})
    return {"purged": True, "user_id": user_id, "deleted": deleted}


# ── POST /users/{user_id}/import ────────────────────────────────────────────────

class ImportRequest(BaseModel):
    memories: List[dict]
    overwrite: bool = False   # reserved; ON CONFLICT already upserts by (user_id, content)


@router.post("/users/{user_id}/import")
def import_user(user_id: str, req: ImportRequest):
    """Bulk restore/seed memories from an export. Re-embeds each item; idempotent on
    (user_id, content). Bypasses the relevance judge — this is restoring vetted data."""
    user_id = user_id.strip().lower()
    items = req.memories or []
    if not items:
        return {"imported": 0, "skipped": 0}

    backend = get_backend()
    conn = get_conn()
    cur = conn.cursor() if backend != "duckdb" else None
    imported, skipped = 0, 0
    try:
        for it in items:
            content = str(it.get("content", "")).strip()
            if len(content) < 2:
                skipped += 1
                continue
            importance = float(it.get("importance", 0.5) or 0.5)
            importance = max(0.0, min(1.0, importance))
            category = str(it.get("category", "") or "").strip().lower() or categorize(content)
            try:
                emb_str = emb_to_db(embed(content), backend)
                if backend == "postgres":
                    cur.execute(
                        "INSERT INTO memories (user_id, content, embedding, importance, category) "
                        "VALUES (%s, %s, %s::vector, %s, %s) "
                        "ON CONFLICT (user_id, content) DO UPDATE SET importance = EXCLUDED.importance",
                        (user_id, content, emb_str, importance, category))
                elif backend == "duckdb":
                    conn.execute(
                        "INSERT INTO memories (user_id, content, embedding, importance, category) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT (user_id, content) DO UPDATE SET importance = excluded.importance",
                        [user_id, content, emb_str, importance, category])
                else:
                    cur.execute(
                        "INSERT INTO memories (user_id, content, embedding, importance, category) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT (user_id, content) DO UPDATE SET importance = excluded.importance",
                        (user_id, content, emb_str, importance, category))
                imported += 1
            except Exception:
                skipped += 1
        if backend != "duckdb":
            conn.commit()
    finally:
        if cur:
            cur.close()
        conn.close()

    log_event("write", "import", user_id, detail={"imported": imported, "skipped": skipped})
    return {"imported": imported, "skipped": skipped}
