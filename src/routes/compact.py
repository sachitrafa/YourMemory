"""
Memory compaction endpoints.

  POST /compact?userId=…        → run compaction now for a user (returns stats)
  GET  /users/{user_id}/archive → view memories that were compressed into summaries

Compaction also runs daily when YOURMEMORY_COMPACTION=1. See src/services/compaction.py.
"""

from fastapi import APIRouter, Query
from typing import Optional

from src.services.compaction import compact_user
from src.db.connection import get_backend, get_conn, duckdb_rows

router = APIRouter()


@router.post("/compact")
def run_compaction(
    userId: str = Query(..., description="User whose memories to compact"),
    minCluster: Optional[int] = Query(None, ge=2, le=100),
    simThreshold: Optional[float] = Query(None, ge=0.3, le=0.99),
):
    return compact_user(userId, min_cluster=minCluster, sim_threshold=simThreshold)


@router.get("/users/{user_id}/archive")
def view_archive(user_id: str, limit: int = Query(100, ge=1, le=1000)):
    """Originals that were compressed into summaries (newest first)."""
    user_id = user_id.strip().lower()
    backend = get_backend()
    conn = get_conn()
    sql_pg    = ("SELECT orig_id, content, category, summary_id, archived_at FROM memory_archive "
                 "WHERE user_id = %s ORDER BY archived_at DESC LIMIT %s")
    sql_other = ("SELECT orig_id, content, category, summary_id, archived_at FROM memory_archive "
                 "WHERE user_id = ? ORDER BY archived_at DESC LIMIT ?")
    cols = ["orig_id", "content", "category", "summary_id", "archived_at"]
    try:
        if backend == "postgres":
            cur = conn.cursor(); cur.execute(sql_pg, (user_id, limit))
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]; cur.close()
        elif backend == "duckdb":
            rows = duckdb_rows(conn.execute(sql_other, [user_id, limit]))
        else:
            cur = conn.cursor(); cur.execute(sql_other, (user_id, limit))
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]; cur.close()
    finally:
        conn.close()
    for r in rows:
        if r.get("archived_at") is not None:
            r["archived_at"] = str(r["archived_at"])
    return {"count": len(rows), "archived": rows}
