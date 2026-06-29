"""
Memory compaction endpoints.

  POST /compact?userId=…        → run compaction now for a user (returns stats)
  GET  /users/{user_id}/archive → view memories that were compressed into summaries

Compaction also runs daily when YOURMEMORY_COMPACTION=1. See src/services/compaction.py.
"""

from threading import Lock
from fastapi import APIRouter, BackgroundTasks, Query
from typing import Optional

from src.services.compaction import compact_user
from src.db.connection import get_backend, get_conn, duckdb_rows

router = APIRouter()

# A full-scan compaction can take minutes on a large store, so it runs in the background
# by default and the endpoint returns immediately. This guard stops a second request from
# launching a duplicate run for the same user while one is in flight.
_running: set = set()
_running_lock = Lock()


@router.post("/compact")
def run_compaction(
    background: BackgroundTasks,
    userId: str = Query(..., description="User whose memories to compact"),
    minCluster: Optional[int] = Query(None, ge=2, le=100),
    simThreshold: Optional[float] = Query(None, ge=0.3, le=0.99),
    wait: bool = Query(False, description="Run synchronously and return stats (small stores only)"),
):
    uid = userId.strip().lower()

    # Synchronous mode (opt-in) — returns the full stats; only use on small stores.
    if wait:
        return compact_user(uid, min_cluster=minCluster, sim_threshold=simThreshold)

    # Default: fire-and-forget. Return immediately; observe via /users/{id}/archive or /audit.
    with _running_lock:
        if uid in _running:
            return {"status": "already_running", "userId": uid}
        _running.add(uid)

    def _job():
        try:
            compact_user(uid, min_cluster=minCluster, sim_threshold=simThreshold)
        finally:
            with _running_lock:
                _running.discard(uid)

    background.add_task(_job)
    return {"status": "started", "userId": uid,
            "note": "compaction running in background — check /users/{id}/archive or /audit (operation=compact)"}


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
