"""
Run daily to prune memories that have decayed below the strength threshold.
Runs automatically every 24 hours via the MCP server's background thread.

Manual usage:
    python -m src.jobs.decay_job
"""

import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

from src.services.decay import compute_strength
from src.db.connection import get_backend, get_conn

load_dotenv()

PRUNE_THRESHOLD = 0.05  # memories weaker than this are deleted


def _parse_dt(value) -> datetime:
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(timezone.utc)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return datetime.now(timezone.utc)


def run():
    backend = get_backend()
    conn    = get_conn()
    cur     = conn.cursor()

    if backend == "postgres":
        from psycopg2.extras import RealDictCursor
        cur.close()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, category, importance, recall_count, last_accessed_at FROM memories")
        edges = [dict(r) for r in cur.fetchall()]
    elif backend == "duckdb":
        from src.db.connection import duckdb_rows
        result = conn.execute("SELECT id, category, importance, recall_count, last_accessed_at FROM memories")
        edges = duckdb_rows(result)
    else:
        cur.execute("SELECT id, category, importance, recall_count, last_accessed_at FROM memories")
        edges = [dict(r) for r in cur.fetchall()]

    updated = 0
    pruned  = 0

    for edge in edges:
        strength = compute_strength(
            last_accessed_at=_parse_dt(edge["last_accessed_at"]),
            recall_count=edge["recall_count"],
            importance=edge["importance"],
            category=edge["category"],
        )

        if strength < PRUNE_THRESHOLD:
            if backend == "postgres":
                cur.execute("DELETE FROM memories WHERE id = %s", (edge["id"],))
            elif backend == "duckdb":
                conn.execute("DELETE FROM memories WHERE id = ?", [edge["id"]])
            else:
                cur.execute("DELETE FROM memories WHERE id = ?", (edge["id"],))
            pruned += 1
        else:
            updated += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"Decay job complete ({backend}) — updated: {updated}, pruned: {pruned}", file=sys.stderr)


if __name__ == "__main__":
    run()
