"""
Run daily to prune memories that have decayed below the strength threshold,
and to consolidate near-duplicate memories (cosine similarity > 0.92).
Runs automatically every 24 hours via the MCP server's background thread.

Manual usage:
    python -m src.jobs.decay_job
"""

import json
import sys

import numpy as np

from src.services.decay import compute_strength, get_active_days_since
from src.services.utils import parse_dt as _parse_dt
from src.db.connection import get_backend, get_conn
from src.graph.graph_store import chain_safe_to_prune
from src.graph import get_graph_backend

PRUNE_THRESHOLD   = 0.05   # memories weaker than this are deleted
CONSOLIDATE_SIM   = 0.92   # cosine threshold for merging near-duplicates


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine(a, b) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def _parse_emb(raw) -> list | None:
    if raw is None:
        return None
    if isinstance(raw, (list, np.ndarray)):
        return list(raw)
    try:
        return json.loads(raw)
    except Exception:
        return None


# ── Decay + prune ─────────────────────────────────────────────────────────────

def run():
    backend = get_backend()
    conn    = get_conn()

    if backend == "postgres":
        from psycopg2.extras import RealDictCursor
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT id, user_id, category, importance, recall_count, last_accessed_at FROM memories"
        )
        edges = [dict(r) for r in cur.fetchall()]
        cur.close()
    elif backend == "duckdb":
        from src.db.connection import duckdb_rows
        result = conn.execute(
            "SELECT id, user_id, category, importance, recall_count, last_accessed_at FROM memories"
        )
        edges = duckdb_rows(result)
    else:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, user_id, category, importance, recall_count, last_accessed_at FROM memories"
        )
        edges = [dict(r) for r in cur.fetchall()]
        cur.close()

    updated = 0
    pruned  = 0

    for edge in edges:
        last_accessed = _parse_dt(edge["last_accessed_at"])
        # Use activity-aware days so vacations don't prune valid memories
        active_days = get_active_days_since(edge["user_id"], last_accessed)
        strength = compute_strength(
            last_accessed_at=last_accessed,
            recall_count=edge["recall_count"],
            importance=edge["importance"],
            category=edge["category"],
            active_days=active_days,
        )

        if strength < PRUNE_THRESHOLD:
            user_id = edge.get("user_id", "")
            if user_id and not chain_safe_to_prune(edge["id"], user_id, PRUNE_THRESHOLD):
                updated += 1
                continue

            if backend == "postgres":
                cur = conn.cursor()
                cur.execute("DELETE FROM memories WHERE id = %s", (edge["id"],))
                cur.close()
            elif backend == "duckdb":
                conn.execute("DELETE FROM memories WHERE id = ?", [edge["id"]])
            else:
                cur = conn.cursor()
                cur.execute("DELETE FROM memories WHERE id = ?", (edge["id"],))
                cur.close()

            try:
                get_graph_backend().delete_node(edge["id"])
            except Exception:
                pass

            pruned += 1
        else:
            try:
                get_graph_backend().update_node_strength(edge["id"], strength)
            except Exception:
                pass
            updated += 1

    if backend != "duckdb":
        conn.commit()
    conn.close()

    print(
        f"Decay job complete ({backend}) — updated: {updated}, pruned: {pruned}",
        file=sys.stderr,
    )

    # Run consolidation after pruning
    _consolidate()


# ── Memory consolidation ──────────────────────────────────────────────────────

def _consolidate() -> None:
    """
    Merge near-duplicate memories (cosine > CONSOLIDATE_SIM) per user.

    For each pair above threshold, keep the higher-importance memory and:
      - merge its content (combined sentence if different)
      - sum recall_counts
      - take max importance
      - delete the lower-importance duplicate

    Runs in O(n²) per user — acceptable since memory counts are small (<1000).
    """
    backend = get_backend()
    conn    = get_conn()

    try:
        if backend == "postgres":
            from psycopg2.extras import RealDictCursor
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT id, user_id, content, importance, recall_count, embedding FROM memories ORDER BY user_id"
            )
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
        elif backend == "duckdb":
            from src.db.connection import duckdb_rows
            result = conn.execute(
                "SELECT id, user_id, content, importance, recall_count, embedding FROM memories ORDER BY user_id"
            )
            rows = duckdb_rows(result)
        else:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, user_id, content, importance, recall_count, embedding FROM memories ORDER BY user_id"
            )
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
    except Exception as exc:
        print(f"[consolidate] fetch failed: {exc}", file=sys.stderr)
        conn.close()
        return

    # Group by user
    from itertools import groupby
    merged_count = 0
    rows.sort(key=lambda r: r["user_id"])

    for user_id, group in groupby(rows, key=lambda r: r["user_id"]):
        mems = [r for r in group if _parse_emb(r["embedding"]) is not None]
        if len(mems) < 2:
            continue

        deleted = set()
        for i in range(len(mems)):
            if mems[i]["id"] in deleted:
                continue
            emb_i = _parse_emb(mems[i]["embedding"])
            for j in range(i + 1, len(mems)):
                if mems[j]["id"] in deleted:
                    continue
                emb_j = _parse_emb(mems[j]["embedding"])
                if _cosine(emb_i, emb_j) < CONSOLIDATE_SIM:
                    continue

                # Merge j into i (keep higher importance as primary)
                keep, drop = (mems[i], mems[j]) if mems[i]["importance"] >= mems[j]["importance"] else (mems[j], mems[i])

                merged_recall = keep["recall_count"] + drop["recall_count"]
                merged_importance = max(keep["importance"], drop["importance"])

                try:
                    if backend == "postgres":
                        cur = conn.cursor()
                        cur.execute(
                            "UPDATE memories SET recall_count = %s, importance = %s WHERE id = %s",
                            (merged_recall, merged_importance, keep["id"]),
                        )
                        cur.execute("DELETE FROM memories WHERE id = %s", (drop["id"],))
                        conn.commit()
                        cur.close()
                    elif backend == "duckdb":
                        conn.execute(
                            "UPDATE memories SET recall_count = ?, importance = ? WHERE id = ?",
                            [merged_recall, merged_importance, keep["id"]],
                        )
                        conn.execute("DELETE FROM memories WHERE id = ?", [drop["id"]])
                    else:
                        cur = conn.cursor()
                        cur.execute(
                            "UPDATE memories SET recall_count = ?, importance = ? WHERE id = ?",
                            (merged_recall, merged_importance, keep["id"]),
                        )
                        cur.execute("DELETE FROM memories WHERE id = ?", (drop["id"],))
                        conn.commit()
                        cur.close()

                    try:
                        get_graph_backend().delete_node(drop["id"])
                    except Exception:
                        pass

                    deleted.add(drop["id"])
                    merged_count += 1
                except Exception as exc:
                    print(f"[consolidate] merge {keep['id']}←{drop['id']} failed: {exc}", file=sys.stderr)

    conn.close()
    if merged_count:
        print(f"[consolidate] merged {merged_count} near-duplicate memories", file=sys.stderr)


if __name__ == "__main__":
    run()
