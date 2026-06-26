"""
Audit trail — append-only, tamper-evident logging of every read/write/delete on a
memory, for SOC 2 / compliance.

Design
------
* Each event records: timestamp (UTC), actor (user_id + optional agent_id), action
  class (read|write|delete|admin), specific operation, target memory id, a small
  non-sensitive detail blob, and the source surface (http|mcp).
* IMMUTABILITY is tamper-evident via a hash chain: every row stores
  row_hash = sha256(prev_hash || canonical(fields)), where prev_hash is the previous
  row's hash. Any later edit or deletion inside the retained window breaks the chain
  and is detected by verify_chain(). The app never UPDATEs/DELETEs audit_log except
  the controlled retention prune.
* RETENTION: prune_expired() deletes rows older than the retention window, which is
  floored at 90 days (a smaller value is silently raised to 90).
* Privacy: we log memory *ids* and lightweight metadata (counts, query length) — never
  raw memory content or query text — so the audit log itself isn't a data-leak vector.
* Fail-open: logging never raises into the caller; a logging failure must not break a
  memory operation. Integrity gaps are themselves detectable via verify_chain().
"""

import hashlib
import json
import os
from datetime import datetime, timezone

from src.db.connection import get_backend, get_conn

GENESIS = "0" * 64
RETENTION_FLOOR_DAYS = 90
_ENABLED = os.getenv("YOURMEMORY_AUDIT", "1").lower() not in ("0", "false", "off", "no")

# Fixed-width UTC timestamp (always microseconds + 'Z'). Stored verbatim as TEXT so it
# round-trips losslessly for the hash chain, and so lexicographic order == chronological
# order for retention compares across all three backends.
_TS_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(_TS_FMT)


def _retention_days() -> int:
    try:
        d = int(os.getenv("YOURMEMORY_AUDIT_RETENTION_DAYS", str(RETENTION_FLOOR_DAYS)))
    except ValueError:
        d = RETENTION_FLOOR_DAYS
    return max(RETENTION_FLOOR_DAYS, d)   # 90-day minimum, never lower


def _row_hash(prev_hash: str, ts: str, user_id: str, agent_id, action: str,
              operation: str, target_id, detail: str) -> str:
    canon = "|".join([
        prev_hash, ts, user_id, agent_id or "", action, operation,
        "" if target_id is None else str(target_id), detail or "",
    ])
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _prev_hash(conn, backend: str) -> str:
    """Hash of the most recent audit row (chain head), or GENESIS if empty."""
    sql = "SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1"
    try:
        if backend == "duckdb":
            row = conn.execute(sql).fetchone()
        else:
            cur = conn.cursor()
            cur.execute(sql)
            row = cur.fetchone()
            cur.close()
        return row[0] if row else GENESIS
    except Exception:
        return GENESIS


def log_event(action: str, operation: str, user_id: str, *, agent_id: str = None,
              target_id=None, detail: dict = None, source: str = "http") -> None:
    """Append one audit row. Never raises into the caller."""
    if not _ENABLED:
        return
    try:
        ts = _now_iso()
        detail_str = json.dumps(detail, separators=(",", ":"), sort_keys=True) if detail else ""
        uid = (user_id or "unknown").strip().lower()
        backend = get_backend()
        conn = get_conn()
        try:
            prev = _prev_hash(conn, backend)
            rh = _row_hash(prev, ts, uid, agent_id, action, operation, target_id, detail_str)
            cols = ("ts, actor_user_id, actor_agent_id, action, operation, "
                    "target_id, detail, source, prev_hash, row_hash")
            vals = (ts, uid, agent_id, action, operation, target_id, detail_str, source, prev, rh)
            if backend == "postgres":
                cur = conn.cursor()
                cur.execute(f"INSERT INTO audit_log ({cols}) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", vals)
                conn.commit()
                cur.close()
            elif backend == "duckdb":
                conn.execute(f"INSERT INTO audit_log ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?)", list(vals))
            else:  # sqlite
                cur = conn.cursor()
                cur.execute(f"INSERT INTO audit_log ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?)", vals)
                conn.commit()
                cur.close()
        finally:
            conn.close()
    except Exception:
        pass  # fail-open: auditing must never break a memory operation


def get_events(user_id: str = None, limit: int = 100, action: str = None,
               operation: str = None) -> list[dict]:
    """Read audit rows (newest first), optionally filtered."""
    backend = get_backend()
    conn = get_conn()
    ph = "%s" if backend == "postgres" else "?"
    where, params = [], []
    if user_id:
        where.append(f"actor_user_id = {ph}"); params.append(user_id.strip().lower())
    if action:
        where.append(f"action = {ph}"); params.append(action)
    if operation:
        where.append(f"operation = {ph}"); params.append(operation)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    limit = max(1, min(int(limit), 1000))
    sql = (f"SELECT id, ts, actor_user_id, actor_agent_id, action, operation, "
           f"target_id, detail, source, prev_hash, row_hash FROM audit_log "
           f"{clause} ORDER BY id DESC LIMIT {limit}")
    cols = ["id", "ts", "actor_user_id", "actor_agent_id", "action", "operation",
            "target_id", "detail", "source", "prev_hash", "row_hash"]
    try:
        if backend == "duckdb":
            rows = conn.execute(sql, params).fetchall()
        else:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def verify_chain() -> dict:
    """Walk the chain oldest→newest and confirm tamper-evidence holds.

    Returns {"ok": bool, "rows": N, "broken_at": id|None, "reason": str|None}.
    The oldest retained row is the anchor (its prev may point at a pruned row), so a
    retention prune of the tail does not register as tampering — only edits/deletes
    *within* the retained window do.
    """
    backend = get_backend()
    conn = get_conn()
    sql = ("SELECT id, ts, actor_user_id, actor_agent_id, action, operation, "
           "target_id, detail, prev_hash, row_hash FROM audit_log ORDER BY id ASC")
    try:
        if backend == "duckdb":
            rows = conn.execute(sql).fetchall()
        else:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cur.close()
    finally:
        conn.close()

    prev_row_hash = None
    for r in rows:
        (rid, ts, uid, agent_id, action, operation, target_id, detail, prev_hash, row_hash) = r
        # Linkage: each row's prev_hash must equal the previous row's row_hash.
        if prev_row_hash is not None and prev_hash != prev_row_hash:
            return {"ok": False, "rows": len(rows), "broken_at": rid,
                    "reason": "prev_hash does not match prior row"}
        # Integrity: recomputed hash must match the stored hash.
        expect = _row_hash(prev_hash, ts if isinstance(ts, str) else str(ts),
                           uid, agent_id, action, operation, target_id, detail)
        if expect != row_hash:
            return {"ok": False, "rows": len(rows), "broken_at": rid,
                    "reason": "row_hash mismatch (row content altered)"}
        prev_row_hash = row_hash
    return {"ok": True, "rows": len(rows), "broken_at": None, "reason": None}


def prune_expired() -> int:
    """Delete audit rows older than the retention window (floored at 90 days).
    The only sanctioned deletion path; returns the number of rows removed.
    Compares against a fixed-format ISO cutoff (ts is stored as TEXT, so lexicographic
    order is chronological)."""
    from datetime import timedelta
    backend = get_backend()
    days = _retention_days()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(_TS_FMT)
    conn = get_conn()
    try:
        if backend == "postgres":
            cur = conn.cursor()
            cur.execute("DELETE FROM audit_log WHERE ts < %s", (cutoff,))
            n = cur.rowcount
            conn.commit()
            cur.close()
        elif backend == "duckdb":
            before = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            conn.execute("DELETE FROM audit_log WHERE ts < ?", [cutoff])
            after = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            n = before - after
        else:  # sqlite
            cur = conn.cursor()
            cur.execute("DELETE FROM audit_log WHERE ts < ?", (cutoff,))
            n = cur.rowcount
            conn.commit()
            cur.close()
        return n if n and n > 0 else 0
    finally:
        conn.close()
