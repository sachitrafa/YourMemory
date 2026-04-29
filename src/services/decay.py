import math
from datetime import datetime, date, timezone

from src.db.connection import get_backend, get_conn

# Base decay rates per category.
# Higher λ = faster decay = shorter survival time.
#
# Category survival (importance=0.5, never recalled, prune threshold=0.05):
#   fact      λ=0.16  → ~24 days
#   assumption λ=0.20 → ~19 days
#   failure   λ=0.35  → ~11 days  (environment changes, old failures go stale fast)
#   strategy  λ=0.10  → ~38 days  (successful strategies are more durable)

DECAY_RATES = {
    "fact":       0.16,
    "assumption": 0.20,
    "failure":    0.35,
    "strategy":   0.10,
}
DEFAULT_DECAY_RATE = 0.16


def compute_strength(
    last_accessed_at: datetime,
    recall_count: int,
    importance: float = 0.5,
    category: str = "fact",
    active_days: float | None = None,
) -> float:
    """
    Ebbinghaus forgetting curve with importance-modulated decay rate,
    tuned per memory category:

        base_λ      = DECAY_RATES[category]
        effective_λ = base_λ × (1 - importance × 0.8)
        strength    = importance × e^(-effective_λ × days) × (1 + recall_count × 0.2)

    active_days: if provided, use the number of *active* user days since
    last_accessed_at instead of wall-clock days. This prevents vacations
    from prematurely decaying valid memories.

    Failure memories decay fastest — a rate-limit from 3 months ago is likely stale.
    Strategy memories decay slowest — successful patterns stay relevant longer.
    """
    if active_days is not None:
        days = max(0.0, active_days)
    else:
        now = datetime.now(timezone.utc)
        if last_accessed_at.tzinfo is None:
            last_accessed_at = last_accessed_at.replace(tzinfo=timezone.utc)
        days = (now - last_accessed_at).total_seconds() / 86400

    base_lambda = DECAY_RATES.get(category, DEFAULT_DECAY_RATE)
    effective_lambda = base_lambda * (1 - importance * 0.8)
    strength = importance * math.exp(-effective_lambda * days) * (1 + recall_count * 0.2)

    return round(min(1.0, strength), 6)


def record_activity(user_id: str) -> None:
    """Record today as an active day for this user (idempotent, best-effort)."""
    today = date.today().isoformat()
    backend = get_backend()
    conn = get_conn()
    try:
        if backend == "postgres":
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO user_activity (user_id, active_on) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (user_id, today),
            )
            conn.commit()
            cur.close()
        elif backend == "duckdb":
            conn.execute(
                "INSERT OR IGNORE INTO user_activity (user_id, active_on) VALUES (?, ?)",
                [user_id, today],
            )
        else:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO user_activity (user_id, active_on) VALUES (?, ?)",
                (user_id, today),
            )
            conn.commit()
            cur.close()
    except Exception:
        pass
    finally:
        conn.close()


def get_active_days_since(user_id: str, since: datetime) -> float:
    """
    Return the number of days user_id was active since `since`.
    Falls back to wall-clock days if the user_activity table has no data.
    """
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    since_date = since.date().isoformat()
    today = date.today().isoformat()

    backend = get_backend()
    conn = get_conn()
    count = None
    try:
        if backend == "postgres":
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM user_activity WHERE user_id = %s AND active_on >= %s AND active_on <= %s",
                (user_id, since_date, today),
            )
            row = cur.fetchone()
            count = row[0] if row else None
            cur.close()
        elif backend == "duckdb":
            from src.db.connection import duckdb_rows
            result = conn.execute(
                "SELECT COUNT(*) AS cnt FROM user_activity WHERE user_id = ? AND active_on >= ? AND active_on <= ?",
                [user_id, since_date, today],
            )
            rows = duckdb_rows(result)
            count = rows[0]["cnt"] if rows else None
        else:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM user_activity WHERE user_id = ? AND active_on >= ? AND active_on <= ?",
                (user_id, since_date, today),
            )
            row = cur.fetchone()
            count = row[0] if row else None
            cur.close()
    except Exception:
        pass
    finally:
        conn.close()

    if count is not None and count > 0:
        return float(count)

    # Fallback: wall-clock days
    now = datetime.now(timezone.utc)
    return max(0.0, (now - since).total_seconds() / 86400)
