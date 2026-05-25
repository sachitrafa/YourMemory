"""
Lightweight temporal query detection and scoring boost.

No new dependencies — uses spaCy (already loaded in extract.py) for DATE/TIME
entity detection, with a pure-Python regex fallback.

Only activates when a temporal expression is detected in the query.
Zero performance impact on non-temporal queries.
"""

import re
from datetime import datetime, timezone, timedelta

TEMPORAL_BOOST = 0.25  # added to hybrid_score for memories within the resolved window

# (regex_pattern, days_ago, window_days)
# days_ago: how far back the window starts
# window_days: width of the window (used for open-ended "recently" type expressions)
_PATTERNS = [
    (r"\btoday\b",                              0,   1),
    (r"\byesterday\b",                          1,   1),
    (r"\blast\s+(?:24\s*hours?|day)\b",         1,   1),
    (r"\bthis\s+week\b",                        7,   7),
    (r"\blast\s+week\b",                        7,   7),
    (r"\bpast\s+week\b",                        7,   7),
    (r"\blast\s+(?:7|seven)\s+days?\b",         7,   7),
    (r"\blast\s+(?:2|two)\s+weeks?\b",         14,  14),
    (r"\bthis\s+(?:sprint|iteration)\b",       14,  14),
    (r"\blast\s+sprint\b",                     14,  14),
    (r"\brecently\b",                          14,  14),
    (r"\bthis\s+month\b",                      30,  30),
    (r"\blast\s+month\b",                      30,  30),
    (r"\bpast\s+month\b",                      30,  30),
    (r"\blast\s+(?:30|thirty)\s+days?\b",      30,  30),
    (r"\bearlier\b",                           30,  30),
    (r"\bpreviously\b",                        30,  30),
    (r"\blast\s+(?:2|two)\s+months?\b",        60,  60),
    (r"\blast\s+(?:3|three)\s+months?\b",      90,  90),
    (r"\blast\s+quarter\b",                    90,  90),
]


def detect_temporal_range(query: str) -> tuple[datetime, datetime] | None:
    """
    Detect a relative time expression in the query and resolve to (start, end) UTC.
    Returns None if no temporal expression found — caller skips the boost entirely.

    Strategy:
      1. Use spaCy DATE/TIME entities (already loaded, zero extra cost)
      2. Fall back to regex patterns
    """
    now = datetime.now(timezone.utc)
    q_lower = query.lower()

    # spaCy pass — reuse the already-loaded model from extract.py
    try:
        from src.services.extract import _nlp
        if _nlp is not None:
            doc = _nlp(query)
            for ent in doc.ents:
                if ent.label_ in ("DATE", "TIME"):
                    result = _match_patterns(ent.text.lower(), now)
                    if result:
                        return result
    except Exception:
        pass

    # Regex fallback
    return _match_patterns(q_lower, now)


def _match_patterns(text: str, now: datetime) -> tuple[datetime, datetime] | None:
    for pattern, days_ago, _ in _PATTERNS:
        if re.search(pattern, text):
            start = now - timedelta(days=days_ago)
            return (start, now)
    return None


def score_boost(created_at, date_range: tuple[datetime, datetime] | None) -> float:
    """
    Return TEMPORAL_BOOST if created_at falls within date_range, else 0.0.
    Safe to call with None created_at or None date_range.
    """
    if not date_range or not created_at:
        return 0.0
    start, end = date_range
    if isinstance(created_at, str):
        from src.services.utils import parse_dt
        created_at = parse_dt(created_at)
    if created_at is None:
        return 0.0
    # ensure timezone-aware comparison
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return TEMPORAL_BOOST if start <= created_at <= end else 0.0
