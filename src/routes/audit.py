"""Audit-trail read API.

  GET /audit          → recent audit events (filterable by user/action/operation)
  GET /audit/verify   → tamper-evidence check over the hash chain
  POST /audit/prune   → run retention prune now (>=90-day floor); also runs daily

The audit log is append-only and hash-chained; these endpoints are read/verify only
(plus the sanctioned retention prune). See src/services/audit.py for the model.
"""

from typing import Optional
from fastapi import APIRouter, Query

from src.services.audit import get_events, verify_chain, prune_expired, _retention_days

router = APIRouter()


@router.get("/audit")
def list_audit(
    userId:    Optional[str] = Query(None, description="Filter by actor user id"),
    action:    Optional[str] = Query(None, description="read | write | delete | admin"),
    operation: Optional[str] = Query(None, description="e.g. retrieve, store, update, delete"),
    limit:     int           = Query(100, ge=1, le=1000),
):
    events = get_events(user_id=userId, limit=limit, action=action, operation=operation)
    return {
        "count":          len(events),
        "retention_days": _retention_days(),
        "events":         events,
    }


@router.get("/audit/verify")
def verify_audit():
    """Confirm the audit chain has not been tampered with."""
    return verify_chain()


@router.post("/audit/prune")
def prune_audit():
    """Run the retention prune now. Deletes only rows older than the (>=90-day) window."""
    removed = prune_expired()
    return {"pruned": removed, "retention_days": _retention_days()}
