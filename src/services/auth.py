"""Caller authentication + trusted-identity resolution.

Turns the existing ``ym_`` agent API keys into real access control. Without this
layer, ``userId`` / ``memberId`` are supplied by the client and the pool RBAC
checks are only advisory (safe solely because the default deployment is local
and loopback-bound).

Two modes, chosen by ``YOURMEMORY_AUTH``:

* ``off`` (default) — backward compatible. If a valid API key is present the
  caller is authenticated and identity is locked to it; otherwise the caller is
  unauthenticated and routes fall back to the client-supplied id, exactly as
  before. This keeps every existing local install (MCP hooks, single OS user)
  working unchanged.

* ``required`` — every request must carry a valid API key. Missing/invalid keys
  get 401, and identity is always taken from the key — the client can no longer
  claim to be someone else. This is the hosted / multi-tenant posture.

Usage in a route:

    from src.services.auth import Caller, require_caller, effective_user

    @router.get("/memories")
    def list_memories(userId: str = Query(...),
                      caller: Caller = Depends(require_caller)):
        userId = effective_user(caller, userId)   # trusted id wins when authed
        ...

``effective_user`` is deliberately safe to call with a non-``Caller`` value, so
functions that are ALSO invoked internally as plain Python (e.g. add_memory,
called by the pool write path) don't misbehave when no dependency was injected.
"""
import os
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, Request

from src.services.api_keys import KEY_PREFIX, validate_api_key


@dataclass
class Caller:
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    authenticated: bool = False


def _mode() -> str:
    return os.getenv("YOURMEMORY_AUTH", "off").strip().lower()


def _extract_key(request: Request) -> Optional[str]:
    """Pull a ym_ key from `Authorization: Bearer ...` or `X-YourMemory-Key`."""
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        tok = auth[7:].strip()
        if tok.startswith(KEY_PREFIX):
            return tok
    tok = request.headers.get("x-yourmemory-key", "").strip()
    return tok if tok.startswith(KEY_PREFIX) else None


def require_caller(request: Request) -> Caller:
    """FastAPI dependency. Returns a Caller; enforces auth when YOURMEMORY_AUTH=required."""
    mode = _mode()
    key = _extract_key(request)

    if key:
        agent = validate_api_key(key)   # None if invalid/revoked
        if agent:
            return Caller(user_id=(agent.get("user_id") or "").strip().lower(),
                          agent_id=agent.get("agent_id"),
                          authenticated=True)
        if mode == "required":
            raise HTTPException(401, "Invalid or revoked API key.")

    if mode == "required":
        raise HTTPException(
            401,
            "Authentication required. Provide a YourMemory API key via "
            "'Authorization: Bearer ym_...' or the 'X-YourMemory-Key' header.",
        )

    # Local / trusted mode, no key: unauthenticated — the route keeps its
    # existing client-supplied identity behaviour.
    return Caller()


def effective_user(caller, claimed: Optional[str]) -> Optional[str]:
    """The identity a request may act as.

    When the caller is authenticated, their key's user_id wins and any
    client-claimed id is ignored — so a user can only ever touch their own
    memory / pools. Otherwise (local mode, or an internal non-HTTP call where
    ``caller`` is not a Caller) the claimed id is used, preserving prior behaviour.
    """
    if isinstance(caller, Caller) and caller.authenticated:
        return caller.user_id
    return claimed
