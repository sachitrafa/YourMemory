"""
Shared memory pools — team / institutional memory.

A pool is a shared namespace many agents contribute to and everyone with access can
query. Individual agents still write their own personal memories; the pool is the
team-level layer on top. Pool memories are stored in the `memories` table under a
namespaced user_id (`pool:<id>`), so they reuse embedding, dedup, recall, decay,
compaction, and audit for free.

  POST /pools                         create a pool
  GET  /pools                         list pools
  POST /pools/{id}/members            add a member with a role (reader/contributor/admin)
  GET  /pools/{id}/members            list members
  POST /pools/{id}/memories           write a memory into the pool (writer role required)
  GET  /pools/{id}/memories           list pool memories (reader role required)
  POST /pools/{id}/retrieve           semantic search within the pool

Role → permissions:  reader = read · contributor = read+write · admin = read+write+manage.

> NOTE: in the default local deployment the caller identity (`memberId`) is supplied by
> the client and access checks are advisory — real enforcement arrives with hosted auth.
> The pool model, membership, and union recall are the durable substrate that hosted
> RBAC will enforce.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List

from src.db.connection import get_backend, get_conn, duckdb_rows
from src.services.audit import log_event

router = APIRouter()

POOL_NS = "pool:"          # memories namespace prefix
_ROLE_PERMS = {
    "reader":      (True,  False),
    "contributor": (True,  True),
    "admin":       (True,  True),
}


def _ns(pool_id: str) -> str:
    return f"{POOL_NS}{pool_id.strip().lower()}"


def _ph(backend: str) -> str:
    return "%s" if backend == "postgres" else "?"


def _member(conn, backend: str, pool_id: str, member_id: str) -> Optional[dict]:
    p = _ph(backend)
    sql = f"SELECT role, can_read, can_write FROM pool_members WHERE pool_id = {p} AND member_id = {p}"
    args = (pool_id, member_id)
    if backend == "duckdb":
        rows = duckdb_rows(conn.execute(sql, list(args)))
        return rows[0] if rows else None
    cur = conn.cursor(); cur.execute(sql, args)
    row = cur.fetchone(); cur.close()
    if not row:
        return None
    return {"role": row[0], "can_read": bool(row[1]), "can_write": bool(row[2])}


def _require(pool_id: str, member_id: str, need: str):
    """Advisory access check. need = 'read' | 'write'. Returns the membership dict."""
    if not member_id:
        raise HTTPException(403, "memberId required to access a pool.")
    backend = get_backend(); conn = get_conn()
    try:
        m = _member(conn, backend, pool_id.strip().lower(), member_id.strip().lower())
    finally:
        conn.close()
    if not m:
        raise HTTPException(403, f"'{member_id}' is not a member of pool '{pool_id}'.")
    if need == "read" and not m["can_read"]:
        raise HTTPException(403, "Member lacks read access to this pool.")
    if need == "write" and not m["can_write"]:
        raise HTTPException(403, "Member lacks write access to this pool.")
    return m


# ── Create / list pools ─────────────────────────────────────────────────────────

class CreatePoolRequest(BaseModel):
    pool_id: str
    name:    Optional[str] = ""
    owner:   str


@router.post("/pools")
def create_pool(req: CreatePoolRequest):
    pool_id = req.pool_id.strip().lower()
    owner   = req.owner.strip().lower()
    if not pool_id or not owner:
        raise HTTPException(422, "pool_id and owner are required.")
    backend = get_backend(); conn = get_conn(); p = _ph(backend)
    try:
        # Create the pool and make the owner an admin member.
        if backend == "duckdb":
            conn.execute(f"INSERT INTO pools (pool_id, name, owner) VALUES (?,?,?) ON CONFLICT (pool_id) DO NOTHING",
                         [pool_id, req.name, owner])
            conn.execute("INSERT INTO pool_members (pool_id, member_id, role, can_read, can_write) "
                         "VALUES (?,?,?,?,?) ON CONFLICT (pool_id, member_id) DO UPDATE SET role=excluded.role",
                         [pool_id, owner, "admin", True, True])
        else:
            cur = conn.cursor()
            cur.execute(f"INSERT INTO pools (pool_id, name, owner) VALUES ({p},{p},{p}) ON CONFLICT (pool_id) DO NOTHING",
                        (pool_id, req.name, owner))
            cur.execute(f"INSERT INTO pool_members (pool_id, member_id, role, can_read, can_write) "
                        f"VALUES ({p},{p},{p},{p},{p}) ON CONFLICT (pool_id, member_id) DO UPDATE SET role=excluded.role",
                        (pool_id, owner, "admin", True, True))
            conn.commit(); cur.close()
    finally:
        conn.close()
    log_event("write", "pool_create", owner, detail={"pool_id": pool_id})
    return {"pool_id": pool_id, "name": req.name, "owner": owner, "role": "admin"}


@router.get("/pools")
def list_pools(memberId: Optional[str] = Query(None, description="Filter to pools this member belongs to")):
    backend = get_backend(); conn = get_conn(); p = _ph(backend)
    try:
        if memberId:
            mid = memberId.strip().lower()
            sql = (f"SELECT p.pool_id, p.name, p.owner, m.role FROM pools p "
                   f"JOIN pool_members m ON p.pool_id = m.pool_id WHERE m.member_id = {p} ORDER BY p.pool_id")
            args = (mid,)
        else:
            sql = "SELECT pool_id, name, owner, '' AS role FROM pools ORDER BY pool_id"
            args = ()
        if backend == "duckdb":
            rows = duckdb_rows(conn.execute(sql, list(args)))
        else:
            cur = conn.cursor(); cur.execute(sql, args)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]; cur.close()
    finally:
        conn.close()
    return {"count": len(rows), "pools": rows}


# ── Membership ──────────────────────────────────────────────────────────────────

class AddMemberRequest(BaseModel):
    member_id: str
    role:      str = "reader"   # reader | contributor | admin


@router.post("/pools/{pool_id}/members")
def add_member(pool_id: str, req: AddMemberRequest):
    pool_id = pool_id.strip().lower()
    member_id = req.member_id.strip().lower()
    role = req.role.strip().lower()
    if role not in _ROLE_PERMS:
        raise HTTPException(422, f"role must be one of {list(_ROLE_PERMS)}")
    can_read, can_write = _ROLE_PERMS[role]
    backend = get_backend(); conn = get_conn(); p = _ph(backend)
    try:
        if backend == "duckdb":
            conn.execute("INSERT INTO pool_members (pool_id, member_id, role, can_read, can_write) "
                         "VALUES (?,?,?,?,?) ON CONFLICT (pool_id, member_id) DO UPDATE SET "
                         "role=excluded.role, can_read=excluded.can_read, can_write=excluded.can_write",
                         [pool_id, member_id, role, can_read, can_write])
        else:
            cur = conn.cursor()
            cur.execute(f"INSERT INTO pool_members (pool_id, member_id, role, can_read, can_write) "
                        f"VALUES ({p},{p},{p},{p},{p}) ON CONFLICT (pool_id, member_id) DO UPDATE SET "
                        f"role=excluded.role, can_read=excluded.can_read, can_write=excluded.can_write",
                        (pool_id, member_id, role, can_read, can_write))
            conn.commit(); cur.close()
    finally:
        conn.close()
    return {"pool_id": pool_id, "member_id": member_id, "role": role,
            "can_read": can_read, "can_write": can_write}


@router.get("/pools/{pool_id}/members")
def list_members(pool_id: str):
    pool_id = pool_id.strip().lower()
    backend = get_backend(); conn = get_conn(); p = _ph(backend)
    sql = f"SELECT member_id, role, can_read, can_write FROM pool_members WHERE pool_id = {p} ORDER BY member_id"
    try:
        if backend == "duckdb":
            rows = duckdb_rows(conn.execute(sql, [pool_id]))
        else:
            cur = conn.cursor(); cur.execute(sql, (pool_id,))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]; cur.close()
    finally:
        conn.close()
    return {"pool_id": pool_id, "count": len(rows), "members": rows}


# ── Pool memories: write / list / retrieve (reuse the memory engine) ────────────

class PoolMemoryRequest(BaseModel):
    memberId:   str
    content:    str
    importance: float = 0.5
    category:   Optional[str] = None


@router.post("/pools/{pool_id}/memories")
def write_pool_memory(pool_id: str, req: PoolMemoryRequest):
    _require(pool_id, req.memberId, "write")
    from src.routes.memories import add_memory, MemoryRequest
    result = add_memory(MemoryRequest(
        userId=_ns(pool_id), content=req.content, importance=req.importance))
    log_event("write", "pool_store", req.memberId.strip().lower(),
              detail={"pool_id": pool_id.strip().lower(), "id": result.get("id")})
    return {"pool_id": pool_id.strip().lower(), **result}


@router.get("/pools/{pool_id}/memories")
def list_pool_memories(pool_id: str, memberId: str = Query(...), limit: int = Query(50, ge=1, le=500)):
    _require(pool_id, memberId, "read")
    from src.routes.memories import list_memories
    return list_memories(userId=_ns(pool_id), limit=limit)


class PoolRetrieveRequest(BaseModel):
    memberId: str
    query:    str
    topK:     int = 6


@router.post("/pools/{pool_id}/retrieve")
def retrieve_pool(pool_id: str, req: PoolRetrieveRequest):
    _require(pool_id, req.memberId, "read")
    from src.services.retrieve import retrieve as _retrieve
    result = _retrieve(_ns(pool_id), req.query, req.topK)
    log_event("read", "pool_retrieve", req.memberId.strip().lower(),
              detail={"pool_id": pool_id.strip().lower(),
                      "count": len(result.get("memories", []))})
    return result
