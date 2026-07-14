"""
Regression tests for shared memory pools.

Guards the bug where GET /pools/{id}/memories returned HTTP 500:
`list_memories` is a FastAPI route, but pools.py / memory_mcp.py called it as a
plain Python function. Un-passed params kept their Query(...) sentinel defaults,
which are TRUTHY — so `if category:` fired and a Query object was bound into SQL
("can't adapt type 'Query'"). Fixed by splitting out list_memories_core() with
real Python defaults.

Uses a dedicated DuckDB so the real DB is never touched.

Run:
    python tests/test_pools.py
"""

import inspect
import os
import sys

# ── Isolated test DB ──────────────────────────────────────────────────────────
# BOTH lines matter. DATABASE_URL must be blanked first: connection.py calls
# load_dotenv(), which would otherwise pull a real postgres URL out of .env and
# these tests would run against the PRODUCTION database. Setting it to "" makes
# get_backend() fall back to DuckDB, and load_dotenv() won't override a key that
# already exists. YOURMEMORY_DB then points DuckDB at a throwaway file.
TEST_DB = "/tmp/yourmemory_test_pools.duckdb"
if os.path.exists(TEST_DB):
    os.remove(TEST_DB)
os.environ["DATABASE_URL"] = ""
os.environ["YOURMEMORY_DB"] = TEST_DB

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.migrate import migrate
migrate()

from fastapi import HTTPException
from fastapi import params as fastapi_params

from src.routes.memories import list_memories, list_memories_core
from src.routes.pools import (
    AddMemberRequest,
    CreatePoolRequest,
    PoolMemoryRequest,
    add_member,
    create_pool,
    list_pool_memories,
    write_pool_memory,
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

POOL = "test_pool"
OWNER = "test_owner"
MEMBER = "test_member"
OUTSIDER = "test_outsider"
results = []


def ok(name):
    print(f"  {PASS}  {name}")
    results.append((name, True))


def fail(name, reason=""):
    print(f"  {FAIL}  {name}" + (f"  ← {reason}" if reason else ""))
    results.append((name, False))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. The root cause: core function must have REAL defaults, not Query sentinels
# ═══════════════════════════════════════════════════════════════════════════════
print("\nlist_memories_core signature (root cause of the 500)")

sig = inspect.signature(list_memories_core)
sentinels = [
    n for n, p in sig.parameters.items()
    if isinstance(p.default, fastapi_params.Param)
]
if not sentinels:
    ok("list_memories_core has real Python defaults (no FastAPI Query sentinels)")
else:
    fail("list_memories_core has real Python defaults", f"Query sentinels: {sentinels}")

# The route may (and should) still use Query() — it's resolved by FastAPI.
route_sig = inspect.signature(list_memories)
if any(isinstance(p.default, fastapi_params.Param) for p in route_sig.parameters.values()):
    ok("route list_memories still declares Query params (FastAPI resolves them)")
else:
    fail("route list_memories still declares Query params")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Calling the core from plain Python with un-passed params must not crash
# ═══════════════════════════════════════════════════════════════════════════════
print("\nDirect Python call (the exact pattern that used to 500)")

try:
    res = list_memories_core(userId="nobody_here", limit=5)
    if isinstance(res, dict) and "memories" in res:
        ok("list_memories_core(userId, limit) works without passing category/agent_id/audit")
    else:
        fail("list_memories_core returns a memories payload", f"got {type(res).__name__}")
except Exception as exc:  # this is what regressed: "can't adapt type 'Query'"
    fail("list_memories_core(userId, limit) works", f"{type(exc).__name__}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. End-to-end pool flow
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPool flow: create → add member → write → list")

try:
    create_pool(CreatePoolRequest(pool_id=POOL, name="Test Pool", owner=OWNER))
    ok("create_pool")
except Exception as exc:
    fail("create_pool", f"{type(exc).__name__}: {exc}")

# create_pool adds the owner to pool_members as admin, so the owner can write.
try:
    w = write_pool_memory(POOL, PoolMemoryRequest(memberId=OWNER, content="owner can write"))
    if w.get("stored"):
        ok("create_pool auto-adds the owner as an admin member (owner can write)")
    else:
        fail("owner can write", str(w))
except HTTPException as exc:
    fail("owner can write", f"HTTP {exc.status_code}: {exc.detail}")

# A user who was never added has no membership row → access denied.
try:
    write_pool_memory(POOL, PoolMemoryRequest(memberId=OUTSIDER, content="should not be stored"))
    fail("non-member write is rejected", "write succeeded without membership")
except HTTPException as exc:
    if exc.status_code == 403:
        ok("non-member write is rejected with 403")
    else:
        fail("non-member write is rejected with 403", f"got {exc.status_code}")

try:
    m = add_member(POOL, AddMemberRequest(member_id=MEMBER, role="admin"))
    if m["can_write"]:
        ok("add_member(role=admin) grants write")
    else:
        fail("add_member(role=admin) grants write", f"can_write={m['can_write']}")
except Exception as exc:
    fail("add_member", f"{type(exc).__name__}: {exc}")

CONTENT = "The team deploys the marketing site on Vercel."
try:
    w = write_pool_memory(POOL, PoolMemoryRequest(memberId=MEMBER, content=CONTENT, importance=0.8))
    if w.get("stored") and w.get("id"):
        ok(f"write_pool_memory stored a memory (id={w['id']})")
    else:
        fail("write_pool_memory stored a memory", str(w))
except Exception as exc:
    fail("write_pool_memory", f"{type(exc).__name__}: {exc}")

# ── THE REGRESSION: this call used to raise "can't adapt type 'Query'" → 500 ──
try:
    listed = list_pool_memories(POOL, memberId=MEMBER, limit=50)
    contents = [m["content"] for m in listed["memories"]]
    if CONTENT in contents:
        ok("list_pool_memories returns the stored memory (regression: used to 500)")
    else:
        fail("list_pool_memories returns the stored memory", f"total={listed['total']}")
except Exception as exc:
    fail("list_pool_memories does not raise (regression)", f"{type(exc).__name__}: {exc}")

# Reads are gated too.
try:
    list_pool_memories(POOL, memberId=OUTSIDER, limit=50)
    fail("non-member read is rejected", "read succeeded without membership")
except HTTPException as exc:
    if exc.status_code == 403:
        ok("non-member read is rejected with 403")
    else:
        fail("non-member read is rejected with 403", f"got {exc.status_code}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("═" * 65)
passed = sum(1 for _, p in results if p)
total = len(results)
print(f"Results: {passed}/{total} passed")

failed = [(n, p) for n, p in results if not p]
if failed:
    print("\nFailed tests:")
    for name, _ in failed:
        print(f"  {FAIL}  {name}")
    sys.exit(1)
else:
    print(f"\n{PASS} All tests passed.")
    sys.exit(0)
