"""
Access-control enforcement tests (src/services/auth.py).

Proves the two postures:
  * YOURMEMORY_AUTH=off       — backward compatible: no key works, client-supplied
                               identity is honoured (the pre-existing behaviour).
  * YOURMEMORY_AUTH=required  — every request needs a valid ym_ key; identity is
                               taken from the key, so a caller can NOT read another
                               user's memory or write to a pool they lack access to.

Isolated DuckDB — never touches the real store.

Run:
    python tests/test_auth.py
"""
import os
import sys

# ── Isolated test DB (blank DATABASE_URL FIRST so .env's postgres is ignored) ──
TEST_DB = "/tmp/yourmemory_test_auth.duckdb"
if os.path.exists(TEST_DB):
    os.remove(TEST_DB)
os.environ["DATABASE_URL"] = ""
os.environ["YOURMEMORY_DB"] = TEST_DB
os.environ["YOURMEMORY_AUTH"] = "off"          # start in the default posture

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.migrate import migrate
migrate()

from fastapi.testclient import TestClient

from src.app import app
from src.services.api_keys import register_agent
from src.routes.memories import add_memory, MemoryRequest
from src.routes.pools import create_pool, add_member, CreatePoolRequest, AddMemberRequest

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = []


def ok(name):
    print(f"  {PASS}  {name}")
    results.append((name, True))


def fail(name, reason=""):
    print(f"  {FAIL}  {name}" + (f"  ← {reason}" if reason else ""))
    results.append((name, False))


def set_auth(mode):
    os.environ["YOURMEMORY_AUTH"] = mode


client = TestClient(app)

# ── Seed: two users each with a key, one shared pool where alice is read-only ──
alice_key = register_agent("alice-agent", "alice")["api_key"]
bob_key = register_agent("bob-agent", "bob")["api_key"]

add_memory(MemoryRequest(userId="alice", content="Alice deploys the API on Railway.", importance=0.8))
add_memory(MemoryRequest(userId="bob", content="Bob ships the frontend on Vercel.", importance=0.8))

create_pool(CreatePoolRequest(pool_id="team", name="Team", owner="owner"))
add_member("team", AddMemberRequest(member_id="alice", role="reader"))   # read-only


# ═══════════════════════════════════════════════════════════════════════════════
# Mode OFF — backward compatible
# ═══════════════════════════════════════════════════════════════════════════════
print("\nYOURMEMORY_AUTH=off  (backward compatibility)")
set_auth("off")

r = client.get("/memories", params={"userId": "alice", "audit": "false"})
if r.status_code == 200 and any("Railway" in m["content"] for m in r.json()["memories"]):
    ok("no key + off: GET /memories?userId=alice works (self-declared id honoured)")
else:
    fail("off: GET /memories works without a key", f"{r.status_code} {r.text[:80]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mode REQUIRED — real enforcement
# ═══════════════════════════════════════════════════════════════════════════════
print("\nYOURMEMORY_AUTH=required  (enforced)")
set_auth("required")

# 1. no key → 401
r = client.get("/memories", params={"userId": "alice", "audit": "false"})
if r.status_code == 401:
    ok("no key is rejected with 401")
else:
    fail("no key is rejected with 401", f"got {r.status_code}")

# 2. invalid key → 401
r = client.get("/memories", params={"userId": "alice", "audit": "false"},
               headers={"Authorization": "Bearer ym_not_a_real_key"})
if r.status_code == 401:
    ok("invalid key is rejected with 401")
else:
    fail("invalid key is rejected with 401", f"got {r.status_code}")

# 3. THE CORE GUARANTEE: alice's key cannot read bob's memory even if she claims userId=bob
r = client.get("/memories", params={"userId": "bob", "audit": "false"},
               headers={"Authorization": f"Bearer {alice_key}"})
contents = " ".join(m["content"] for m in r.json().get("memories", [])) if r.status_code == 200 else ""
if r.status_code == 200 and "Railway" in contents and "Vercel" not in contents:
    ok("alice's key + userId=bob returns ALICE's memory, not bob's (identity locked)")
else:
    fail("cross-user read is blocked", f"{r.status_code} contents={contents[:80]!r}")

# 4. pool write denied — alice is only a reader of 'team'
r = client.post("/pools/team/memories",
                json={"memberId": "alice", "content": "should be blocked", "importance": 0.5},
                headers={"Authorization": f"Bearer {alice_key}"})
if r.status_code == 403:
    ok("pool write denied for a read-only member (403)")
else:
    fail("pool write denied for read-only member", f"got {r.status_code}")

# 5. non-member pool write denied — bob is not in 'team' at all
r = client.post("/pools/team/memories",
                json={"memberId": "bob", "content": "outsider", "importance": 0.5},
                headers={"Authorization": f"Bearer {bob_key}"})
if r.status_code == 403:
    ok("pool write denied for a non-member (403)")
else:
    fail("pool write denied for non-member", f"got {r.status_code}")

# 6. grant write → now alice can write to the pool
add_member("team", AddMemberRequest(member_id="alice", role="contributor"))
r = client.post("/pools/team/memories",
                json={"memberId": "alice", "content": "Team uses trunk-based development.", "importance": 0.6},
                headers={"Authorization": f"Bearer {alice_key}"})
if r.status_code == 200 and r.json().get("stored"):
    ok("pool write succeeds once the member has write access")
else:
    fail("pool write succeeds with write access", f"{r.status_code} {r.text[:80]}")

# 7. a caller CANNOT spoof another member's id on a pool write (bob's key, claims alice)
r = client.post("/pools/team/memories",
                json={"memberId": "alice", "content": "spoofed", "importance": 0.5},
                headers={"Authorization": f"Bearer {bob_key}"})
if r.status_code == 403:
    ok("cannot spoof another member's id (bob's key + memberId=alice → 403)")
else:
    fail("cannot spoof another member's id", f"got {r.status_code}")


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
