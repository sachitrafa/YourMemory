"""
Tests for hook self-refresh (src/services/hook_sync.py).

`yourmemory-setup` copies the hook templates into ~/.claude/hooks and registers
those copies. Upgrading the package therefore does NOT update the hooks that
actually run — a user on a stale Stop hook stores nothing, silently, forever.
sync_installed_hooks() closes that gap from the server, which always ships the
current templates.

Uses a temporary fake home — the real ~/.claude is never touched.

Run:
    python tests/test_hook_sync.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.hook_sync import sync_installed_hooks

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results = []


def ok(name):
    print(f"  {PASS}  {name}")
    results.append((name, True))


def fail(name, reason=""):
    print(f"  {FAIL}  {name}" + (f"  ← {reason}" if reason else ""))
    results.append((name, False))


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "hook_templates")
STORE_TEMPLATE = open(os.path.join(TEMPLATE_DIR, "yourmemory_store.py")).read()


def fake_home(with_hooks=True):
    home = tempfile.mkdtemp()
    if with_hooks:
        os.makedirs(os.path.join(home, ".claude", "hooks"))
    return home


def hook_path(home, fname="yourmemory_store.py"):
    return os.path.join(home, ".claude", "hooks", fname)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. A stale copy gets refreshed
# ═══════════════════════════════════════════════════════════════════════════════
print("\n1. Stale hooks are refreshed")

home = fake_home()
with open(hook_path(home), "w") as f:
    f.write("# ancient buggy hook\n")

updated = sync_installed_hooks(home)

if "yourmemory_store.py" in updated:
    ok("stale hook reported as updated")
else:
    fail("stale hook reported as updated", f"got {updated}")

if open(hook_path(home)).read() == STORE_TEMPLATE:
    ok("stale hook content replaced with the packaged template")
else:
    fail("stale hook content replaced with the packaged template")
shutil.rmtree(home)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. An up-to-date copy is left alone (idempotent, no needless writes)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n2. Current hooks are left alone")

home = fake_home()
with open(hook_path(home), "w") as f:
    f.write(STORE_TEMPLATE)
before_mtime = os.path.getmtime(hook_path(home))

updated = sync_installed_hooks(home)

if updated == []:
    ok("no updates reported when already current")
else:
    fail("no updates reported when already current", f"got {updated}")

if os.path.getmtime(hook_path(home)) == before_mtime:
    ok("file not rewritten when content matches")
else:
    fail("file not rewritten when content matches")

# Second run must also be a no-op.
if sync_installed_hooks(home) == []:
    ok("sync is idempotent")
else:
    fail("sync is idempotent")
shutil.rmtree(home)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. It never CREATES hooks a user doesn't have
# ═══════════════════════════════════════════════════════════════════════════════
print("\n3. Never creates hooks")

# A user who removed one hook on purpose must not get it back.
home = fake_home()
with open(hook_path(home), "w") as f:
    f.write("# stale\n")
sync_installed_hooks(home)

if not os.path.exists(hook_path(home, "yourmemory_observe.py")):
    ok("absent hook is not created")
else:
    fail("absent hook is not created", "sync installed a hook the user didn't have")
shutil.rmtree(home)

# A user who never ran setup has no hooks dir — sync must be a clean no-op.
home = fake_home(with_hooks=False)
updated = sync_installed_hooks(home)

if updated == []:
    ok("no hooks dir is a no-op")
else:
    fail("no hooks dir is a no-op", f"got {updated}")

if not os.path.exists(os.path.join(home, ".claude", "hooks")):
    ok("no hooks dir is not created")
else:
    fail("no hooks dir is not created")
shutil.rmtree(home)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Failures are survivable — the server must still start
# ═══════════════════════════════════════════════════════════════════════════════
print("\n4. Failures never raise")

home = fake_home()
with open(hook_path(home), "w") as f:
    f.write("# stale\n")
os.chmod(os.path.join(home, ".claude", "hooks"), 0o500)   # read-only dir

try:
    sync_installed_hooks(home)
    ok("read-only hooks dir does not raise")
except Exception as exc:
    fail("read-only hooks dir does not raise", repr(exc))
finally:
    os.chmod(os.path.join(home, ".claude", "hooks"), 0o700)
    shutil.rmtree(home)


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
