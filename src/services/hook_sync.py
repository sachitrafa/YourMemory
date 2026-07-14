"""Keep the installed Claude hooks in ~/.claude/hooks in sync with the package.

WHY THIS EXISTS
---------------
`yourmemory-setup` COPIES the hook templates into ~/.claude/hooks and registers
those copies in settings.json. So the copies — not the package — are what actually
run. A user who upgrades the package but never re-runs setup keeps the OLD hooks
forever. That is not cosmetic: the Stop hook shipped a transcript-parsing bug that
silently dropped the assistant's answer on every tool-using turn, so those users
would go on storing nothing, with no error and no way to notice.

WHY IN THE SERVER, NOT IN A HOOK
--------------------------------
A hook cannot reliably heal itself: the stale copy is the thing running, and it is
the thing lacking the fix. But every session launches the SERVER from the installed
package (SessionStart → yourmemory_server.py start), so the server always has the
current templates even when the hooks on disk are ancient. Refreshing from here
therefore repairs users who are one, or ten, versions behind.

WHAT IT WILL NOT DO
-------------------
- Never CREATES hooks: it only refreshes files that already exist, so a user who
  never ran setup (or who deliberately removed a hook) does not silently acquire one.
- Never touches settings.json. Registration is setup's job; this only refreshes the
  file contents behind paths that are already registered.
- Never raises. A sync failure must not stop the server from starting.

Sync is by CONTENT HASH rather than a version stamp — there is no runtime version
constant, and a hash additionally catches edits made in a dev checkout.
"""

import hashlib
import os

HOOK_FILES = (
    "yourmemory_recall.py",
    "yourmemory_server.py",
    "yourmemory_session_start.py",
    "yourmemory_store.py",
    "yourmemory_observe.py",
    "yourmemory_recall.sh",
    "yourmemory_user.sh",
)

EXECUTABLE = {"yourmemory_recall.sh", "yourmemory_user.sh"}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sync_installed_hooks(home: str | None = None) -> list[str]:
    """Refresh stale hook copies in ~/.claude/hooks from the packaged templates.

    Returns the names of the files that were updated (empty when already current).
    """
    home = home or os.path.expanduser("~")
    hooks_dir = os.path.join(home, ".claude", "hooks")
    if not os.path.isdir(hooks_dir):
        return []                      # setup never ran; nothing to keep in sync

    try:
        from importlib.resources import files as _ir_files
        tdir = _ir_files("src.hook_templates")
    except Exception:
        return []

    updated = []
    for fname in HOOK_FILES:
        dest = os.path.join(hooks_dir, fname)
        if not os.path.exists(dest):
            continue                   # never create — only refresh what setup installed

        try:
            template = (tdir / fname).read_text()
        except Exception:
            continue                   # template absent from this build; leave the copy alone

        try:
            with open(dest) as f:
                installed = f.read()
        except Exception:
            continue

        if _digest(installed) == _digest(template):
            continue

        try:
            with open(dest, "w") as f:
                f.write(template)
            if fname in EXECUTABLE:
                os.chmod(dest, 0o755)
            updated.append(fname)
        except Exception:
            continue                   # read-only dir / permissions — not fatal

    return updated
