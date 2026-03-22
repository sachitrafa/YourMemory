import os
import glob

# Fallback entry used when no agents/ dir exists or agent_id is "user"
_USER_ENTRY = {
    "agent_id": "user",
    "visibility_default": "shared",
    "can_read": [],        # empty = no restriction
    "can_write": ["shared", "private"],
    "description": "Human user",
}

_registry: dict = {}
_loaded = False


def _parse_md(path: str) -> dict | None:
    """Parse YAML frontmatter from an agent MD file."""
    with open(path) as f:
        lines = f.readlines()

    if not lines or lines[0].strip() != "---":
        return None

    frontmatter = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            # Parse lists like [a, b, c]
            if value.startswith("[") and value.endswith("]"):
                value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
            frontmatter[key] = value

    return frontmatter if "agent_id" in frontmatter else None


def load_registry(agents_dir: str = None) -> dict:
    global _registry, _loaded

    if agents_dir is None:
        agents_dir = os.path.join(os.path.dirname(__file__), "..", "..", "agents")

    agents_dir = os.path.abspath(agents_dir)
    _registry = {"user": _USER_ENTRY}

    if not os.path.isdir(agents_dir):
        _loaded = True
        return _registry

    for path in glob.glob(os.path.join(agents_dir, "*.md")):
        config = _parse_md(path)
        if config:
            _registry[config["agent_id"]] = config

    _loaded = True
    return _registry


def get_registry() -> dict:
    if not _loaded:
        load_registry()
    return _registry


def get_agent(agent_id: str) -> dict | None:
    return get_registry().get(agent_id)


def is_registered(agent_id: str) -> bool:
    return agent_id in get_registry()


def can_write_visibility(agent_id: str, visibility: str) -> bool:
    agent = get_agent(agent_id)
    if not agent:
        return False
    allowed = agent.get("can_write", ["shared", "private"])
    return visibility in allowed


def can_read_from(agent_id: str, stored_by: str) -> bool:
    """Check if agent_id is allowed to read memories stored by stored_by."""
    agent = get_agent(agent_id)
    if not agent:
        return False
    can_read = agent.get("can_read", [])
    # Empty list = no restriction (reads all shared)
    if not can_read:
        return True
    return stored_by in can_read


def default_visibility(agent_id: str) -> str:
    agent = get_agent(agent_id)
    if not agent:
        return "shared"
    return agent.get("visibility_default", "shared")
