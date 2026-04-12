# YourMemory — Agent Memory Rules

Copy this file into your project's `CLAUDE.md`, system prompt, or agent instructions.
It tells your AI agent exactly when and how to use the YourMemory MCP tools.

---

## Memory Workflow (follow on every task)

### Step 1 — Recall before acting

At the start of **every task**, call `recall_memory` with keywords from the request:

```
recall_memory(query="<keywords from the request>", user_id="<your_name>")
```

Use the recalled context to answer without asking the user to repeat themselves.

---

### Step 2 — Decide: store, update, or ignore

Apply this policy to every fact that surfaces in the conversation:

| Case | Condition | Action |
|---|---|---|
| **New knowledge** | Fact is new — no existing memory covers it | `store_memory(content, importance)` |
| **Extension** | New fact adds detail to a recalled memory | `update_memory(memory_id, merged_sentence, importance)` |
| **Contradiction** | New fact directly conflicts with a recalled memory | `update_memory(memory_id, new_content, importance)` — replace only |
| **Ignore** | Question, conversational filler, or no lasting value | Do nothing |

**A fact has lasting value if it reveals:**
- A user preference, habit, constraint, or goal
- A project-level decision, architecture choice, or tech stack fact
- A failure mode or error that recurred (so it won't be forgotten)
- A strategy or approach that worked well

**Never store:**
- Questions the user asked
- Claude's own responses or opinions
- Temporary session state ("the user is currently on step 3")

---

### Step 3 — Persist immediately

Call the MCP tool right when you identify a storable fact. Do not batch or defer.

---

## Importance Scale

You **must always** supply `importance`. Never omit it.

| Value | When to use | Examples |
|---|---|---|
| `0.9 – 1.0` | Core identity, permanent facts | "User's name is Sachit", "User prefers Python", "Project uses PostgreSQL" |
| `0.7 – 0.8` | Strong recurring preferences, architectural decisions | "User never wants docstrings added", "Auth uses JWT with 1h expiry" |
| `0.5` | Regular project facts, one-time choices | "This sprint's goal is X", "Decided to use DuckDB for local dev" |
| `0.2 – 0.3` | Transient context, session-specific notes | "User is currently debugging the payment webhook" |

---

## Category

Pick the category that best matches the content. It controls how fast the memory decays.

| Category | Decay | Use for |
|---|---|---|
| `fact` | ~24 days | User preferences, identity, stable knowledge (default) |
| `strategy` | ~38 days (slowest) | Approaches that worked well — keep these longest |
| `assumption` | ~19 days | Inferred beliefs, uncertain context |
| `failure` | ~11 days (fastest) | What went wrong — stale errors should fade |

---

## Writing Memory Content

- **User facts:** `"Sachit prefers X"` / `"Sachit uses X at work"`
- **Project facts:** `"The project uses X"` / `"The API returns X"`
- **Failures:** `"OAuth failed for client X because of Y — fixed by Z"`
- **Strategies:** `"Pagination with cursor solved the timeout on large DB queries"`

One sentence per memory. Merge related facts rather than storing two overlapping entries.

---

## Quick Reference

```python
# Recall before every task
recall_memory(query="auth token expiry", user_id="sachit")

# Store a new fact
store_memory(
    content="Sachit uses JWT with 1-hour expiry for all internal APIs",
    importance=0.8,
    category="strategy",
    user_id="sachit"
)

# Update an existing memory (use the id from recall)
update_memory(
    memory_id=42,
    new_content="Sachit uses JWT with 2-hour expiry after the mobile app rollout",
    importance=0.8
)
```
