# YourMemory

![YourMemory Demo](demo.gif)

Persistent, decaying memory for Claude — backed by PostgreSQL + pgvector.

Memories fade like real ones do. Frequently recalled memories stay strong. Forgotten ones are pruned automatically. Claude decides what to remember and how important it is.

> Still early — ideating on where to take this next. Feedback welcome.

---

## How it works

### Ebbinghaus Forgetting Curve

```
effective_λ = 0.16 × (1 - importance × 0.8)
strength    = importance × e^(-effective_λ × days) × (1 + recall_count × 0.2)
```

Importance controls both the starting value and how fast a memory decays:

| importance | effective λ | survives (never recalled) |
|------------|-------------|--------------------------|
| 1.0        | 0.032       | ~94 days                 |
| 0.9        | 0.045       | ~64 days                 |
| 0.5        | 0.096       | ~24 days                 |
| 0.2        | 0.134       | ~10 days                 |

Memories recalled frequently gain `recall_count` boosts that counteract decay.

### Retrieval scoring

```
score = cosine_similarity × Ebbinghaus_strength
```

Results rank by how relevant and how fresh a memory is — not just one or the other.

---

## Setup

**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/) — runs Postgres
- [Ollama](https://ollama.com) — runs local embeddings

**Install and start:**

```bash
git clone https://github.com/sachitrafa/cognitive-ai-memory
cd cognitive-ai-memory
./setup.sh
```

The script pulls the embedding model, installs the Python package, creates your `.env`, and starts Postgres. DB migration and the decay scheduler run automatically on first boot.

**Start the MCP server:**

```bash
yourmemory
```

**Wire into Claude (`~/.claude/settings.json`):**

```json
{
  "mcpServers": {
    "yourmemory": {
      "command": "yourmemory"
    }
  }
}
```

Reload Claude Code (`Cmd+Shift+P` → `Developer: Reload Window`).

**Add memory instructions to your project:**

Copy `sample_CLAUDE.md` into your project root as `CLAUDE.md` and replace the two placeholders:
- `YOUR_NAME` — your name (e.g. `Alice`)
- `YOUR_USER_ID` — used to namespace memories (e.g. `alice`)

Claude will now follow the recall → store → update workflow automatically on every task.

---

## MCP Tools

Claude gets three tools:

| Tool | When to call |
|------|-------------|
| `recall_memory` | Start of every task — surface relevant context |
| `store_memory` | After learning a new preference, fact, or instruction |
| `update_memory` | When a recalled memory is outdated or needs merging |

### Example session

```
User: "I prefer tabs over spaces in all my Python projects"

Claude:
  → recall_memory("tabs spaces Python preferences")   # nothing found
  → store_memory("Sachit prefers tabs over spaces in Python", importance=0.9)

Next session:
  → recall_memory("Python formatting")
  ← {"content": "Sachit prefers tabs over spaces in Python", "strength": 0.87}
  → Claude now knows without being told again
```

---

## Decay Job

The decay job runs automatically every 24 hours on startup — no cron needed. Memories that decay below strength `0.05` are pruned automatically.

---

## REST API

### `POST /memories` — store a memory

```bash
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"userId":"u1","content":"Prefers dark mode","importance":0.8}'
```

### `POST /retrieve` — semantic search

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"userId":"u1","query":"UI preferences"}'
```

### `GET /memories` — list all memories

```bash
curl "http://localhost:8000/memories?userId=u1"
```

### `PUT /memories/{id}` — update a memory

```bash
curl -X PUT http://localhost:8000/memories/42 \
  -H "Content-Type: application/json" \
  -d '{"content":"Prefers dark mode in all apps","importance":0.85}'
```

### `DELETE /memories/{id}` — remove a memory

```bash
curl -X DELETE http://localhost:8000/memories/42
```

---

## Stack

- **PostgreSQL + pgvector** — vector similarity search + relational in one DB
- **Ollama** — local embeddings (`nomic-embed-text`, 768 dims) + classification (`llama3.2:3b`)
- **spaCy** — question detection, fact/assumption categorization
- **FastAPI** — REST server
- **APScheduler** — automatic decay job (runs every 24h on startup)
- **MCP** — Claude integration via Model Context Protocol

---

## Architecture

```
Claude Code
    │
    ├── recall_memory(query)
    │       └── embed(query) → cosine similarity → score = sim × strength → top-k
    │
    ├── store_memory(content, importance)
    │       └── is_question? → reject
    │           categorize() → fact | assumption
    │           embed() → INSERT memories
    │
    └── update_memory(id, new_content, importance)
            └── embed(new_content) → UPDATE memories SET content, embedding, importance

REST API (FastAPI)
    ├── POST   /memories         — store
    ├── PUT    /memories/{id}    — update
    ├── DELETE /memories/{id}    — delete
    ├── GET    /memories         — list all (with live strength)
    └── POST   /retrieve         — semantic search

PostgreSQL (pgvector)
    └── memories table
        ├── embedding vector(768)    — cosine similarity
        ├── importance float         — user/LLM-assigned base weight
        ├── recall_count int         — reinforcement counter
        └── last_accessed_at         — for Ebbinghaus decay
```
