<!-- mcp-name: io.github.sachitrafa/yourmemory -->
<div align="center">
<img src="logo.svg.png" alt="YourMemory" width="110" /><br>
<h1>YourMemory</h1>

**Persistent memory for AI agents — built on the science of how humans remember.**

[![PyPI](https://img.shields.io/pypi/v/yourmemory?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/yourmemory/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/yourmemory?color=brightgreen)](https://pypi.org/project/yourmemory/)
[![Python](https://img.shields.io/pypi/pyversions/yourmemory)](https://pypi.org/project/yourmemory/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![GitHub Stars](https://img.shields.io/github/stars/sachitrafa/YourMemory?style=social)](https://github.com/sachitrafa/YourMemory)
[![GitHub Issues](https://img.shields.io/github/issues/sachitrafa/YourMemory)](https://github.com/sachitrafa/YourMemory/issues)
[![Last Commit](https://img.shields.io/github/last-commit/sachitrafa/YourMemory)](https://github.com/sachitrafa/YourMemory/commits/main)
[![Docker Build](https://img.shields.io/github/actions/workflow/status/sachitrafa/YourMemory/docker-publish.yml?branch=main&label=docker&logo=docker)](https://github.com/sachitrafa/YourMemory/actions/workflows/docker-publish.yml)

[![LoCoMo Recall@5](https://img.shields.io/badge/LoCoMo%20Recall%405-59%25-brightgreen)](BENCHMARKS.md)
[![LongMemEval Recall@5](https://img.shields.io/badge/LongMemEval%20Recall%405-89%25-brightgreen)](BENCHMARKS.md)
[![HotpotQA BOTH@5](https://img.shields.io/badge/HotpotQA%20BOTH%405-71.5%25-brightgreen)](BENCHMARKS.md)
[![oosmetrics](https://api.oosmetrics.com/api/v1/badge/achievement/9106de02-3dae-41ff-bc28-109da93fe87d.svg)](https://oosmetrics.com/repo/sachitrafa/YourMemory)

</div>

---

## What Is YourMemory?

Every session, your AI assistant starts from zero. It asks the same questions, forgets your preferences, re-learns your stack. **There is no memory between conversations.**

YourMemory fixes that with a one-command install that plugs into Claude, Cursor, Cline, Windsurf, or any MCP client. It gives your AI a persistent memory layer modelled on human cognition:

- **Things that matter stick** — importance score controls how quickly a memory decays
- **Outdated facts get replaced** — subject-aware deduplication merges or supersedes memories automatically
- **Related context surfaces together** — entity graph links memories that share people, places, or concepts
- **Old memories fade naturally** — Ebbinghaus forgetting curve prunes stale context every 24 hours

Zero infrastructure required. SQLite by default, Postgres for teams.

---

## Table of Contents

- [Benchmarks](#benchmarks)
- [Quick Start](#quick-start)
- [Memory Dashboard](#memory-dashboard)
- [Ask Without an LLM Call](#ask-without-calling-the-api)
- [MCP Tools](#mcp-tools)
- [How It Works](#how-it-works)
- [Multi-Agent Memory](#multi-agent-memory)
- [Stack](#stack)
- [Architecture](#architecture)
- [Contributing](#contributing)

---

## Benchmarks

Three external datasets, all scripts open source and reproducible. Full methodology in [BENCHMARKS.md](BENCHMARKS.md).

### LongMemEval-S — 500 questions, ~53 distractor sessions each

The hardest standard benchmark for long-term memory systems. Each question is backed by ~53 conversation sessions; the model must retrieve the right one(s) from the haystack.

| Metric | Score |
|--------|:-----:|
| **Recall@5** (any gold session in top-5) | **89.4%** |
| Recall-all@5 (all gold sessions in top-5) | 84.8% |
| nDCG@5 (ranking quality) | 87.4% |

**By question type (Recall@5):**

| Question Type | Recall@5 | n |
|---------------|:--------:|:-:|
| single-session-assistant | 98.2% | 56 |
| knowledge-update | 96.2% | 78 |
| multi-session | 95.5% | 133 |
| single-session-preference | 90.0% | 30 |
| temporal-reasoning | 84.2% | 133 |
| single-session-user | 72.9% | 70 |

### LoCoMo-10 — 1,534 QA pairs across 10 multi-session conversations

Conversations spanning weeks to months. Every system ingests the same session summaries in the same order.

| System | Recall@5 | 95% CI |
|--------|:--------:|:------:|
| **YourMemory** (BM25 + vector + graph + decay) | **59%** | 56–61% |
| Zep Cloud | 28% | 26–30% |
| Supermemory | 31%* | 28–33% |
| Mem0 | 18%* | 16–20% |

> **2× better recall than Zep Cloud across all 10 samples.** \* Supermemory and Mem0 exhausted free-tier quotas mid-benchmark; scores computed over full 1,534 pairs using 0 for unfinished samples.

### HotpotQA — 200 multi-hop questions requiring two facts from different articles

| System | BOTH_FOUND@5 |
|--------|:------------:|
| **YourMemory** (vector + BM25 + entity graph) | **71.5%** |
| YourMemory (no entity edges) | 59.5% |

Entity graph edges add **+12 pp** — they traverse from Fact 1 to Fact 2 even when Fact 2 has low embedding similarity to the query.

*Writeup: [I built memory decay for AI agents using the Ebbinghaus forgetting curve](https://dev.to/sachit_mishra_686a94d1bb5/i-built-memory-decay-for-ai-agents-using-the-ebbinghaus-forgetting-curve-1b0e)*

---

## Quick Start

**Supports Python 3.11–3.14. No Docker, no database setup. All memory stored locally in `~/.yourmemory/`.**

### Before you install — what this does

| Behavior | Detail |
|---|---|
| **Activation** | Requires a free one-time token. Visit [yourmemoryai.xyz](https://yourmemoryai.xyz/), enter your email, verify with a 6-digit code, and copy your token. Token is verified against `val.run` once on first run. |
| **Global rule injection** | `yourmemory-setup` writes memory instructions into `~/.cursor/rules/memory.mdc` and other detected AI client config files (Claude, VS Code, etc.) so the assistant can call memory tools automatically. You can remove these files at any time. |
| **MCP tool behavior** | The `recall_memory` tool can be called by your AI assistant when persistent context would help. The assistant decides when to call it based on the request. |
| **Telemetry** | A UUID (no personal data) is sent to `val.run` on first setup only. Opt out: `YOURMEMORY_TELEMETRY=off` |

**Activation steps:**
1. Visit [yourmemoryai.xyz](https://yourmemoryai.xyz/) and enter your email
2. Check your inbox for a 6-digit verification code
3. Enter the code on the website — your token is shown instantly
4. Run the three commands below:

```bash
pip install yourmemory
yourmemory-register <your-token>
yourmemory-setup
```

---

## Memory Dashboard

Two built-in browser UIs — no extra setup, start automatically with the MCP server.

### Memory Browser — `http://localhost:3033/ui`

A full read/write view of everything stored in memory.

| What you see | Details |
|---|---|
| **Stats bar** | Total · Strong ≥50% · Fading 5–50% · Near prune <10% |
| **Agent tabs** | All / User / per-agent views |
| **Memory cards** | Content · strength bar · category · recall count · last accessed |
| **Filters** | Category (fact / strategy / assumption / failure) · Sort by strength, recency, recall |

Pass `?user=<id>` to pre-load a specific user: `http://localhost:3033/ui?user=sachit`

### Graph Visualiser — `http://localhost:3033/graph`

An interactive force-directed map of how memories connect.

```
http://localhost:3033/graph?memoryId=42&userId=sachit&depth=2
```

- Root memory as a larger cyan node; neighbours color-coded by category
- Edge thickness = connection strength
- Click any node for full content; drag, zoom, reposition freely

---

## Ask Without Calling the API

The only memory system that can answer questions **without making any LLM API call.**

```bash
yourmemory ask "what database does this project use"
# → YourMemory uses DuckDB locally and Postgres in production.

yourmemory ask "what port does the dashboard run on"
# → 3033

yourmemory ask "how do I fix a kubernetes deployment"
# → Not enough memory context to answer without Claude.
```

When memory is strong enough, it answers instantly — zero tokens, zero cloud cost, zero latency. When it isn't, it declines cleanly rather than hallucinating.

| Query | Mem0 / Zep / LangMem | YourMemory |
|---|---|---|
| "What port does the server run on?" | Full LLM API call | Instant, $0 |
| "What database does this project use?" | Full LLM API call | Instant, $0 |
| "How do I fix a k8s deployment?" | Full LLM API call | Declines → Claude |
| Privacy | Query sent to cloud | Never leaves your machine |

---

## MCP Tools

Three tools, called by your AI automatically.

| Tool | When your AI calls it | What it does |
|------|-----------------------|--------------|
| `recall_memory(query, current_path?)` | Start of every task | Surfaces memories ranked by similarity × decay strength; spatial boost for path-matched memories |
| `store_memory(content, importance, category?, context_paths?)` | After learning something new | Embeds, deduplicates, stores with decay; tags optional file/dir paths |
| `update_memory(id, new_content, importance)` | When a stored fact is outdated | Re-embeds and replaces; logs old content to audit trail |

```python
# Store with spatial context
store_memory(
    "Sachit prefers tabs over spaces in Python",
    importance=0.9,
    category="fact",
    context_paths=["/projects/backend"]
)

# Next session — spatial boost fires when working in that directory
recall_memory("Python formatting", current_path="/projects/backend")
# → {"content": "Sachit prefers tabs over spaces in Python", "strength": 0.87}
```

### Memory categories control decay rate

| Category | Half-life | Best for |
|----------|-----------|----------|
| `strategy` | ~38 days | Patterns that worked, architectural decisions |
| `fact` | ~24 days | Preferences, identity, stable knowledge |
| `assumption` | ~19 days | Inferred context, uncertain beliefs |
| `failure` | ~11 days | Errors, wrong approaches, environment-specific issues |

---

## How It Works

### Ebbinghaus Forgetting Curve

Memory strength decays exponentially. Importance and recall frequency slow that decay:

```
effective_λ  = base_λ × (1 − importance × 0.8)
strength     = clamp(importance × e^(−effective_λ × active_days) × (1 + recall_count × 0.2), 0, 1)
hybrid_score = 0.4 × bm25_norm + 0.6 × cosine_similarity
```

`active_days` counts only days the user was active — vacations don't cause memory loss. Memories below strength `0.05` are pruned automatically every 24 hours.

**Session wrap-up:** recalled memory IDs are tracked per session. When a session goes idle (30 min default), those memories get a `recall_count` boost. Set `YOURMEMORY_SESSION_IDLE` to change the window.

**Recall throttling:** identical (user, query) pairs are cached within a configurable window. Set `YOURMEMORY_RECALL_COOLDOWN` (seconds, default 0 = off).

### Hybrid Retrieval: Vector + BM25 + Entity Graph

Retrieval runs in two rounds:

**Round 1 — Hybrid search:** cosine similarity + BM25 keyword scoring, returns top-k candidates above threshold.

**Round 2 — Graph expansion:** BFS traversal from Round 1 seeds surfaces memories that share context but not vocabulary — connected via semantic or entity edges.

```
recall("Python backend")
  Round 1 → [1] Python/MongoDB    (sim=0.61)
             [2] DuckDB/spaCy     (sim=0.19)
  Round 2 → [5] Docker/Kubernetes (sim=0.29 — below cut-off, surfaced via shared entity "backend")
```

**Chain-aware pruning:** a decayed memory is kept alive if any graph neighbour is above the prune threshold. Related memories age together.

### Subject-Aware Deduplication

Before storing, YourMemory checks whether the new memory is about the same entity as the nearest existing one:

```
"Sachit uses DuckDB"      vs  "YourMemory uses DuckDB"
 subject: Sachit               subject: YourMemory
 → different entities → stored separately ✓

"YourMemory uses DuckDB"  vs  "YourMemory stores data in DuckDB"
 subject: YourMemory           subject: YourMemory
 → same entity → merged ✓
```

Subject comparison embeds the first two tokens of each sentence — no hardcoded word lists, generalises to any language.

---

## Multi-Agent Memory

Multiple agents can share one YourMemory instance — each with isolated private memories and controlled access to shared context.

```python
from src.services.api_keys import register_agent

result = register_agent(
    agent_id="coding-agent",
    user_id="sachit",
    can_read=["shared", "private"],
    can_write=["shared", "private"],
)
# → result["api_key"]  — ym_xxxx (shown once only)
```

```python
# Agent stores a private failure memory
store_memory(
    "Staging uses self-signed cert — skip SSL verify",
    importance=0.7, category="failure",
    api_key="ym_xxxx", visibility="private"
)

# Recalls shared + its own private memories; other agents see shared only
recall_memory("staging SSL", api_key="ym_xxxx")
```

---

## Stack

| Component | Role |
|-----------|------|
| **DuckDB** | Default vector DB — zero setup, native cosine similarity |
| **NetworkX** | Default graph backend — persists at `~/.yourmemory/graph.pkl` |
| **sentence-transformers** | Local embeddings (`multi-qa-mpnet-base-dot-v1`, 768 dims) |
| **spaCy** | Local NLP for deduplication and entity extraction |
| **APScheduler** | Automatic 24h decay and pruning job |
| **PostgreSQL + pgvector** | Optional — for teams or large datasets |
| **Neo4j** | Optional graph backend |


---

## Architecture

```
Claude / Cline / Cursor / Any MCP client
    │
    ├── recall_memory(query, current_path?, api_key?)
    │       └── throttle check → embed → hybrid search (Round 1)
    │               → graph BFS expansion (Round 2)
    │               → score = sim × strength
    │               → spatial boost (+0.08) if current_path matches context_paths
    │               → temporal boost (+0.25) if query has time window expression
    │               → session tracking → recall_count bump on session end
    │
    ├── store_memory(content, importance, category?, context_paths?, api_key?)
    │       └── question? → reject
    │               subject-aware dedup → same entity? merge/reinforce : new
    │               embed() → INSERT → index_memory() → graph node + edges
    │               record_activity(user_id) → active days log
    │
    └── update_memory(id, new_content, importance)
            └── log old content → memory_history (audit trail)
                    embed(new_content) → UPDATE → refresh graph node

  Vector DB (Round 1)              Graph DB (Round 2)
  DuckDB (default)                 NetworkX (default)
    memories.duckdb                  graph.pkl
    ├── embedding FLOAT[768]         ├── nodes: memory_id, strength
    ├── importance FLOAT             └── edges: sim × verb_weight ≥ 0.4
    ├── recall_count INTEGER
    ├── context_paths JSON         Neo4j (opt-in)
    ├── created_at TIMESTAMP         └── bolt://localhost:7687
    ├── visibility VARCHAR
    ├── agent_id VARCHAR
    user_activity  (active days log)
    memory_history (supersession audit)
```

---

## Contributing

PRs are welcome. See [CONTRIBUTORS.md](CONTRIBUTORS.md) for contributors who have already improved YourMemory.

---

## Dataset References

- [LoCoMo](https://github.com/snap-research/locomo) — Maharana et al. (2024). *LoCoMo: Long Context Multimodal Benchmark for Dialogue.* Snap Research.
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — Wu et al. (2024). *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.*
- [HotpotQA](https://hotpotqa.github.io/) — Yang et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.*

---

## License

Copyright 2026 **Sachit Misra** — Licensed under [CC-BY-NC-4.0](LICENSE).

**Free for:** personal use, education, academic research, open-source projects.
**Not permitted:** commercial use without a separate written agreement.

Commercial licensing: mishrasachit1@gmail.com
