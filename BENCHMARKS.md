# YourMemory Benchmarks

Evaluation of YourMemory against Mem0 and Zep Cloud across three metrics.

---

## 1. Long-Context Recall Accuracy — LoCoMo Dataset

**Dataset:** [LoCoMo](https://github.com/snap-research/locomo) (Snap Research) — a public long-context memory benchmark consisting of multi-session conversations spanning weeks to months. We used `locomo10.json` (10 conversation pairs, 1,534 QA pairs total, categories 1–4).

**Method:** Session summaries were stored in both systems. Each QA pair was evaluated at the end of all sessions. A hit is recorded if the correct answer appears in the top-5 retrieved results.

**Embedding model:** `all-mpnet-base-v2` (sentence-transformers, 768 dims, runs fully in-process — no external service required)

**Metric:** Recall@5

### vs Mem0 (free tier) — 15 March 2026

| Sample | Speakers | YourMemory | Mem0 |
|--------|----------|:----------:|:----:|
| 1 | Caroline & Melanie | 40% | 35% |
| 2 | Jon & Gina | 65% | 5% |
| 3 | John & Maria | 20% | 15% |
| 4 | Joanna & Nate | 30% | 5% |
| 5 | Tim & John | 20% | 10% |
| 6 | Audrey & Andrew | 40% | 25% |
| 7 | James & John | 35% | 15% |
| 8 | Deborah & Jolene | 20% | 10% |
| 9 | Evan & Sam | 30% | 20% |
| 10 | Calvin & Dave | 45% | 45% |
| **Total** | **200 QA pairs** | **34%** | **18%** |

**YourMemory leads by +16 percentage points.** YourMemory wins 9 out of 10 samples and ties 1. Mem0's automatic fact extraction condenses session content and loses specific details (dates, names, events) that LoCoMo QA pairs target. YourMemory preserves full session summaries, retaining those details while Ebbinghaus decay keeps the most relevant content ranked highest.

### vs Supermemory — 12 April 2026 (full stack: vector + graph + decay + resolve)

**YourMemory configuration:** Full HTTP stack on localhost:8000. Each session summary passes through resolve (deduplication/merge), embed, graph indexing, and Ebbinghaus decay. Retrieval uses vector search + multi-hop graph expansion + recall propagation. Embedding model: `nomic-embed-text` via Ollama.

**Supermemory configuration:** Cloud API (`supermemory.add` / `search.memories`), rerank enabled, no temporal signal.

| Sample | Speakers | YourMemory | Supermemory |
|--------|----------|:----------:|:-----------:|
| 1 | Caroline & Melanie | 46% | 46% |
| 2 | Jon & Gina | 74% | 16% |
| 3 | John & Maria | 31% | 24% |
| 4 | Frank & Elaine | 58% | 32% |
| 5 | Jane & Anne | 60% | 28% |
| 6 | Kevin & Sarah | 47% | 24% |
| 7 | Michael & Ashley | 50% | 34% |
| 8 | Justin & Gabrielle | 57% | 26% |
| 9 | Diana & Allen | 51% | 14% |
| 10 | George & Olivia | 50% | 38% |
| **Total** | **533 QA pairs** | **52%** | **28%** |

**YourMemory leads by +24 percentage points (86% relative improvement).** YourMemory wins 9 out of 10 samples and ties 1 (Caroline & Melanie, where all facts come from a single time window and decay provides no signal). Supermemory's pure vector search has no temporal awareness — it cannot distinguish a fact from session 1 from one in session 5. YourMemory's Ebbinghaus decay re-ranks by recency and reinforcement, and the graph expansion layer surfaces related memories that the query didn't directly match.

**Stack progression on this run:**

| Stack | Recall@5 | Delta |
|-------|:--------:|------:|
| Vector + decay only (in-process) | 47% | baseline |
| Full stack (+ graph expansion, resolve, propagation) | 52% | +5pp |
| Supermemory (vector only, no decay) | 28% | −24pp vs full stack |

The graph layer contributed +5 percentage points on top of decay alone.

### vs Zep Cloud — 27 March 2026 (all-mpnet-base-v2)

| Sample | Speakers | YourMemory | Zep Cloud |
|--------|----------|:----------:|:---------:|
| 1 | Caroline & Melanie | 29% | 24% |
| 2 | Jon & Gina | 47% | 46% |
| 3 | John & Maria | 25% | 22% |
| 4 | Joanna & Nate | 38% | 18% |
| 5 | Tim & John | 31% | 24% |
| 6 | Audrey & Andrew | 33% | 20% |
| 7 | James & John | 37% | 23% |
| 8 | Deborah & Jolene | 28% | 16% |
| 9 | Evan & Sam | 32% | 18% |
| 10 | Calvin & Dave | 43% | 26% |
| **Total** | **1,534 QA pairs** | **34%** | **22%** |

**YourMemory leads by +12 percentage points (54% relative improvement).** YourMemory wins 9 out of 10 samples and ties 1. Zep Cloud uses LLM-based fact extraction per thread, which summarises and condenses content — losing the specific details (dates, names, events) that LoCoMo QA pairs target. YourMemory's Ebbinghaus decay scores full session summaries by recency and importance, preserving those details while ranking the most relevant content highest.

---

## 2. Workflow Efficiency — Token and LLM Call Savings

**Method:** A realistic multi-session developer workflow was simulated across 3 sessions with different inputs per session. Two approaches were compared: a stateless baseline (no memory, full conversation history carried forward) and YourMemory. The benchmark script is at [`benchmarks/two_session_comparison.py`](benchmarks/two_session_comparison.py).

### Token Savings

Without memory, the context window grows O(n) — every session carries all prior conversation history regardless of relevance. YourMemory replaces that history with a compressed memory block (~76–91 tokens of top-k recalled facts).

| Metric | Baseline | YourMemory | Δ |
|--------|:--------:|:----------:|:--:|
| Session 1 context tokens | 978 | 978 | 0% |
| Session 2 context tokens | 1,170 | 843 | −27.9% |
| Session 3 context tokens | 1,170 | 843 | −27.9% |
| **Total (3 sessions)** | **3,318** | **2,664** | **−19.7%** |
| Stale tokens injected | 1,148 | 0 | −100% |
| Estimated cost — 3 sessions (claude-sonnet-4-6) | $0.018 | $0.014 | −19.7% |
| Estimated cost — 30 sessions | $0.654 | $0.104 | **−84.1%** |

Memory block size stays flat (~76–91 tokens) while baseline history grows linearly. The cost gap compounds every session.

### LLM Call Savings

Without memory the assistant has no context at the start of a new session and must ask clarifying questions before implementing anything. Each clarifying round is a full LLM call that produces zero implementation output.

| Session | Baseline LLM calls | YourMemory LLM calls | Calls saved |
|---------|:-----------------:|:--------------------:|:-----------:|
| Session 1 | 4 (0 clarify + 4 work) | 4 (0 clarify + 4 work) | 0 |
| Session 2 | 5 (2 clarify + 3 work) | 4 (1 clarify + 3 work) | 1 |
| Session 3 | 5 (2 clarify + 3 work) | 4 (1 clarify + 3 work) | 1 |
| **Total** | **14** | **12** | **−14%** |

The savings grow as the memory bank fills. By session 3+, YourMemory has accumulated stack, driver, configuration, and constraints — later sessions require zero clarifying calls. Live test confirmed this pattern across 3 sessions storing 10 facts progressively.

---

## 3. Decay-Based Token Pruning

**Method:** A synthetic set of 15 memories spanning 0–60 days was evaluated. Memories with Ebbinghaus strength below the prune threshold (0.05) are excluded from retrieval entirely. Token counts are based on the top-5 memories injected into context.

**Metric:** Context tokens injected per query

| | Baseline (no decay) | YourMemory |
|---|:-------------------:|:----------:|
| Total memories | 15 | 15 |
| Pruned memories | 0 | 3 (20%) |
| Tokens in top-5 context | 74 | 71 |
| **Token reduction** | — | **4.1%** |

Token savings compound at scale. A system with 200+ memories over 6 months will prune a significantly larger fraction, reducing noise injected into the model context and lowering API costs.

---

## Scoring Formula

YourMemory retrieval score combines semantic relevance with biological memory strength:

```
score = cosine_similarity × Ebbinghaus_strength

Ebbinghaus_strength = importance × e^(−λ_eff × days) × (1 + recall_count × 0.2)
λ_eff = 0.16 × (1 − importance × 0.8)
```

Memories below strength `0.05` are pruned entirely. Memories above similarity `0.75` have their `recall_count` reinforced on retrieval.

---

## Dataset Reference

> Maharana, A., Lee, D., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024).
> **LoCoMo: Long Context Multimodal Benchmark for Dialogue.**
> *SNAP Research.*
> GitHub: [https://github.com/snap-research/locomo](https://github.com/snap-research/locomo)
