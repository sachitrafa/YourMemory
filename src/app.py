import os
import sys
import json
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time
from collections import defaultdict, deque
from threading import Lock
from fastapi import Request
from fastapi.responses import JSONResponse
from src.routes import memories, retrieve, agents, ui, graph_viz, proxy, audit, dsar, compact
from src.jobs.decay_job import run as run_decay
from src.services.audit import prune_expired as prune_audit
from src.db.migrate import migrate


scheduler = AsyncIOScheduler()


def _daily_jobs():
    run_decay()
    try:
        prune_audit()   # retention: drop audit rows older than the (>=90-day) window
    except Exception:
        pass
    # Opt-in: compress clusters of related memories so the store stays lean over time.
    if os.getenv("YOURMEMORY_COMPACTION", "0") == "1":
        try:
            from src.services.compaction import compact_user
            from src.db.connection import get_conn, get_backend
            b = get_backend(); conn = get_conn()
            try:
                if b == "duckdb":
                    users = [r[0] for r in conn.execute("SELECT DISTINCT user_id FROM memories").fetchall()]
                else:
                    cur = conn.cursor(); cur.execute("SELECT DISTINCT user_id FROM memories")
                    users = [r[0] for r in cur.fetchall()]; cur.close()
            finally:
                conn.close()
            for u in users:
                try: compact_user(u)
                except Exception: pass
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    migrate()
    scheduler.add_job(_daily_jobs, "interval", hours=24, id="decay_job")
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="YourMemory", version="0.1.0", lifespan=lifespan)

app.include_router(memories.router)
app.include_router(retrieve.router)
app.include_router(agents.router)
app.include_router(ui.router)
app.include_router(graph_viz.router)
app.include_router(proxy.router)
app.include_router(audit.router)
app.include_router(dsar.router)
app.include_router(compact.router)


# ── Rate limiting (abuse prevention) ────────────────────────────────────────────
# Per-client sliding-window limiter. Loopback (127.0.0.1) is exempt by default — the
# local trust boundary is the OS user, and the recall/store hooks burst legitimately;
# the limit targets network-exposed (hosted) clients. Configure with:
#   YOURMEMORY_RATE_LIMIT   max requests per window per client  (default 300; 0 disables)
#   YOURMEMORY_RATE_WINDOW  window length in seconds            (default 60)
#   YOURMEMORY_RATE_LIMIT_LOOPBACK=1  also limit loopback clients
_RATE_LIMIT  = int(os.getenv("YOURMEMORY_RATE_LIMIT", "300"))
_RATE_WINDOW = int(os.getenv("YOURMEMORY_RATE_WINDOW", "60"))
_RATE_LOOPBACK = os.getenv("YOURMEMORY_RATE_LIMIT_LOOPBACK", "0") == "1"
_rate_hits: dict = defaultdict(deque)
_rate_lock = Lock()
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    if _RATE_LIMIT > 0 and request.url.path != "/health":
        client = request.client.host if request.client else "unknown"
        if _RATE_LOOPBACK or client not in _LOOPBACK:
            now = time.time()
            with _rate_lock:
                dq = _rate_hits[client]
                cutoff = now - _RATE_WINDOW
                while dq and dq[0] < cutoff:
                    dq.popleft()
                if len(dq) >= _RATE_LIMIT:
                    retry = max(1, int(dq[0] + _RATE_WINDOW - now))
                    return JSONResponse(
                        {"error": "rate limit exceeded", "retry_after": retry},
                        status_code=429, headers={"Retry-After": str(retry)})
                dq.append(now)
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Verbatim conversation buffer (opt-in headless lean-window mode) ──────────
# Keeps the last N exchanges raw (no qwen distillation) so a flushed context window
# can be reconstructed losslessly. Pairs with /auto-store: buffer = recent verbatim,
# memories = distilled long-term. Recall injects the buffer ALWAYS + gated facts.
class BufferStoreRequest(BaseModel):
    user_id: str
    user_text: str = ""
    assistant_text: str = ""
    keep: int = 3


@app.post("/buffer-store")
def buffer_store(req: BufferStoreRequest):
    from src.db.connection import get_conn, get_backend
    backend = get_backend()
    conn = get_conn()
    keep = max(1, min(int(req.keep or 3), 20))
    ut, at = (req.user_text or "")[:4000], (req.assistant_text or "")[:8000]
    if not (ut or at):
        return {"ok": False, "error": "empty exchange"}
    try:
        if backend == "postgres":
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO conversation_buffer (user_id, user_text, assistant_text) VALUES (%s, %s, %s)",
                (req.user_id, ut, at))
            cur.execute(
                "DELETE FROM conversation_buffer WHERE user_id=%s AND id NOT IN "
                "(SELECT id FROM conversation_buffer WHERE user_id=%s ORDER BY id DESC LIMIT %s)",
                (req.user_id, req.user_id, keep))
            conn.commit()
            cur.close()
        else:  # duckdb / sqlite share the ?-param dialect
            conn.execute(
                "INSERT INTO conversation_buffer (user_id, user_text, assistant_text) VALUES (?, ?, ?)",
                [req.user_id, ut, at])
            conn.execute(
                "DELETE FROM conversation_buffer WHERE user_id=? AND id NOT IN "
                "(SELECT id FROM conversation_buffer WHERE user_id=? ORDER BY id DESC LIMIT ?)",
                [req.user_id, req.user_id, keep])
            if backend == "sqlite":
                conn.commit()
        return {"ok": True, "kept": keep}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.get("/buffer")
def buffer_get(userId: str, n: int = 3):
    """Return the last n verbatim exchanges (oldest → newest) for injection."""
    from src.db.connection import get_conn, get_backend
    backend = get_backend()
    conn = get_conn()
    n = max(1, min(int(n or 3), 20))
    try:
        if backend == "postgres":
            cur = conn.cursor()
            cur.execute(
                "SELECT user_text, assistant_text FROM conversation_buffer "
                "WHERE user_id=%s ORDER BY id DESC LIMIT %s", (userId, n))
            rows = cur.fetchall()
            cur.close()
        else:
            rows = conn.execute(
                "SELECT user_text, assistant_text FROM conversation_buffer "
                "WHERE user_id=? ORDER BY id DESC LIMIT ?", [userId, n]).fetchall()
        exchanges = [{"user_text": r[0] or "", "assistant_text": r[1] or ""} for r in reversed(rows)]
        return {"buffer": exchanges, "count": len(exchanges)}
    except Exception as e:
        return {"buffer": [], "count": 0, "error": str(e)[:200]}
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Observation capture (knowledge from work output) ────────────────────────
# Unlike /auto-store (conversational facts about the user), /observe extracts technical/
# factual statements from WORK OUTPUT — file contents, command output, documents — so an
# agent can recall them later instead of re-reading the source. This is what gives recall
# the work-detail. Stores each fact via the proven single-fact path (/memories).
class ObserveRequest(BaseModel):
    text:    str
    user_id: str | None = None
    source:  str | None = None


@app.post("/observe")
def observe_endpoint(req: ObserveRequest):
    import getpass
    import urllib.request as _ur
    OLLAMA_URL   = os.getenv("YOURMEMORY_OLLAMA_URL",   "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("YOURMEMORY_OLLAMA_MODEL", "qwen2.5:7b")
    user_id = req.user_id or os.getenv("YOURMEMORY_USER", "") or getpass.getuser()
    text = (req.text or "")[:8000]
    if len(text) < 200:
        return {"stored": 0, "facts": []}
    src = f" (source: {req.source})" if req.source else ""

    prompt = (
        "You build an agent's KNOWLEDGE memory from work output (code, file contents, command "
        "output, documents, logs). Extract atomic FACTUAL statements an agent would want to recall "
        "later INSTEAD OF re-reading the source.\n"
        "Capture: what a file/function/component does; key parameters, values, thresholds, names; "
        "structure and relationships between parts; configuration; results/outputs; decisions "
        "encoded in the work.\n"
        "Rules:\n- ONE atomic fact per item.\n- SELF-CONTAINED: keep exact names, numbers, and "
        "file/function names verbatim so it stands alone.\n- One plain declarative sentence, no markdown.\n"
        "- Skip imports, license headers, boilerplate and trivia.\n- If nothing durable, return an empty list.\n\n"
        "Return JSON: {\"facts\":[{\"fact\":\"<sentence>\",\"importance\":\"HIGH|MED|LOW\","
        "\"category\":\"fact|assumption|failure|strategy\"}]}\n\n"
        f"WORK OUTPUT{src}:\n{text}"
    )
    schema = {
        "type": "object",
        "properties": {"facts": {"type": "array", "items": {"type": "object", "properties": {
            "fact":       {"type": "string"},
            "importance": {"type": "string", "enum": ["HIGH", "MED", "LOW"]},
            "category":   {"type": "string", "enum": ["fact", "assumption", "failure", "strategy"]},
        }, "required": ["fact", "importance", "category"]}}},
        "required": ["facts"],
    }
    EXTRACT_BACKEND = os.getenv("YOURMEMORY_EXTRACT_BACKEND", "ollama").lower()

    if EXTRACT_BACKEND == "anthropic":
        from src.services.extract import extract_facts_anthropic
        try:
            facts = extract_facts_anthropic(prompt)
        except Exception as exc:
            return {"stored": 0, "error": str(exc)[:200]}
    else:
        payload = json.dumps({
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": schema,
            "keep_alive": os.getenv("YOURMEMORY_OLLAMA_KEEPALIVE", "30m"),
            "options": {"temperature": 0, "num_predict": 700},
        }).encode()
        try:
            rq = _ur.Request(f"{OLLAMA_URL}/api/generate", data=payload,
                             headers={"Content-Type": "application/json"})
            with _ur.urlopen(rq, timeout=90) as r:
                raw = json.loads(r.read()).get("response", "").strip()
            facts = json.loads(raw).get("facts", [])
        except Exception as exc:
            return {"stored": 0, "error": str(exc)[:200]}

    if not facts:
        return {"stored": 0, "facts": []}

    from src.services.extract import is_question as _is_question, should_store_llm as _judge

    IMP = {"HIGH": 0.8, "MED": 0.6, "LOW": 0.4}
    stored = 0
    ids = []
    for it in facts:
        if not isinstance(it, dict):
            continue
        c = str(it.get("fact", "")).strip()
        if len(c) < 12:
            continue
        # Same mandatory relevance gate as /auto-store: work-output capture tends to
        # surface transient status notes ("/retrieve returned 6 hits") that aren't durable.
        if _is_question(c) or not _judge(c):
            continue
        imp = IMP.get(str(it.get("importance", "MED")).upper(), 0.6)
        cat = str(it.get("category", "fact")).strip().lower()
        if cat not in ("fact", "assumption", "failure", "strategy"):
            cat = "fact"
        try:
            body = json.dumps({"userId": user_id, "content": c,
                               "importance": imp, "category": cat}).encode()
            rq = _ur.Request("http://localhost:3033/memories", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
            with _ur.urlopen(rq, timeout=10) as r:
                mid = json.loads(r.read()).get("id")
            if mid is not None:
                ids.append(mid)
            stored += 1
        except Exception:
            pass

    # Phase 3 — structural co-occurrence edges: facts from the SAME source form a
    # connected region in the graph, so recalling one surfaces the whole region.
    if len(ids) > 1:
        try:
            from src.graph.graph_store import link_memories
            link_memories(ids, relation="co_occurrence", weight=0.55)
        except Exception:
            pass

    return {"stored": stored, "facts": [it.get("fact") for it in facts]}


class AskRequest(BaseModel):
    query: str
    user_id: str | None = None
    top_k: int = 3


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    import getpass
    from src.services.retrieve import retrieve as _retrieve

    OLLAMA_URL      = os.getenv("YOURMEMORY_OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL    = os.getenv("YOURMEMORY_OLLAMA_MODEL", "llama3.2:3b")
    MIN_SCORE       = 0.55   # direct cosine+BM25 matches (raised to cut false positives)
    MIN_GRAPH_SCORE = 0.20   # graph-expanded nodes (capped at 0.6×0.74≈0.444)

    user_id = req.user_id or os.getenv("YOURMEMORY_USER", "") or getpass.getuser()

    results  = _retrieve(user_id, req.query, top_k=req.top_k)
    memories = results.get("memories", [])

    direct = [m for m in memories if not m.get("via_graph")]
    if not direct or direct[0].get("score", 0) < MIN_SCORE:
        return {"answer": "Not enough memory context to answer without Claude.", "grounded": False}

    # Keyword grounding: if the query contains a capitalised or quoted term that
    # appears in none of the retrieved memories, the match is topically adjacent
    # but not specifically relevant (e.g. Fly.io memory answering a Netlify query).
    import re as _re
    query_terms = set(_re.findall(r'[A-Z][a-z]{2,}|[a-z]{4,}', req.query))
    combined_memory_text = " ".join(m["content"] for m in direct).lower()
    # Only apply the check when query has specific tech terms (≥4 chars, not common words)
    _STOP = {"what", "does", "does", "have", "this", "that", "with", "from",
             "best", "good", "using", "used", "many", "should", "their", "will"}
    specific_terms = [t for t in query_terms if t.lower() not in _STOP and len(t) >= 4]
    if specific_terms and not any(_re.search(r'\b' + _re.escape(t.lower()) + r'\b', combined_memory_text) for t in specific_terms):
        return {"answer": "Not enough memory context to answer without Claude.", "grounded": False}

    memory_lines = "\n".join(
        f"{i+1}. {m['content']}"
        for i, m in enumerate(memories)
        if m.get("score", 0) >= (MIN_GRAPH_SCORE if m.get("via_graph") else MIN_SCORE)
    )

    prompt = f"""You are a memory assistant. Answer ONLY using the provided memories below.
Be concise and direct. If the answer is not clearly supported by the memories, say exactly: "I don't know — ask Claude."

Memories:
{memory_lines}

Question: {req.query}
Answer:"""

    def stream_ollama():
        payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}).encode()
        try:
            ollama_req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(ollama_req, timeout=30) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            yield f"Ollama error: {exc}"

    return StreamingResponse(stream_ollama(), media_type="text/plain")


class AutoStoreRequest(BaseModel):
    user_text: str
    assistant_text: str
    user_id: str | None = None


@app.post("/auto-store")
def auto_store_endpoint(req: AutoStoreRequest):
    """Extract and store memorable facts from a conversation exchange using the local LLM."""
    import getpass
    from src.services.extract import categorize, is_question, should_store_llm
    from src.services.embed import embed
    from src.services.resolve import resolve
    from src.db.connection import get_conn, get_backend, emb_to_db

    OLLAMA_URL   = os.getenv("YOURMEMORY_OLLAMA_URL",   "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("YOURMEMORY_OLLAMA_MODEL", "qwen2.5:7b")

    user_id = req.user_id or os.getenv("YOURMEMORY_USER", "") or getpass.getuser()

    # Generous truncation — the pricing and the actual deliverable usually live in
    # the assistant's longer output, which a tight 1000-char cut used to discard.
    user_text = req.user_text[:2000]
    asst_text = req.assistant_text[:4000]

    prompt = (
        "You build an AI agent's long-term memory. From the exchange below, extract every "
        "durable, reusable fact about the user and their world — anything that would help the "
        "agent serve them better in a FUTURE, possibly unrelated conversation.\n\n"
        "Capture knowledge of ANY kind. For example:\n"
        "- Identity & background — name, role, job, company, location, age, languages, expertise\n"
        "- Preferences & style — likes/dislikes, habits, tools, brands, how they want to be communicated with\n"
        "- Goals & plans — what they want, are working toward, deadlines, upcoming events (with dates)\n"
        "- Relationships — people, family, pets, colleagues, clients they mention, and who each one is\n"
        "- Work & projects — what they're building, their stack/tools, decisions, constraints, requirements\n"
        "- Possessions & environment — what they own or use; their setup (devices, OS, home, car, pets)\n"
        "- Events & experiences — things that happened, are happening, or will happen — keep the timing\n"
        "- Asserted knowledge — facts, findings, numbers, or results they state as true\n"
        "- Problems & failures — what went wrong, blockers, mistakes, pitfalls to avoid\n"
        "- Solutions & strategies — approaches, fixes, or tactics that worked\n"
        "- Standing instructions — rules for how the agent should behave for this user\n\n"
        "Rules for each fact:\n"
        "- ONE atomic fact per item; split compound statements.\n"
        "- SELF-CONTAINED — resolve pronouns and keep the actual names, numbers, dates, places, and "
        "entities verbatim, so it stands alone, out of context, weeks later "
        "(\"Aylin's daughter starts at NYU in September\", not \"her kid starts soon\").\n"
        "- ANCHOR TIME — turn relative time into concrete terms where possible "
        "(\"running a marathon on March 3\", \"broke his ankle, out ~6 weeks\").\n"
        "- Capture facts about the USER and meaningful facts about OTHER people/things discussed; "
        "note who owns or did what — don't blur the speakers.\n"
        "- One plain declarative sentence. No markdown, no preamble.\n\n"
        "Do NOT store: greetings, small talk, acknowledgements, filler; questions, or the assistant's "
        "own suggestions/answers (unless the user adopted them as a decision); one-off throwaway actions "
        "with no future relevance; anything you'd have to guess — only what is stated or clearly implied.\n\n"
        "Tag each fact:\n"
        "- importance: HIGH = identity, key relationships, strong preferences, decisions, deadlines, "
        "critical facts; MED = useful context, ordinary preferences, projects, findings; LOW = minor/background.\n"
        "- category: 'fact' = a stated/stable detail or preference; 'assumption' = inferred, not confirmed; "
        "'failure' = something that went wrong / to avoid; 'strategy' = an approach that worked.\n\n"
        "If nothing durable is worth remembering, return an empty list.\n\n"
        "Return JSON: {\"facts\":[{\"fact\":\"<sentence>\",\"importance\":\"HIGH|MED|LOW\","
        "\"category\":\"fact|assumption|failure|strategy\"}]}\n\n"
        "Example —\n"
        "User: Finally moved to Berlin for the new job at Zalando — I'm a backend engineer there now. "
        "Been learning Rust on weekends. My cat Mochi hated the flight. Also remind me: my partner "
        "Aylin's birthday is March 12.\n"
        "Assistant: Congrats on the move!\n"
        "{\"facts\":["
        "{\"fact\":\"The user moved to Berlin for a new job.\",\"importance\":\"HIGH\",\"category\":\"fact\"},"
        "{\"fact\":\"The user works as a backend engineer at Zalando.\",\"importance\":\"HIGH\",\"category\":\"fact\"},"
        "{\"fact\":\"The user is learning Rust on weekends.\",\"importance\":\"MED\",\"category\":\"fact\"},"
        "{\"fact\":\"The user has a cat named Mochi.\",\"importance\":\"MED\",\"category\":\"fact\"},"
        "{\"fact\":\"The user's partner Aylin has a birthday on March 12.\",\"importance\":\"HIGH\",\"category\":\"fact\"}"
        "]}\n\n"
        f"User: {user_text}\n\nAssistant: {asst_text}"
    )

    schema = {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact":       {"type": "string"},
                        "importance": {"type": "string", "enum": ["HIGH", "MED", "LOW"]},
                        "category":   {"type": "string", "enum": ["fact", "assumption", "failure", "strategy"]},
                    },
                    "required": ["fact", "importance", "category"],
                },
            }
        },
        "required": ["facts"],
    }

    payload = json.dumps({
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        # Schema-constrained output: the model returns valid structured JSON, which
        # removes the fragile free-text line parsing (markdown/tag/preamble leaks).
        "format":  schema,
        # keep_alive holds the model resident so the next extraction doesn't pay
        # a cold-load (a 7B model evicted from RAM can exceed the request timeout).
        "keep_alive": os.getenv("YOURMEMORY_OLLAMA_KEEPALIVE", "30m"),
        "options": {"temperature": 0, "num_predict": 600},
    }).encode()

    if os.getenv("YOURMEMORY_EXTRACT_BACKEND", "ollama").lower() == "anthropic":
        from src.services.extract import extract_facts_anthropic
        try:
            facts_list = extract_facts_anthropic(prompt)
        except Exception as exc:
            return {"stored": 0, "error": str(exc)}
    else:
        try:
            ollama_req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(ollama_req, timeout=60) as resp:
                raw_response = json.loads(resp.read()).get("response", "").strip()
        except Exception as exc:
            return {"stored": 0, "error": str(exc)}
        try:
            facts_list = json.loads(raw_response).get("facts", [])
        except Exception:
            return {"stored": 0, "facts": [], "error": "non-JSON extraction response"}

    if not facts_list:
        return {"stored": 0, "facts": []}

    backend = get_backend()
    conn    = get_conn()

    cur = conn.cursor() if backend == "postgres" else None

    IMPORTANCE_MAP = {"HIGH": 0.85, "MED": 0.65, "LOW": 0.45}
    VALID_CATS     = {"fact", "assumption", "failure", "strategy"}
    # Structured output removes the markdown/preamble/tag noise the old free-text
    # parser had to scrub, so only a light guard against empties/stray fragments remains.
    BAD_PREFIXES = ("subject:", "dear ", "regards", "to:", "cc:")

    stored   = []
    to_index = []   # (memory_id, content, importance, category, embedding) — indexed AFTER conn closes
    for item in facts_list:
        if not isinstance(item, dict):
            continue
        fact       = str(item.get("fact", "")).strip().strip('"').strip()
        importance = IMPORTANCE_MAP.get(str(item.get("importance", "MED")).upper(), 0.65)
        category   = str(item.get("category", "")).strip().lower()
        if category not in VALID_CATS:
            category = categorize(fact)        # fall back to heuristic if the model omits/garbles it
        if len(fact) < 12 or len(fact.split()) < 2:
            continue
        if fact.lower().startswith(BAD_PREFIXES):
            continue
        # Drop verbatim questions cheaply, then run the mandatory LLM relevance judge —
        # it gates out meta-observations ("the user asked about X") and ephemeral filler
        # that the extractor leaks. Runs in the fire-and-forget worker, so no turn latency;
        # fails open (stores) only if Ollama is unreachable.
        if is_question(fact):
            continue
        if not should_store_llm(fact):
            continue
        try:
            embedding  = embed(fact)
            resolution = resolve(user_id, fact, embedding, conn)
            action     = resolution["action"]
            existing   = resolution.get("existing")

            if action == "new":
                emb_str  = emb_to_db(embedding, backend)
                if backend == "postgres":
                    cur.execute(
                        "INSERT INTO memories (user_id, content, embedding, importance, category) "
                        "VALUES (%s, %s, %s::vector, %s, %s) "
                        "ON CONFLICT (user_id, content) DO UPDATE "
                        "SET recall_count = memories.recall_count + 1, last_accessed_at = NOW() "
                        "RETURNING id",
                        (user_id, fact, emb_str, importance, category),
                    )
                    row = cur.fetchone()
                    if row:
                        to_index.append((row[0], fact, importance, category, embedding))
                elif backend == "duckdb":
                    conn.execute(
                        "INSERT INTO memories (user_id, content, embedding, importance, category) VALUES (?, ?, ?, ?, ?)",
                        [user_id, fact, emb_str, importance, category],
                    )
                    row = conn.execute(
                        "SELECT id FROM memories WHERE user_id = ? AND content = ?",
                        [user_id, fact],
                    ).fetchone()
                    if row:
                        to_index.append((row[0], fact, importance, category, embedding))
                stored.append(fact)

            elif action in ("replace", "merge") and existing:
                new_content = resolution["content"]
                new_emb     = embed(new_content)
                emb_str     = emb_to_db(new_emb, backend)
                new_cat     = category   # carry the model-assigned category onto the merged memory
                if backend == "postgres":
                    cur.execute(
                        "UPDATE memories SET content=%s, embedding=%s::vector, category=%s, "
                        "recall_count = recall_count + 1, last_accessed_at = NOW() WHERE id=%s",
                        (new_content, emb_str, new_cat, existing["id"]),
                    )
                    to_index.append((existing["id"], new_content, importance, new_cat, new_emb))
                elif backend == "duckdb":
                    conn.execute(
                        "UPDATE memories SET content=?, embedding=? WHERE id=?",
                        [new_content, emb_str, existing["id"]],
                    )
                    to_index.append((existing["id"], new_content, importance, new_cat, new_emb))
                stored.append(new_content)

            elif action == "reinforce" and existing:
                if backend == "postgres":
                    cur.execute(
                        "UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW() WHERE id=%s",
                        (existing["id"],),
                    )
                elif backend == "duckdb":
                    conn.execute(
                        "UPDATE memories SET recall_count = recall_count + 1 WHERE id=?",
                        [existing["id"]],
                    )

            # Postgres needs an explicit commit; commit per-item so one bad fact
            # (which aborts the transaction) never loses the facts before it.
            if backend == "postgres":
                conn.commit()

        except Exception:
            if backend == "postgres":
                try:
                    conn.rollback()
                except Exception:
                    pass
            continue

    # Commit + close the memories connection BEFORE graph indexing — index_memory
    # opens its own connection, and DuckDB is single-writer, so overlapping writes
    # would deadlock.
    try:
        if backend == "postgres":
            conn.commit()
            if cur:
                cur.close()
        conn.close()
    except Exception:
        pass

    # Add stored memories to the graph so entity edges form. Without this the hook
    # path populates no graph, and graph expansion surfaces nothing. Best-effort.
    if to_index:
        try:
            from src.graph.graph_store import index_memory
            for mem_id, content, imp, category, emb in to_index:
                try:
                    index_memory(memory_id=mem_id, user_id=user_id, content=content,
                                 strength=imp, importance=imp, category=category,
                                 embedding=list(emb))
                except Exception as _ge:
                    print(f"[graph] auto-store index_memory failed: {_ge}", file=sys.stderr)
        except Exception as _ie:
            print(f"[graph] auto-store graph import failed: {_ie}", file=sys.stderr)

    # Audit one clickable event per stored/updated memory (target_id = memory id),
    # so each write in the trail links straight to its memory. Falls back to a single
    # batch event if no ids were captured (e.g. only reinforcements).
    try:
        from src.services.audit import log_event
        if to_index:
            for mem_id, content, _imp, cat, _emb in to_index:
                log_event("write", "store", user_id, target_id=mem_id,
                          detail={"category": cat, "len": len(content or "")})
        elif stored:
            log_event("write", "auto_store", user_id, detail={"count": len(stored)})
    except Exception:
        pass

    return {"stored": len(stored), "facts": stored}
