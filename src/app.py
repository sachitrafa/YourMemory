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
from src.routes import memories, retrieve, agents, ui, graph_viz, proxy
from src.jobs.decay_job import run as run_decay
from src.db.migrate import migrate


scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    migrate()
    scheduler.add_job(run_decay, "interval", hours=24, id="decay_job")
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
    from src.services.extract import categorize
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

    return {"stored": len(stored), "facts": stored}
