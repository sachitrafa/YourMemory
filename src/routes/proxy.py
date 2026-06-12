import os
import json
import getpass
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

from src.services.retrieve import retrieve as _retrieve

router = APIRouter(prefix="/proxy")

OPENAI_BASE    = "https://api.openai.com"
ANTHROPIC_BASE = "https://api.anthropic.com"


def _user_id(request: Request) -> str:
    return (
        request.headers.get("x-yourmemory-user")
        or os.getenv("YOURMEMORY_USER", "")
        or getpass.getuser()
    )


def _memory_block(user_id: str, query: str) -> str:
    try:
        result = _retrieve(user_id, query, top_k=5)
        mems = result.get("memories", [])
        if not mems:
            return ""
        lines = ["[Recalled from YourMemory]"]
        for m in mems:
            lines.append(f"- {m['content']}")
        return "\n".join(lines)
    except Exception:
        return ""


def _last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return str(c)
    return ""


def _execute_tool(name: str, args: dict, user_id: str) -> str:
    try:
        if name == "store_memory":
            from src.services.retrieve import retrieve as _r
            from src.db.connection import get_connection
            from src.services.embed import embed
            import numpy as np
            content    = args.get("content", "")
            importance = float(args.get("importance", 0.7))
            category   = args.get("category", "fact")
            if not content:
                return "error: content required"
            # Use the existing store logic via HTTP to local server to avoid circular imports
            import urllib.request as _ur
            payload = json.dumps({
                "userId": user_id,
                "content": content,
                "importance": importance,
                "category": category,
            }).encode()
            req = _ur.Request("http://localhost:3033/memories", data=payload,
                              headers={"Content-Type": "application/json"}, method="POST")
            with _ur.urlopen(req, timeout=5) as resp:
                return resp.read().decode()

        if name == "update_memory":
            memory_id  = args.get("memory_id") or args.get("id")
            new_content = args.get("new_content") or args.get("content", "")
            importance  = float(args.get("importance", 0.7))
            if not memory_id or not new_content:
                return "error: memory_id and new_content required"
            import urllib.request as _ur
            payload = json.dumps({
                "userId": user_id,
                "newContent": new_content,
                "importance": importance,
            }).encode()
            req = _ur.Request(f"http://localhost:3033/memories/{memory_id}", data=payload,
                              headers={"Content-Type": "application/json"}, method="PUT")
            with _ur.urlopen(req, timeout=5) as resp:
                return resp.read().decode()

    except Exception as e:
        return f"error: {e}"
    return "ok"


# ── Tool schemas injected into every request ───────────────────────────────────

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": "Store a new fact about the user or project for future sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content":    {"type": "string",  "description": "One sentence fact to remember."},
                    "importance": {"type": "number",  "description": "0.0–1.0. Use 0.9 for core identity, 0.7 for preferences, 0.5 for regular facts."},
                    "category":   {"type": "string",  "enum": ["fact", "strategy", "assumption", "failure"]},
                },
                "required": ["content", "importance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": "Update or correct an existing memory by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id":   {"type": "string", "description": "ID of the memory to update."},
                    "new_content": {"type": "string", "description": "Replacement content."},
                    "importance":  {"type": "number"},
                },
                "required": ["memory_id", "new_content", "importance"],
            },
        },
    },
]

ANTHROPIC_TOOLS = [
    {
        "name": "store_memory",
        "description": "Store a new fact about the user or project for future sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content":    {"type": "string"},
                "importance": {"type": "number"},
                "category":   {"type": "string", "enum": ["fact", "strategy", "assumption", "failure"]},
            },
            "required": ["content", "importance"],
        },
    },
    {
        "name": "update_memory",
        "description": "Update or correct an existing memory by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id":   {"type": "string"},
                "new_content": {"type": "string"},
                "importance":  {"type": "number"},
            },
            "required": ["memory_id", "new_content", "importance"],
        },
    },
]


# ── OpenAI-compatible proxy ────────────────────────────────────────────────────

@router.post("/openai/v1/chat/completions")
async def proxy_openai(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing OpenAI API key in Authorization header")

    body     = await request.json()
    uid      = _user_id(request)
    messages = body.get("messages", [])
    block    = _memory_block(uid, _last_user_text(messages))

    if block:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = block + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": block})
        body["messages"] = messages

    # Merge memory tools with any existing tools
    existing_tools = body.get("tools", [])
    body["tools"] = existing_tools + OPENAI_TOOLS
    if body.get("tool_choice") is None:
        body["tool_choice"] = "auto"

    headers = {"Authorization": auth, "Content-Type": "application/json"}
    stream  = body.get("stream", False)

    async with httpx.AsyncClient(timeout=120) as client:
        if stream:
            # For streaming: pass through — tool calls still appear in stream chunks
            # but we can't intercept mid-stream easily. Model stores via next turn.
            async def _stream():
                async with client.stream("POST", f"{OPENAI_BASE}/v1/chat/completions",
                                         headers=headers, json=body) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
            return StreamingResponse(_stream(), media_type="text/event-stream")

        # Non-streaming: intercept tool calls and execute them
        r    = await client.post(f"{OPENAI_BASE}/v1/chat/completions", headers=headers, json=body)
        data = r.json()

        choice = (data.get("choices") or [{}])[0]
        msg    = choice.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            # Execute each memory tool call
            messages.append(msg)
            for tc in tool_calls:
                fn   = tc.get("function", {})
                name = fn.get("name", "")
                args = json.loads(fn.get("arguments", "{}"))
                if name in ("store_memory", "update_memory"):
                    result = _execute_tool(name, args, uid)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

            # Second call to get the final response
            body["messages"] = messages
            body.pop("tools", None)
            body.pop("tool_choice", None)
            r2   = await client.post(f"{OPENAI_BASE}/v1/chat/completions", headers=headers, json=body)
            return JSONResponse(content=r2.json(), status_code=r2.status_code)

        return JSONResponse(content=data, status_code=r.status_code)


# ── Anthropic-compatible proxy ─────────────────────────────────────────────────

@router.post("/anthropic/v1/messages")
async def proxy_anthropic(request: Request):
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing Anthropic API key in x-api-key header")

    body     = await request.json()
    uid      = _user_id(request)
    messages = body.get("messages", [])
    block    = _memory_block(uid, _last_user_text(messages))

    if block:
        existing = body.get("system", "")
        body["system"] = block + ("\n\n" + existing if existing else "")

    body["tools"] = body.get("tools", []) + ANTHROPIC_TOOLS

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
        "content-type":      "application/json",
    }
    stream = body.get("stream", False)

    async with httpx.AsyncClient(timeout=120) as client:
        if stream:
            async def _stream():
                async with client.stream("POST", f"{ANTHROPIC_BASE}/v1/messages",
                                         headers=headers, json=body) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
            return StreamingResponse(_stream(), media_type="text/event-stream")

        r    = await client.post(f"{ANTHROPIC_BASE}/v1/messages", headers=headers, json=body)
        data = r.json()

        # Intercept tool use blocks
        content_blocks = data.get("content", [])
        tool_uses      = [b for b in content_blocks if b.get("type") == "tool_use"
                          and b.get("name") in ("store_memory", "update_memory")]

        if tool_uses:
            messages.append({"role": "assistant", "content": content_blocks})
            tool_results = []
            for tu in tool_uses:
                result = _execute_tool(tu["name"], tu.get("input", {}), uid)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tu["id"],
                    "content":     result,
                })
            messages.append({"role": "user", "content": tool_results})
            body["messages"] = messages
            body.pop("tools", None)
            r2   = await client.post(f"{ANTHROPIC_BASE}/v1/messages", headers=headers, json=body)
            return JSONResponse(content=r2.json(), status_code=r2.status_code)

        return JSONResponse(content=data, status_code=r.status_code)
