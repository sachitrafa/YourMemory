# YourMemory — Security & SOC 2 Compliance Posture

**Last full audit:** 2026-06-24 · **Audited version:** 1.4.65
**Scope:** Full source tree — `src/` (FastAPI service, MCP tools, services, DB, graph),
`memory_mcp.py`, `cloudflare-worker/src/index.ts`, hook templates.

This document is the durable record of YourMemory's security review. It maps the
codebase to the relevant SOC 2 Trust Service Criteria, records every vulnerability
found and remediated, and states the residual/accepted risks and the threat model
so the audit does not have to be re-derived from scratch each time.

---

## 1. Threat model (read this first)

YourMemory runs as a **local, single-user, loopback-bound** service by default:

- The FastAPI HTTP service binds `127.0.0.1:3033`. The MCP/SSE server binds
  `YOURMEMORY_HOST` (default `127.0.0.1`).
- The HTTP API is **intentionally unauthenticated** on loopback. The trust boundary
  is the local machine: any process running as the user already has equivalent
  access to the DuckDB file at `~/.yourmemory/`.
- Cross-origin browser attacks (CSRF/DNS-rebinding) against the loopback API are
  mitigated because every write endpoint requires `Content-Type: application/json`,
  which forces a CORS preflight the server never approves (no `Access-Control`
  headers are emitted by the FastAPI app).
- The **Cloudflare Worker** (`yourmemory-backend`) is the only internet-facing
  component. It handles email/OTP activation and an anonymous install counter.

**The single most important operational control:** never bind the service to a
non-loopback address (`0.0.0.0`) without putting an authenticating reverse proxy
in front of it. Doing so exposes all memory CRUD, `/buffer`, `/observe`,
`/retrieve`, and `/agents/register` to the network with no authentication. This is
by design for the local use case; it is documented here as the boundary condition.

---

## 2. SOC 2 Trust Service Criteria mapping

| TSC | Area | Status | Notes |
|-----|------|--------|-------|
| CC6.1 | Logical access — authentication | **Partial (by design)** | Local API is unauthenticated on loopback (see threat model). Agent API keys (`ym_…`) gate the `private` visibility partition and are SHA-256 hashed at rest, never stored in plaintext, shown once. |
| CC6.1 | Authorization — IDOR protection | **Pass** | All memory mutate paths (HTTP DELETE/PUT, MCP `update_memory`) enforce `user_id` ownership. Graph traversal filters by `user_id`. |
| CC6.6 | Boundary protection | **Pass (default)** | Loopback binding default; SSE host configurable and defaults to `127.0.0.1`. |
| CC6.7 | Data in transit | **Pass** | Worker is HTTPS-only (Cloudflare edge). Activation token sent in POST body, not URL. |
| CC6.8 | Malicious code / injection | **Pass** | All SQL parameterized. No `eval`/`exec`/`os.system` on untrusted input. XSS sinks escaped. |
| CC7.1 | Vulnerability detection | **Pass** | Repeat security reviews logged in §3. Pickle/SSRF/injection vectors analyzed and cleared. |
| CC7.2 | Secrets management | **Pass** | No hardcoded secrets in source. Worker secrets (`EMAILS_SECRET`, `RESEND_API_KEY`, `ANTHROPIC_API_KEY`) are Cloudflare env bindings. API keys hashed. |
| A1.2 | Availability — fail-open design | **Pass (intentional)** | LLM judge, graph indexing, and recall fail open so a downed dependency never blocks the user. Not a confidentiality risk. |
| C1.1 | Confidentiality — user isolation | **Pass** | Cross-user leakage paths (cold-start, graph BFS, recall) all filter by `user_id`. |

---

## 3. Remediation history (all verified in place at 1.4.65)

### v1.4.62
1. **Stored XSS** — `src/routes/ui.py`: agent_id rendered via inline `onclick`.
   → Replaced with `data-tab-id` + `addEventListener`; all dynamic values run
   through `escHtml()`.
2. **IDOR — DELETE** `/memories/{id}`: no ownership check.
   → `WHERE id = ? AND user_id = ?`.
3. **IDOR — PUT** `/memories/{id}`: no ownership check.
   → Added `userId` query param + 403 on `owner_id != userId`.
4. **Auth bypass** — `memory_mcp._check_registration`: `_token_verified` set before
   validity confirmed. → Moved inside the `data.get("valid")` branch.
5. **Network exposure** — MCP/SSE bound `0.0.0.0`.
   → `host = os.getenv("YOURMEMORY_HOST", "127.0.0.1")`.

### v1.4.63
6. **Auth bypass (regression risk)** — PUT `/memories`: the 403 `HTTPException` was
   swallowed by `except Exception: pass`, falling through to the un-scoped UPDATE.
   → Added `except HTTPException: raise` ahead of the generic handler.
7. **Stored XSS** — `src/routes/graph_viz.py`: memory `label`/`category` injected
   raw into `innerHTML`. → `escHtml()` on all values; error path uses `textContent`.
8. **Cross-user exposure** — `/graph/data` BFS + SQL had no `user_id` filter.
   → `AND user_id = ?` and post-fetch pruning of `nodes_to_include`; returns empty
   if the seed node is not owned by the requester.
9. **Stored XSS** — Worker `/emails`: `email`/`instance_id`/`created_at` raw in HTML.
   → `escHtml()` helper applied to all three.
10. **Token in URL** — `GET /verify-token?token=…` leaked the activation token to
    logs/Referer/history. → Changed to `POST /verify-token` with token in JSON body;
    `memory_mcp.py` client updated to match.

### v1.4.64
11. **Auth bypass** — `memory_mcp._check_registration` still used the old
    `GET ?token=` form, which now hits the worker's HTML catch-all, throws on
    `json.loads`, is swallowed, and silently passes verification forever.
    → Updated to POST with JSON body.
12. **IDOR** — MCP `update_memory` tool had no caller ownership check.
    → Added `user_id` to the tool schema and a `user_id_owner != caller` 403-equivalent.
13. **CSS-class injection** — `graph_viz.py`: `category` used as a CSS class name
    after HTML-escaping (which permits spaces). → `safeCat()` allowlist
    (`fact|assumption|failure|strategy`).

### v1.4.65
14. **OTP brute-force** — Worker `/verify-otp` had no per-code attempt cap. A
    6-digit code (1e6 space) could be brute-forced inside the 10-minute window
    since only `/send-otp` was rate-limited. → Added an `attempts` column; the
    most-recent code is burned after 5 wrong guesses (HTTP 429).

---

## 4. Residual / accepted risks

| ID | Risk | Severity | Decision |
|----|------|----------|----------|
| R1 | Local HTTP API unauthenticated on loopback | Low (local trust boundary) | **Accepted** — matches single-user design. Mitigation: keep bound to `127.0.0.1`. |
| R2 | `/agents/register` & `/agents/revoke` unauthenticated | Low (localhost-only) | **Accepted** — only meaningful if bound to a non-loopback host. API keys gate only the `private` partition. |
| R3 | Worker `/emails` & `/debug` admin auth via `?token=` query param compared with `!==` | Low | **Accepted** — single-admin, Cloudflare edge jitter defeats practical timing attacks; secret is a long random value. Future hardening: move to `Authorization` header + constant-time compare. |
| R4 | `pickle.load` on `~/.yourmemory/graph.pkl` | None (not network-reachable) | **Cleared** — file is local-only; exploiting it requires pre-existing local write access. |

---

## 5. Verified-clean vectors (analyzed, no action needed)

- **SQL injection** — every query uses `?`/`%s` placeholders. The one f-string
  (`graph_viz` IN-clause) interpolates only `int`s from graph traversal; user scope
  values in `retrieve._scope_fragment` are **bound**, not interpolated.
- **SSRF** — outbound hosts (`OLLAMA_URL`, OpenAI/Anthropic bases) come from env
  vars / hardcoded constants, never from request input.
- **Command injection / RCE** — no `os.system`, `subprocess` with shell, or `eval`
  on untrusted input anywhere in the request path.
- **CSRF/DNS-rebinding on loopback** — blocked by the JSON content-type preflight
  requirement (server emits no CORS allow headers).
- **Secrets in source** — none. Confirmed across Python and the worker.

---

## 6. How to re-run this audit

1. Re-read every file in §Scope; diff against this document's §3 to confirm fixes
   are still in place (grep anchors: `except HTTPException`, `YOURMEMORY_HOST`,
   `You do not own this memory`, `safeCat`, `POST /verify-token`, `MAX_OTP_ATTEMPTS`).
2. Search for new sinks: `innerHTML`, f-string SQL, `pickle.load`, `subprocess`,
   `eval`, `os.system`, new unauthenticated routes.
3. Append any new findings to §3 with the version that fixed them; update §4 if the
   risk acceptance changes.

> Reporting a vulnerability: email mishrasachit1@gmail.com.
