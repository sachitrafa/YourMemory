# YourMemory — SOC 2 Readiness Report

**Report type:** Internal readiness self-assessment (NOT a SOC 2 attestation)
**Prepared:** 2026-06-28 · **Tool version assessed:** 1.4.73
**Prepared by:** Automated codebase assessment (Claude Code)
**Scope:** `src/` (FastAPI service, MCP tools, services, DB, graph), `memory_mcp.py`,
`cloudflare-worker/src/index.ts`, hook templates.

> **What this is / isn't.** This is a control-by-control readiness assessment derived
> from the source code and verified behaviour. It is the evidence document to walk into
> a free third-party readiness tool (e.g. Comp AI, SecurityWall, Cyberday) or a CPA
> auditor with. It is **not** a SOC 2 report — only a licensed CPA firm can issue a
> Type I / Type II attestation, and only against a registered legal entity.

---

## 1. Executive summary

| Dimension | Readiness |
|---|---|
| **Technical / product controls** | **Strong** — most Common Criteria technical controls implemented and verified |
| **Organizational / process controls** | **Early** — policies, access governance, vendor management largely undocumented |
| **Blocking prerequisite** | **No legal entity** — required before any attestation |
| **Overall** | **Audit-ready on the technical layer; not yet entity/policy-ready** |

**Headline strengths:** enforced per-user authorization (no IDOR), parameterized SQL,
escaped XSS sinks, hashed API keys, no hardcoded secrets, and a tamper-evident,
append-only **audit trail with 90-day retention**.

**Headline gaps:** no legal entity, no written security policies, no formal access
reviews / MFA / onboarding-offboarding, no monitoring & incident-response process.

---

## 2. Trust Service Criteria — control assessment

Legend: ✅ implemented & verified · ◑ partial / by-design · ❌ not yet in place

### CC1 — Control environment (governance)
| Control | Status | Notes |
|---|---|---|
| Legal entity / org structure | ❌ | No registered entity yet — **blocks attestation** |
| Security policies (InfoSec, AUP) | ❌ | Not documented |
| Defined roles & responsibilities | ❌ | Informal |
| Background checks / HR security | ❌ | N/A pre-entity |

### CC2 — Communication & information
| Control | Status | Notes |
|---|---|---|
| Security commitments published | ◑ | `SECURITY.md` posture doc exists; vuln-report email present |
| Internal control documentation | ◑ | `SECURITY.md` + this report; no policy set |

### CC5 / CC6 — Logical & physical access
| Control | Status | Notes |
|---|---|---|
| Authorization / least privilege (IDOR) | ✅ | Ownership enforced on all mutate paths (HTTP DELETE/PUT, MCP `update_memory`); graph + recall filter by `user_id` |
| Authentication | ◑ | Local API unauthenticated **by design** (loopback trust boundary, documented). Agent API keys gate the `private` partition, SHA-256 hashed, shown once. Hosted/multi-tenant mode needs real auth + RBAC |
| Boundary protection | ✅ | Loopback (`127.0.0.1`) default; SSE host configurable; CSRF blocked by JSON content-type preflight |
| MFA, access reviews, onboarding/offboarding | ❌ | Not in place |
| Secrets management | ✅ | No hardcoded secrets; worker secrets are platform env bindings; API keys hashed |

### CC6.7 — Data in transit / at rest
| Control | Status | Notes |
|---|---|---|
| Encryption in transit | ✅ | Cloudflare Worker HTTPS-only; activation token in POST body (not URL) |
| Encryption at rest | ◑ | Relies on host disk encryption; local DuckDB unencrypted by design (local trust boundary) |

### CC7 — System operations (monitoring, change, audit logging)
| Control | Status | Notes |
|---|---|---|
| **Audit logging** | ✅ | **Append-only, hash-chained `audit_log`**: every read/write/delete with timestamp, user+agent id, action, target. Tamper-evident (`/audit/verify`), **90-day minimum retention**, daily prune |
| Injection / malicious-code defense | ✅ | Parameterized SQL throughout; XSS sinks escaped; no `eval`/shell on untrusted input |
| Vulnerability management | ✅ | Repeat security reviews (14 findings remediated v1.4.62–65); documented in `SECURITY.md` |
| Monitoring & alerting | ❌ | No centralized logging/alerting/SIEM |
| Incident response process | ❌ | No documented IR plan |

### CC8 — Change management
| Control | Status | Notes |
|---|---|---|
| Version control & history | ✅ | Git; every change committed with rationale |
| Release process | ◑ | Versioned PyPI releases; no formal change-approval / CI gate documented |

### A1 — Availability
| Control | Status | Notes |
|---|---|---|
| Fail-open resilience | ✅ (intentional) | LLM judge, graph indexing, recall fail open so a downed dependency never blocks the user |
| Backup / DR plan | ❌ | No documented backup/restore or DR runbook |

### C1 — Confidentiality
| Control | Status | Notes |
|---|---|---|
| User data isolation | ✅ | Cross-user leakage paths (cold-start, graph BFS, recall) all filter by `user_id` |
| Audit-log privacy | ✅ | Logs ids + metadata only — never raw memory content |

### P / PI — Privacy & processing integrity (if in scope)
| Control | Status | Notes |
|---|---|---|
| Relevance gate on stored data | ✅ | Mandatory LLM judge gates every auto-stored fact (v1.4.72) |
| Data retention / deletion | ◑ | Ebbinghaus decay + prune for memories; audit retention 90d. No formal data-retention/DSAR policy |

---

## 3. Remediated vulnerabilities (evidence of CC7.1)

14 findings fixed and verified across v1.4.62–1.4.65: stored XSS (×3), IDOR (×3),
auth bypass (×3), network exposure, cross-user graph leak, token-in-URL, CSS-class
injection, OTP brute-force. Full detail and code anchors in `SECURITY.md` §3.

---

## 4. Gap remediation plan (path to a real SOC 2)

| Priority | Gap | Action |
|---|---|---|
| P0 | No legal entity | Incorporate; assign control ownership |
| P0 | No written policies | Adopt InfoSec, Access Control, Incident Response, Change Mgmt, Vendor Mgmt, BCP/DR (use a compliance platform's templates) |
| P1 | Access governance | Enforce MFA, quarterly access reviews, onboarding/offboarding checklist |
| P1 | Monitoring & IR | Centralized logging + alerting; documented incident-response runbook |
| P1 | Vendor management | Subprocessor register (Cloudflare, Resend, Anthropic/Ollama, Railway/host) |
| P2 | Hosted auth | Real authentication + RBAC for shared-Postgres / multi-tenant mode |
| P2 | Backup/DR | Documented backup + restore test cadence |

**Suggested sequence:** entity → free readiness scan (Comp AI / SecurityWall) →
compliance platform (Vanta / Drata / TrustCloud startup tier) for policies + evidence
automation → CPA firm for Type I, then Type II.

---

## 5. How to get an independent (free) third-party readiness report

1. **Comp AI** — https://www.trycomp.ai/soc2-readiness-assessment (instant gap analysis + score)
2. **SecurityWall** — 10-min, no signup, browser-local
3. **Cyberday.ai** — https://www.cyberday.ai/assessment/soc-2

Use §2 of this report as your answer key — most questions map directly to the
control rows above. These produce a **readiness** report, not an attestation.

> Vulnerability reports: mishrasachit1@gmail.com
