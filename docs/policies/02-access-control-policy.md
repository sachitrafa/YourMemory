# Access Control Policy

**Policy ID:** ACP-02 · **Version:** 0.1 (draft) · **SOC 2:** CC6.1–6.3
**Owner:** Security Lead · **Approved:** _pending entity formation_
**Review cadence:** annually, or on material change

---

## 1. Purpose

Define how access to YourMemory systems and customer data is granted, enforced,
reviewed, and revoked, under the principle of least privilege.

## 2. Data access model (as implemented)

> **Clarification:** YourMemory does **not** use shared user accounts or shared passwords.
> Access is enforced by per-user namespacing and hashed, per-agent API keys.

1. **Per-user isolation.** Every memory is scoped to a `user_id`. All reads, writes, and
   deletes filter by `user_id`; cross-user access paths (cold-start, graph traversal,
   recall) enforce the same filter. This is the primary tenant-isolation boundary.
2. **Agent API keys.** Programmatic/agent access uses keys (`ym_…`) that are
   **SHA-256 hashed at rest, displayed once at creation, and never stored in plaintext.**
   Each key carries explicit `can_read` / `can_write` permission scopes.
3. **Visibility partitions.** Memories are `shared` or `private`; private memories are
   readable only by the owning `agent_id`. `can_write` governs which partition a key may
   write.
4. **Ownership enforcement.** Mutating endpoints (`DELETE`/`PUT` memories, MCP
   `update_memory`) verify `user_id` ownership and reject mismatches (no IDOR).
5. **Local trust boundary.** In the default local deployment the HTTP API binds
   `127.0.0.1` and is unauthenticated by design — the trust boundary is the operating-
   system user account, which already controls the on-disk data store.

## 3. Authentication requirements

| Context | Requirement |
|---|---|
| Local API (loopback) | OS-user trust boundary; no network auth (documented design) |
| Agent/programmatic access | Valid `ym_` API key, validated on every call |
| Activation | Verified token via email→OTP (Cloudflare Worker) |
| Source/cloud accounts (GitHub, Cloudflare, PyPI, host) | **MFA required** (see §5) |

## 4. Provisioning, review, and revocation

- **Provisioning** follows least privilege; access is granted per role/need.
- **API keys** can be revoked at any time (`revoke_agent`); revoked keys fail validation
  immediately.
- **Access reviews** are performed at least quarterly to confirm active accounts and keys
  are still required.
- **Offboarding:** on personnel/contractor departure, all account access and keys are
  revoked within 24 hours.

## 5. Roadmap controls (committed, not yet implemented)

- **MFA** on all administrative cloud accounts (GitHub, Cloudflare, PyPI, hosting).
- **RBAC roles** beyond per-key read/write scopes for multi-tenant/hosted deployments.
- **Authenticated, network-exposed API** for any non-loopback (hosted) deployment.

These are tracked as open items in `SOC2_READINESS_REPORT.md` and the risk register.

## 6. Accountability

All access events (read/write/delete) are recorded in the tamper-evident audit log with
actor `user_id`, `agent_id`, timestamp, and action, supporting after-the-fact review.

## 7. Ownership & review

Reviewed at least annually and after any access-related incident.
