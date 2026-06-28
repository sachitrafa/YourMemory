# Data Retention & Deletion Policy

**Policy ID:** DRD-06 · **Version:** 0.1 (draft) · **SOC 2:** C1, CC6.7, Privacy
**Owner:** Security Lead · **Approved:** _pending entity formation_
**Review cadence:** annually, or on material change

---

## 1. Purpose

Define how long YourMemory retains data, how it is deleted, and how a user's
right-to-forget is honored.

## 2. Data categories and retention

| Data | Store | Retention |
|---|---|---|
| **Memories** (user facts) | DuckDB/Postgres/SQLite | Subject to Ebbinghaus **decay + prune** — strength decays per category; memories below the prune threshold (0.05) are removed by the daily job. No fixed maximum; recall reinforces. |
| **Conversation buffer** (verbatim, opt-in) | DB | Rolling cap of N most-recent exchanges per user |
| **Audit log** | DB | **≥ 90 days** (floor enforced); pruned daily beyond the window |
| **Activation data** (email, instance ID, OTP) | Cloudflare D1 | Email/instance retained while active; OTP codes expire (10 min) and are single-use |
| **Application/edge logs** | Local / Cloudflare | Ephemeral / provider default |

## 3. Deletion mechanisms (as implemented)

- **Per-memory deletion:** `DELETE /memories/{id}` (ownership-checked) — removes the
  memory; the deletion is itself recorded in the audit log.
- **Automatic expiry:** decay-based pruning of weak memories; audit-log retention prune.

## 4. Right-to-forget (data subject deletion)

On a verified request, YourMemory will delete all data associated with a user:
- All memories for the `user_id` (and their graph nodes/edges).
- The user's conversation buffer.
- Activation records (email/instance) on request.

> **Implementation status:** per-memory and automatic deletion are implemented. A
> single-call **purge-all endpoint** (`DELETE all memories for a user_id`, audited) is a
> committed roadmap item to make full right-to-forget one operation. Until shipped, a full
> purge is performed by deleting the user's records directly, and logged.

Audit-log entries about a deleted user are retained (immutability/accountability) but
contain only IDs and metadata — never memory content — so they are not personal content.

## 5. Data minimization (privacy by design)

- The audit log records IDs and metadata only, **never raw memory content**.
- A mandatory relevance judge gates what is stored, reducing unnecessary data capture.
- Default deployment is local — memory data does not leave the user's machine.

## 6. Backups

Backup retention and restoration are governed by the Business Continuity & DR Policy (08).

## 7. Ownership & review

Reviewed at least annually and on any change to retention periods or applicable law.
