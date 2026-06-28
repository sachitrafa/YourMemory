# Business Continuity & Disaster Recovery Policy

**Policy ID:** BCP-08 · **Version:** 0.1 (draft) · **SOC 2:** A1.2–A1.3
**Owner:** Security Lead · **Approved:** _pending entity formation_
**Review cadence:** annually, and after any major outage

---

## 1. Purpose

Define how YourMemory maintains availability of its services and recovers data and
operations after a disruption.

## 2. Architecture & resilience (as implemented)

- **Local-first default.** In the default deployment, the memory store lives on the
  user's machine; there is no central service whose outage affects users.
- **Fail-open design.** Non-critical dependencies (LLM judge, graph indexing, recall
  expansion) fail open, so a downed dependency degrades quality rather than blocking the
  user. This is intentional for availability and never weakens confidentiality.
- **Stateless edge.** The Cloudflare Worker (activation) runs on Cloudflare's globally
  distributed, highly available edge; D1 is managed by Cloudflare.
- **Immutable releases.** Package versions on PyPI are immutable and independently
  reinstallable; the site and Worker redeploy from Git.

## 3. Recovery objectives (targets)

| Component | RPO (data loss) | RTO (time to restore) |
|---|---|---|
| Local memory store | User-managed (see §4) | Reinstall + restore from export |
| Hosted Postgres deployment | ≤ 24h (with daily backup — roadmap) | ≤ 8h |
| Cloudflare Worker / D1 | Provider-managed | Provider-managed |
| Source / releases | 0 (Git + PyPI) | Minutes |

## 4. Backup & restore

- **Source & releases:** Git (GitHub) + immutable PyPI artifacts — effectively continuous.
- **Cloudflare D1 (activation):** managed by Cloudflare.
- **Memory data:**
  - *Local:* covered by the user's own machine backup; **bulk export/import** to enable
    user-driven backup is a committed roadmap item.
  - *Hosted Postgres:* **automated daily backups with periodic restore tests** are a
    committed roadmap item before hosted GA.

## 5. Roadmap controls (committed)

- Documented backup schedule + restore-test cadence for hosted deployments.
- External uptime monitoring + alerting on the activation endpoint and any hosted API.
- Documented uptime SLA for enterprise/hosted offerings.

## 6. Ownership & review

Reviewed at least annually and after any significant availability incident.
