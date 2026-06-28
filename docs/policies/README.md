# YourMemory — Information Security Policies

This directory holds YourMemory's information-security policy set, mapped to the SOC 2
Common Criteria (Security) and Confidentiality trust service criteria.

> **Status: DRAFT pending formal adoption.** These policies document controls that are
> already implemented or committed. They take legal effect once YourMemory is
> incorporated and the policies are formally approved and dated by the accountable owner
> (see each policy's *Ownership & review* section). Until then they are the organization's
> documented intent and operating practice.

## Policy index

| # | Policy | SOC 2 mapping | Status |
|---|--------|---------------|--------|
| 01 | [Information Security Policy](01-information-security-policy.md) | CC1, CC2 (master) | Draft |
| 02 | [Access Control Policy](02-access-control-policy.md) | CC6.1–6.3 | Draft |
| 03 | [Incident Response Plan](03-incident-response-plan.md) | CC7.3–7.4 | Draft |
| 04 | [Vulnerability Management Policy](04-vulnerability-management-policy.md) | CC7.1 | Draft |
| 05 | [Vendor Management Policy + Subprocessor Register](05-vendor-management-policy.md) | CC9.2 | Draft |
| 06 | [Data Retention & Deletion Policy](06-data-retention-deletion-policy.md) | C1, CC6.7, Privacy | Draft |
| 07 | [Change Management Policy](07-change-management-policy.md) | CC8.1 | Draft |
| 08 | [Business Continuity & Disaster Recovery Policy](08-business-continuity-dr-policy.md) | A1.2, A1.3 | Draft |
| 09 | [Risk Assessment Policy](09-risk-assessment-policy.md) | CC3.1–3.4 | Draft |

## Scope

These policies apply to the YourMemory product (the `yourmemory` package — FastAPI HTTP
service, MCP server, hook templates), the Cloudflare Worker activation backend, the
marketing site, and all personnel and contractors with access to YourMemory systems or
customer data.

## Trust criteria in scope

**Security** (Common Criteria) and **Confidentiality**. Availability, Privacy, and
Processing Integrity are roadmap items, not yet in audit scope.

## Supporting evidence documents
- [`SECURITY.md`](../../SECURITY.md) — security posture, threat model, remediation history
- [`SOC2_READINESS_REPORT.md`](../../SOC2_READINESS_REPORT.md) — control-by-control readiness
