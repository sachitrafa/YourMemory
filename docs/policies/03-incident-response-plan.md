# Incident Response Plan

**Policy ID:** IRP-03 · **Version:** 0.1 (draft) · **SOC 2:** CC7.3–7.4
**Owner:** Security Lead (acting Incident Commander) · **Approved:** _pending entity formation_
**Review cadence:** annually, and after every Sev-1/Sev-2 incident

---

## 1. Purpose

Define how YourMemory detects, responds to, contains, and recovers from security
incidents and data breaches, and how affected parties are notified.

## 2. What is an incident

Any event that compromises, or is reasonably suspected to compromise, the
confidentiality, integrity, or availability of YourMemory systems or customer data —
e.g. unauthorized memory access, leaked API key or secret, cross-user data exposure,
audit-chain tampering, dependency compromise, or worker/account takeover.

## 3. Severity classification

| Severity | Definition | Target response |
|---|---|---|
| **Sev-1 Critical** | Confirmed breach of customer data, secret exposure, or cross-tenant leak | Begin within 1 hour |
| **Sev-2 High** | Exploitable vulnerability under active risk; suspected unauthorized access | Begin within 4 hours |
| **Sev-3 Medium** | Non-exploited vulnerability or control failure with limited impact | Begin within 2 business days |
| **Sev-4 Low** | Minor policy deviation, no data impact | Next business day |

## 4. Roles

- **Incident Commander (IC):** the Security Lead (Founder). Owns the incident end to end.
- **Technical responder:** investigates and remediates (may be the IC pre-scale).
- **Communications:** the IC handles customer/regulator notifications until a dedicated
  role exists.

## 5. Response process

1. **Detect & report.** Anyone who suspects an incident reports it to the Security Lead
   (mishrasachit1@gmail.com) immediately. External reports use the same channel.
2. **Triage.** IC assigns a severity and opens an incident record (timestamped log of
   actions, findings, and decisions).
3. **Investigate.** Use the **audit trail** (`/audit`, `/audit/verify`) to establish
   who/what/when, confirm scope, and check chain integrity for tampering.
4. **Contain.** Limit damage immediately — e.g. **revoke affected API keys**
   (`revoke_agent`), rotate exposed secrets (Cloudflare/Resend/Anthropic/host),
   take the affected service to loopback-only or offline, block the actor.
5. **Eradicate.** Remove the root cause (patch the vulnerability, close the access path).
6. **Recover.** Restore normal operation; verify integrity (audit-chain verify, data
   spot-checks); monitor for recurrence.
7. **Notify.** See §6.
8. **Post-incident review.** Within 5 business days of resolution, document root cause,
   timeline, impact, and corrective actions; feed corrective actions into the risk
   register and this plan.

## 6. Breach notification

- **Customers / affected users:** notified without undue delay once a breach of their
  data is confirmed, with what happened, data involved, and remediation steps.
- **Regulatory (e.g. GDPR):** where applicable, notify the relevant supervisory authority
  within **72 hours** of becoming aware of a personal-data breach.
- **Subprocessors:** if an incident originates with a vendor (Cloudflare, Resend,
  Anthropic, host), coordinate with their security/IR contacts.

## 7. Evidence & detection inputs

- Tamper-evident **audit log** (primary forensic source)
- Application/edge logs (FastAPI, `server.log`, Cloudflare Worker observability)
- Dependency-scan alerts (Dependabot / `pip-audit`)

## 8. Roadmap

Centralized log aggregation + alerting and an external on-call/uptime monitor are
committed improvements (see Risk Assessment Policy).

## 9. Ownership & review

Reviewed annually and after every Sev-1/Sev-2 incident; tabletop exercise performed at
least annually once the team exceeds one person.
