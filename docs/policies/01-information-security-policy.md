# Information Security Policy

**Policy ID:** ISP-01 · **Version:** 0.1 (draft) · **SOC 2:** CC1, CC2
**Owner:** Founder / acting Security Lead · **Approved:** _pending entity formation_
**Review cadence:** annually, or on material change

---

## 1. Purpose

This is YourMemory's master information-security policy. It states management's
commitment to protecting the confidentiality, integrity, and availability of YourMemory
systems and the customer data they process, and it sets the framework under which all
other policies in this set operate.

## 2. Scope

Applies to all YourMemory systems (the `yourmemory` package and its HTTP/MCP services,
the Cloudflare Worker activation backend, the marketing site, source repositories), all
data processed by them, and all personnel and contractors with access to those systems.

## 3. Information security principles

1. **Confidentiality first.** YourMemory stores user "memories" that frequently contain
   personal information. Protecting that data and maintaining strict per-user isolation
   is the organization's highest security priority.
2. **Least privilege.** Access to systems and data is granted only as needed to perform a
   role, and removed when no longer needed.
3. **Defense in depth.** Controls are layered (network boundary, authorization,
   input validation, audit logging) so no single failure exposes data.
4. **Accountability.** Security-relevant actions are logged in a tamper-evident audit
   trail and attributable to an actor.
5. **Secure by default.** Services bind to loopback by default; secrets are never stored
   in source; fail-open behavior is limited to availability, never confidentiality.

## 4. Roles and responsibilities

| Role | Responsibility |
|------|----------------|
| Security Lead (currently the Founder) | Owns this policy set, approves changes, accountable for the security program |
| Engineering | Implements and maintains technical controls; follows the change-management and vulnerability-management policies |
| All personnel/contractors | Follow these policies; report suspected incidents per the Incident Response Plan |

> **Segregation of duties note:** while the organization operates with a single founder,
> segregation of duties is not yet achievable. This is a documented residual risk to be
> remediated as the team grows.

## 5. Policy framework

The following sub-policies implement this master policy and are maintained in this
directory: Access Control (02), Incident Response (03), Vulnerability Management (04),
Vendor Management (05), Data Retention & Deletion (06), Change Management (07), Business
Continuity & DR (08), Risk Assessment (09).

## 6. Implemented controls (evidence)

- **Authorization / isolation** — all memory operations are scoped and filtered by
  `user_id`; mutate paths enforce ownership (no IDOR). See `SECURITY.md`.
- **Audit logging** — append-only, hash-chained audit trail; 90-day retention;
  tamper-verifiable.
- **Secrets management** — no secrets in source; platform-managed env bindings; agent
  API keys SHA-256 hashed at rest.
- **Secure development** — recurring security reviews before releases; remediation
  history tracked (14 findings fixed).

## 7. Compliance and enforcement

Violations of this policy may result in revocation of access and, for personnel,
disciplinary action. Exceptions must be documented, risk-assessed, and approved by the
Security Lead with an expiry date.

## 8. Ownership & review

This policy is reviewed at least annually and after any material change to the system or
a significant security incident.
