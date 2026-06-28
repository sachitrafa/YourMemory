# Risk Assessment Policy & Register

**Policy ID:** RAP-09 · **Version:** 0.1 (draft) · **SOC 2:** CC3.1–3.4
**Owner:** Security Lead · **Approved:** _pending entity formation_
**Review cadence:** at least annually, and on material change

---

## 1. Purpose

Define how YourMemory identifies, evaluates, and treats risks to the security and
confidentiality of its systems and customer data.

## 2. Process

1. **Identify** risks across the system (product, infrastructure, vendors, organization).
2. **Assess** each by likelihood × impact → Low / Medium / High.
3. **Treat** via mitigate / accept / transfer, with an owner and target date.
4. **Review** the register at least annually and after significant changes or incidents.

## 3. Current risk register

| ID | Risk | Likelihood | Impact | Rating | Treatment | Status |
|----|------|-----------|--------|--------|-----------|--------|
| R1 | No legal entity → cannot attest; unclear data ownership | High | High | **High** | Incorporate; assign control ownership | Open |
| R2 | No external penetration test (VAPT) | Medium | High | **High** | Commission third-party VAPT | Open |
| R3 | No formal policies (now drafted) | Medium | Medium | **Medium** | This policy set; formal adoption post-entity | In progress |
| R4 | No MFA / access reviews on cloud accounts | Medium | High | **High** | Enable MFA; quarterly access reviews | Open |
| R5 | No centralized monitoring / alerting | Medium | Medium | **Medium** | Log aggregation + alerting; uptime monitor | Open |
| R6 | No incident response history / tabletop | Low | High | **Medium** | IRP-03 adopted; run annual tabletop | In progress |
| R7 | Hosted/multi-tenant mode lacks network auth + RBAC | Medium | High | **High** | Build auth + RBAC before hosted GA | Open |
| R8 | No automated dependency scanning | Medium | Medium | **Medium** | Dependabot + `pip-audit` | In progress |
| R9 | No formal backup/DR for hosted data | Medium | High | **High** | Daily backups + restore tests (hosted) | Open |
| R10 | Single founder → no segregation of duties | High | Medium | **Medium** | Add reviewer/role as team grows | Accepted (interim) |
| R11 | Local API unauthenticated (loopback) | Low | Low | **Low** | Accepted by design; documented threat boundary | Accepted |
| R12 | LLM extractor false-positives storing low-value data | Low | Low | **Low** | Mandatory relevance judge (v1.4.72) | Mitigated |

## 4. Acceptance

Risks rated Low and explicitly accepted (R11) are documented design decisions reviewed at
each cycle. High risks require an owner and target remediation date.

## 5. Ownership & review

The Security Lead maintains this register and reviews it at least annually and after any
material change or incident.
