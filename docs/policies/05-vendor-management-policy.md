# Vendor Management Policy & Subprocessor Register

**Policy ID:** VMP-05 · **Version:** 0.1 (draft) · **SOC 2:** CC9.2
**Owner:** Security Lead · **Approved:** _pending entity formation_
**Review cadence:** annually, and on adding/removing a vendor

---

## 1. Purpose

Define how YourMemory selects, assesses, and monitors third-party vendors and
subprocessors that handle YourMemory systems or customer data.

## 2. Vendor selection criteria

When selecting a vendor that will process customer data, YourMemory prefers providers
that:
1. Hold a recognized security attestation (SOC 2, ISO 27001) or equivalent.
2. Offer a Data Processing Agreement (DPA) and publish a security/privacy posture.
3. Support encryption in transit and at rest.
4. Provide a documented incident/breach notification commitment.

## 3. Risk assessment

Before adoption, each data-handling vendor is assessed against the criteria in §2 and
assigned a risk level (Low/Medium/High) based on the sensitivity of data exposed. High-
risk vendors require a signed DPA before production use.

## 4. Subprocessor Register

| Vendor | Purpose | Data exposed | Compliance | Risk |
|---|---|---|---|---|
| **Cloudflare** | Worker activation backend, D1 database, edge | Emails, instance IDs, OTP codes | SOC 2, ISO 27001 | Low |
| **Resend** | Transactional email (OTP, activation tokens) | Email addresses | SOC 2 | Low |
| **Anthropic** | Claude API — *optional* extraction backend; worker token-count | Extraction text *only if* `YOURMEMORY_EXTRACT_BACKEND=anthropic` | SOC 2, ISO 27001 | Medium (opt-in) |
| **Ollama** | Local LLM extraction/judge | **None — runs on-device, no data leaves the machine** | N/A (local) | Low |
| **Hosting provider** (e.g. Railway) | Shared-Postgres deployments | All memory data when self-hosted there | Provider-dependent | Medium |
| **PyPI** | Package distribution | None (public artifacts) | — | Low |
| **GitHub** | Source hosting, CI, Dependabot | Source code (no customer data) | SOC 2, ISO 27001 | Low |
| **Hugging Face** | Embedding model download | None (model pull only) | — | Low |
| **unpkg / Google Fonts / Vercel Analytics** | Marketing site assets/analytics | Site visitor data only | — | Low |

> **Default-deployment note:** in the default local deployment, memory data never leaves
> the user's machine — extraction/embedding/storage are all local. Cloud subprocessors
> are involved only for activation (email/OTP) and optional hosted/Anthropic modes.

## 5. Ongoing monitoring

- The register is reviewed at least annually and updated whenever a vendor is added or
  removed.
- Material vendor security incidents are handled per the Incident Response Plan.

## 6. Contractual requirements

New data-processing vendor relationships require a DPA and security terms before
production use. Existing reputable vendors are being brought under DPAs as part of the
compliance roadmap.

## 7. Ownership & review

Reviewed at least annually and on any change to the subprocessor list.
