# Change Management Policy

**Policy ID:** CMP-07 · **Version:** 0.1 (draft) · **SOC 2:** CC8.1
**Owner:** Security Lead · **Approved:** _pending entity formation_
**Review cadence:** annually, or on material change

---

## 1. Purpose

Ensure changes to YourMemory systems are made deliberately, reviewed, tested, traceable,
and reversible.

## 2. Scope

All changes to the `yourmemory` codebase, the Cloudflare Worker, the marketing site,
infrastructure configuration, and dependencies.

## 3. Change process (as implemented)

1. **Version control.** All changes are made in Git; every commit has a descriptive
   message stating what changed and why. History is the authoritative change record.
2. **Testing.** Changes are verified before release — smoke tests against a live server,
   build validation (`python -m build`, `twine check`), and headless render checks for
   the site. Security-relevant changes get a security review (per VMP-04).
3. **Versioned releases.** Application changes ship as semver-tagged PyPI releases; each
   release maps to specific commits. The Worker is deployed via `wrangler`.
4. **Reversibility.** Releases are immutable and prior versions remain installable; the
   Worker and site can be redeployed from a prior commit. Risky changes back up the
   prior artifact before replacement.
5. **Traceability.** Commit → version → deployed artifact is traceable end to end.

## 4. Roadmap controls (committed)

- **CI gate:** automated test + dependency-scan (Dependabot/`pip-audit`) checks on pull
  requests before merge.
- **Peer review:** PR review/approval once the team exceeds one engineer (segregation of
  duties).
- **Documented release checklist** for production changes.

## 5. Emergency changes

Security hotfixes may bypass normal cadence but must be committed with rationale,
released, and retroactively reviewed within 2 business days.

## 6. Ownership & review

Reviewed at least annually and after any change-related incident.
