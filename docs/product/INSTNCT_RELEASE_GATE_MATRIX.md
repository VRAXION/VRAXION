# INSTNCT Release Gate Matrix

Status: 056 productization planning artifact.

This document defines the release gates between the current research artifact
and future SDK, deployment, pilot, and commercial packages.

## Gate Matrix

| Gate | Target | Required Evidence | Release Status |
| --- | --- | --- | --- |
| 056 | Product/license/compliance blueprint | Docs, boundaries, roadmap | Planning positive |
| 057 | SDK/API release candidate | Stable SDK surface, examples, schemas, errors, license headers | Not started |
| 058 | Deployment harness | Local/private deployment scripts, rollback, logs, security baseline | Not started |
| 059 | Hospital/school pilot boundary | Pilot agreements, allowed/forbidden use checks, privacy gates | Not started |
| 060 | License package | Source-available license, commercial license template, public-benefit rider | Not started |
| 061 | Release candidate package | Versioned artifact, docs, checksums, regression suite, support boundary | Not started |

## Universal Gates

Every future release must satisfy:

- No black-box long run.
- Continuous progress writeouts.
- Reproducible commands.
- Schema versioning.
- Checkpoint save/load verification.
- Rollback proof where training/search occurs.
- Regression suite.
- Acceptable-use boundary.
- Claim boundary.
- Security/privacy review for any sensitive data.

## High-Stakes Domain Gates

Before hospital or school use beyond non-clinical/non-high-stakes support:

- Formal intended-use statement.
- Legal/regulatory classification.
- Data protection agreement.
- Human oversight plan.
- External validation plan.
- Incident response process.
- Compliance approval.

## Claim Boundary

056 supports productization planning only. It does not support production
release, public beta promotion, production API readiness, clinical readiness,
school high-stakes readiness, full VRAXION, language grounding, consciousness,
biological/FlyWire equivalence, or physical quantum behavior.

