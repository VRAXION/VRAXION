# INSTNCT Deployment Modes

Status: 056 productization planning artifact.

This document defines deployment modes that can be planned after 056. It does
not approve any production deployment.

## Mode 1: Local Research

Purpose:

- Internal experiments.
- Reproduction of bounded evals.
- Visual replay and debugging.

Allowed data:

- Synthetic data.
- Public non-sensitive research data.
- Customer data only with written permission and isolation controls.

Default status: allowed for research.

## Mode 2: Private Enterprise Evaluation

Purpose:

- Customer-hosted or isolated evaluation.
- Non-production proof of value.
- Internal workflow simulation.

Required:

- Written evaluation agreement.
- Source-available or commercial evaluation license.
- Data handling plan.
- Audit logs.
- No production automation.

## Mode 3: Hosted SaaS Later

Purpose:

- Managed hosted product after release gates.

Required before enablement:

- Security review.
- Privacy review.
- Tenant isolation.
- Abuse monitoring.
- Data retention policy.
- Incident response policy.
- Production support plan.

Status in 056: not ready.

## Mode 4: Hospital/School Pilot

Purpose:

- Public-benefit pilot within restricted non-clinical or non-high-stakes scope.

Required:

- Written pilot agreement.
- Approved use case.
- Explicit exclusion of diagnosis, treatment, triage, grading, admissions, and
  student ranking.
- Data minimization.
- Human oversight.
- Rollback/disable path.

Status in 056: planning only.

## Deployment Gates

Every deployment mode must declare:

- Who operates it.
- What data it processes.
- Whether users are internal staff, researchers, patients, students, teachers,
  or public users.
- Whether the output can affect rights, access, healthcare, education, or
  essential services.
- What rollback and shutdown mechanisms exist.

## Claim Boundary

056 supports productization planning only. It does not support production
release, public beta promotion, production API readiness, clinical readiness,
school high-stakes readiness, full VRAXION, language grounding, consciousness,
biological/FlyWire equivalence, or physical quantum behavior.

