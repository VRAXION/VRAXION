# INSTNCT Pilot Rejection Reasons

Status: 059 pilot rejection reasons.

A public-benefit pilot candidate must be rejected when any reason below
applies.

## Healthcare Rejection Reasons

- Clinical diagnosis.
- Treatment recommendation.
- Triage.
- Medication decision.
- Clinical decision support or CDS.
- Emergency prioritization.
- Patient-specific risk scoring.
- Direct patient-care decision.

## Education Rejection Reasons

- Grading.
- Admissions.
- Student ranking.
- Placement.
- Proctoring.
- Discipline decision.
- High-stakes profiling.
- Prohibited-behavior detection.

## Data And Deployment Rejection Reasons

- PHI or student records without agreement.
- Production automation.
- Hosted/SaaS request.
- Public beta request.
- Customer-facing unsupervised use.
- Missing human owner.
- Unclear data retention.
- Claim-boundary conflict.

## Claim Boundary

059 supports public-benefit pilot boundary review only.

Exact boundary tokens:

```text
no production deployment
no hosted SaaS
no public beta
no production API readiness
no clinical use
no high-stakes education use
no PHI/student records without separate written agreement
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```
