# INSTNCT Hospital Pilot Boundary

Status: 059 public-benefit pilot boundary artifact.

Hospitals may request review for a public-benefit pilot candidate. This
document is an intake boundary, not permission to deploy. Hospital candidates
are non-clinical only and require written approval before any external pilot.

## Allowed Review Candidates

The following hospital use cases may request review:

- Non-clinical administration exploration.
- Research support using synthetic or approved non-sensitive data.
- Internal document-routing experiments without patient-care decisions.
- Model behavior visualization and audit/demo workflows.
- Non-clinical education or simulation.

## Forbidden Before Separate Compliance Review

The following are not allowed in the 059 pilot boundary:

- Diagnosis.
- Treatment recommendation.
- Triage.
- Medication decisions.
- Clinical decision support.
- Emergency prioritization.
- Patient-specific risk scoring.
- Any direct patient-care decision.

## Deployment Boundary

Pilot candidates are restricted to:

- `local_research`
- `private_evaluation`

Rejected deployment requests:

- hosted SaaS
- production deployment
- public beta
- customer-facing unsupervised use

## Required Controls

- Written use-case request.
- Named human owner.
- Human review before any operational use of outputs.
- Audit log review.
- Disable or rollback path.
- Incident contact.
- Closeout review.

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
