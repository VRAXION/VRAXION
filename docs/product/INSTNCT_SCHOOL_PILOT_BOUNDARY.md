# INSTNCT School Pilot Boundary

Status: 059 public-benefit pilot boundary artifact.

Schools may request review for a public-benefit pilot candidate. This document
is an intake boundary, not permission to deploy. School candidates are
non-high-stakes only and require written approval before any external pilot.

## Allowed Review Candidates

The following school use cases may request review:

- Tutoring and explanation support.
- Practice generation.
- Classroom demonstration.
- Teacher-support planning.
- Non-high-stakes learning aid.
- Research or demonstration use.

## Forbidden Before Separate Compliance Review

The following are not allowed in the 059 pilot boundary:

- Grading.
- Admissions.
- Student ranking.
- Placement decisions.
- Discipline decisions.
- Proctoring.
- Prohibited-behavior detection.
- High-stakes profiling.

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
