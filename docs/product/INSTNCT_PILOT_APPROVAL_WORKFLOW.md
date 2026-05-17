# INSTNCT Pilot Approval Workflow

Status: 059 pilot approval workflow.

This workflow decides whether a public-benefit pilot candidate may advance to
separate written approval. It does not itself start a pilot.

## Workflow

1. Request intake.
2. Organization and use-case classification.
3. Data classification.
4. Human oversight review.
5. Deployment mode review.
6. Claim-boundary review.
7. Approval or rejection decision.
8. Written approval package, if allowed.
9. Pilot closeout review after completion.

## Approval Preconditions

- Intended use is non-clinical only or non-high-stakes only.
- Deployment mode is `local_research` or `private_evaluation`.
- Data handling checklist passes.
- Human oversight checklist passes.
- Rejection reasons do not apply.
- Written approval is issued.

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
