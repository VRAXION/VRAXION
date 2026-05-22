# INSTNCT Public-Benefit Pilot Request Template

Status: 059 public-benefit pilot request template.

This template is used to request review for a pilot candidate. Submission does
not authorize a pilot. Every external pilot requires written approval.

## Request Classification

Organization type:

- [ ] hospital
- [ ] school
- [ ] nonprofit
- [ ] research organization

Intended use:

- [ ] research
- [ ] internal evaluation
- [ ] classroom demonstration
- [ ] non-clinical administration exploration
- [ ] other, requiring written explanation

User population:

- [ ] internal staff
- [ ] researchers
- [ ] teachers
- [ ] students
- [ ] patients
- [ ] public users

Data categories:

- [ ] synthetic data
- [ ] public non-sensitive data
- [ ] internal admin data
- [ ] PHI
- [ ] student education records
- [ ] minors sensitive data
- [ ] biometric data
- [ ] financial data
- [ ] live clinical data
- [ ] live grading/admissions data

Required yes/no classification:

- [ ] minors are involved
- [ ] patients are involved
- [ ] outputs affect rights, access, health, or education outcomes
- [ ] requested deployment mode is `local_research`
- [ ] requested deployment mode is `private_evaluation`
- [ ] named human owner is assigned
- [ ] rollback or disable plan exists
- [ ] data retention plan exists

## Required Narrative

- Requested pilot purpose.
- Why the use is non-clinical only or non-high-stakes only.
- Data minimization plan.
- Human oversight process.
- Audit log review plan.
- Closeout review plan.

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
