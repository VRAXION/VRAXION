# INSTNCT Pilot Data Handling Checklist

Status: 059 pilot data handling checklist.

This checklist classifies data before any public-benefit pilot candidate can
advance. It does not authorize regulated data processing.

## Data Category Matrix

| Data category | Allowed by default | Requires separate written agreement | Forbidden before compliance review |
|---|---:|---:|---:|
| synthetic data | yes | no | no |
| public non-sensitive data | yes | no | no |
| internal admin data | no | yes | no |
| PHI | no | yes | yes |
| student education records | no | yes | yes |
| minors sensitive data | no | yes | yes |
| biometric data | no | yes | yes |
| financial data | no | yes | yes |
| live clinical data | no | yes | yes |
| live grading/admissions data | no | yes | yes |

## Required Checks

- Data source identified.
- Data owner identified.
- Data minimization plan written.
- Retention period written.
- Deletion path written.
- Access list written.
- Audit log review owner assigned.
- Sensitive data is excluded unless a separate written agreement exists.

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
