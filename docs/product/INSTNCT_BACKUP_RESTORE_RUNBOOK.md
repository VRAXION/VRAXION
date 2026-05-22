# INSTNCT Backup Restore Runbook

Status: 064 synthetic backup/restore runbook draft.

064 validates only a local synthetic restore drill. The fixture is generated
under the 064 target output and is not customer data, PHI, student records,
clinical data, grading/admissions data, or a repo source file backup.

## Drill Steps

1. Generate a synthetic fixture under the 064 output directory.
2. Compute `original_sha256_normalized_lf`.
3. Copy the fixture into a local backup directory under the same 064 output root.
4. Restore the fixture into a separate restore directory under the same output root.
5. Compute `restored_sha256_normalized_lf`.
6. Require `hash_match = true`.

No OpenTelemetry compliance.
No SOC2 readiness.
No HIPAA readiness.
No FERPA readiness.
No production monitoring.
No production deployment.
No hosted SaaS.
No public beta.
No production API readiness.
No production SRE readiness.
No SLA.
No disaster recovery guarantee.
No clinical use.
No high-stakes education use.
No PHI/student records.
No full VRAXION.
No language grounding.
No consciousness.
No biological/FlyWire equivalence.
No physical quantum behavior.

## Boundary

Exact boundary tokens:

```text
no production deployment
no hosted SaaS
no public beta
no production API readiness
no production SRE readiness
no SLA
no disaster recovery guarantee
no clinical use
no high-stakes education use
no PHI/student records
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```
