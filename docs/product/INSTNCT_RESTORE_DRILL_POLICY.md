# INSTNCT Restore Drill Policy

Status: 064 restore drill policy draft.

The 064 restore drill is synthetic-only and local-only. It records normalized-LF
SHA-256 hashes before and after restore and fails if the restored hash differs.

## Required Fields

`backup_manifest.json` and `restore_verification.json` must include:

```text
original_sha256_normalized_lf
restored_sha256_normalized_lf
hash_match = true
```

The drill must state synthetic-only status and must not use customer data, PHI,
student records, clinical data, grading/admissions data, or repo source files.

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
