# INSTNCT Redaction Policy

Status: 064 redaction policy draft.

064 verifies redaction with explicit synthetic sensitive samples. The raw values
are used in memory by the gate and must not appear in `structured_log_sample.jsonl`.

## Required Redaction Coverage

- bearer token sample
- api key sample
- password sample
- patient name sample
- student name sample
- PHI marker sample
- student-record marker sample
- secret sample

`redaction_report.json` must include `raw_sensitive_values_tested`,
`raw_sensitive_values_found = 0`, and `redacted_fields_count > 0`.

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
