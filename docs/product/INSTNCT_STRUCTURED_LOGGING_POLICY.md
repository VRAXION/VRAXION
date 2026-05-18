# INSTNCT Structured Logging Policy

Status: 064 structured logging policy draft.

Structured logs for local/private evaluation must be JSONL, timestamped, and
scoped to operational events. The 064 sample log exists only to validate shape
and redaction behavior.

## Required Fields

- `timestamp_ms`
- `level`
- `event`
- `request_id` when applicable
- `route` or operation name when applicable
- status or result field

## Redaction Requirement

`structured_log_sample.jsonl` must not contain raw bearer token sample, api key
sample, password sample, patient name sample, student name sample, PHI marker
sample, student-record marker sample, or secret sample. Redacted values must be
visible as redaction markers, and `redaction_report.json` must record
`raw_sensitive_values_tested`, `raw_sensitive_values_found = 0`, and
`redacted_fields_count > 0`.

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
