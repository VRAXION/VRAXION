# INSTNCT Metrics Traces Policy

Status: 064 metrics and trace policy draft.

064 records local metrics and trace samples to prove the evaluation stack can
emit basic operational signals. The samples are local JSON/JSONL artifacts, not
external telemetry exports.

## Required Metrics

```text
service_health_pass
error_rate_threshold
request_count_sample
error_count_sample
redaction_candidate_fields
artifact_validation_pass
```

## Required Trace Shape

`trace_sample.jsonl` must include a synthetic trace ID, span IDs, operation
names, status, and sample duration fields. The trace is a local sample only.

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
