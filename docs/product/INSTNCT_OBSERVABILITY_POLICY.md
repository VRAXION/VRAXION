# INSTNCT Observability Policy

Status: 064 observability policy draft for local/private evaluation.

064 records local health, log, metric, trace, and restore-drill evidence so an
operator can see whether a bounded local/private evaluation run is observable at
a basic level. The policy is evidence-oriented: every long or multi-phase gate
writes `progress.jsonl` instead of waiting for final completion.

## Required Local Signals

- `health_signals.json` from the 062 service alpha healthcheck.
- `structured_log_sample.jsonl` with sensitive sample values redacted.
- `metrics_snapshot.json` with local readiness counters and thresholds.
- `trace_sample.jsonl` with synthetic trace/span IDs.
- `slo_alert_evaluation.json` with boundary-only local alert signals.
- `restore_drill_summary.json` for synthetic backup/restore status.

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
