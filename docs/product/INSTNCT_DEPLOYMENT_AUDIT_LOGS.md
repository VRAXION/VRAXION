# INSTNCT Deployment Audit Logs

Status: 058 audit log contract.

The deployment harness writes append-only JSONL audit records to
`audit_log.jsonl`.

## Required Events

```text
config_loaded
config_validated
policy_decision
healthcheck_started
healthcheck_completed
sdk_smoke_started
sdk_smoke_completed
artifact_validation_started
artifact_validation_completed
final_verdict
```

Each event includes a timestamp, status, normalized-LF config SHA-256, and
structured details.

## Rejection Events

Regulated deployment configs may write only progress, audit, and rejection
summary outputs. They must not create SDK smoke artifacts, checkpoints, or
visual bundles.

## Claim Boundary

058 supports local/private deployment harness engineering only.

Exact boundary tokens:

```text
no production deployment
no hosted SaaS
no public beta
no clinical use
no high-stakes education use
no production API readiness
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```
