# INSTNCT Service Alpha Job Orchestration

Status: 062 job orchestration alpha boundary.

Jobs are local/private alpha jobs under `target/pilot_wave/`. They wrap the
058 deployment harness with safe subprocess list arguments. The service must
not use shell-concatenated commands or `shell=True`.

Each job records:

- `job_created_at`
- `request_id`
- `job_id`
- request body hash
- optional `idempotency_key`
- exact child command
- child exit code
- progress log
- audit log
- artifact manifest
- claim boundary

The alpha child command is a safe harness healthcheck command. This validates
service orchestration and harness integration without claiming a new model
experiment.

No black-box run is allowed. Long-running work must write progress over time.
The alpha smoke remains bounded, but job progress and audit rows are still
written.

## Boundary

Exact boundary tokens:

```text
no production API readiness
no production deployment
no hosted SaaS
no public beta
no multi-tenant IAM
no clinical use
no high-stakes education use
no commercial launch
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```

