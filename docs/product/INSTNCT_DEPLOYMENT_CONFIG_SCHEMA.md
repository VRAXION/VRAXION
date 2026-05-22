# INSTNCT Deployment Config Schema

Status: 058 config schema.

The 058 harness accepts `instnct_deployment_config_v1`.

## Required Fields

```json
{
  "schema_version": "instnct_deployment_config_v1",
  "sdk_schema_version": "instnct_sdk_candidate_v1",
  "deployment_mode": "local_research",
  "intended_use": "research",
  "out_dir": "target/pilot_wave/stable_loop_phase_lock_058_deployment_harness/example_local",
  "seed": 2026,
  "progress_path": "progress.jsonl",
  "audit_log_path": "audit_log.jsonl",
  "production_default_training_enabled": false,
  "public_beta_promoted": false,
  "production_api_ready": false
}
```

Allowed deployment modes:

```text
local_research
private_evaluation
```

Allowed intended uses:

```text
research
internal_evaluation
```

`out_dir` must be a relative `target/pilot_wave/...` path. Repository source
directories, absolute system directories, and path traversal are rejected.

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
