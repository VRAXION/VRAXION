# INSTNCT Deployment Harness

Status: 058 local/private deployment harness artifact.

The deployment harness wraps the 057 doc-hidden SDK candidate in a bounded,
auditable local/private execution flow. It is not a production deployment
system and does not promote the SDK candidate to a production API.

## Canonical Entry Point

`tools/instnct_deploy/instnct_deploy.py` is the source of truth.

PowerShell scripts are convenience wrappers for the current Windows workspace:

```text
tools/instnct_deploy/run_local.ps1
tools/instnct_deploy/healthcheck.ps1
tools/instnct_deploy/smoke.ps1
```

## Capabilities

- Load and validate a deployment config.
- Write `resolved_config.json` and normalized-LF config SHA-256.
- Run a side-effect-light healthcheck.
- Run the 057 SDK candidate smoke through safe subprocess list arguments.
- Write deployment `progress.jsonl` and `audit_log.jsonl`.
- Verify fresh SDK smoke artifacts.
- Verify checkpoint save/load, rollback, and visual export evidence.
- Reject regulated or production-contaminated deployment configs.

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
