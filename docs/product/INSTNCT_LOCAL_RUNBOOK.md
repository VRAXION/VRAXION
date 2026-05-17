# INSTNCT Local Runbook

Status: 058 local runbook.

Use this runbook for bounded local research execution of the 057 SDK candidate
through the 058 deployment harness.

## Quick Commands

```powershell
python tools/instnct_deploy/instnct_deploy.py validate-config --config tools/instnct_deploy/config/example.local.json
python tools/instnct_deploy/instnct_deploy.py healthcheck --config tools/instnct_deploy/config/example.local.json
powershell -ExecutionPolicy Bypass -File tools/instnct_deploy/smoke.ps1 -Config tools/instnct_deploy/config/example.local.json -Out target/pilot_wave/stable_loop_phase_lock_058_deployment_harness/smoke
```

## Expected Outputs

The local smoke writes deployment artifacts under the selected `target/pilot_wave`
directory and child SDK artifacts under `sdk_smoke/`.

Required deployment files:

```text
progress.jsonl
audit_log.jsonl
resolved_config.json
healthcheck.json
artifact_validation.json
summary.json
report.md
```

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
