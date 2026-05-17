# INSTNCT Deployment Harness

Status: 058 local/private deployment harness.

`instnct_deploy.py` is the canonical harness. The PowerShell scripts in this
directory are convenience wrappers for the current Windows workspace.

## Commands

```powershell
python tools/instnct_deploy/instnct_deploy.py validate-config --config tools/instnct_deploy/config/example.local.json
python tools/instnct_deploy/instnct_deploy.py healthcheck --config tools/instnct_deploy/config/example.local.json
powershell -ExecutionPolicy Bypass -File tools/instnct_deploy/smoke.ps1 -Config tools/instnct_deploy/config/example.local.json -Out target/pilot_wave/stable_loop_phase_lock_058_deployment_harness/smoke
```

## Boundary

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
