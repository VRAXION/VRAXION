# INSTNCT RC_001 Smoke Test Guide

Status: release candidate smoke guide.

These commands validate the existing SDK candidate and deployment harness
surface for local/private evaluation. They are documented for reviewer and
maintainer use; the 061 static checker verifies the command text but does not
run the commands. RC_001 is not GA, not production deployment, not public beta,
not hosted SaaS launch, and not final legal terms.

## Required Commands

```powershell
cargo check -p instnct-core --example instnct_sdk_candidate_smoke
cargo test -p instnct-core sdk_candidate
python tools/instnct_deploy/instnct_deploy.py validate-config --config tools/instnct_deploy/config/example.local.json
python tools/instnct_deploy/instnct_deploy.py healthcheck --config tools/instnct_deploy/config/example.local.json
powershell -ExecutionPolicy Bypass -File tools/instnct_deploy/smoke.ps1 -Config tools/instnct_deploy/config/example.local.json -Out target/pilot_wave/stable_loop_phase_lock_061_release_candidate_package/smoke
```

## Expected Scope

The smoke path exercises the 057 SDK candidate through the 058 deployment
harness. It writes local/private evaluation artifacts under `target/pilot_wave/`
when the smoke command is run manually. These generated outputs are not part of
the committed RC_001 package.

## Boundary

Exact boundary tokens:

```text
no GA
no production deployment
no hosted SaaS launch
no public beta
no production API readiness
no production readiness
no clinical use
no high-stakes education use
no final legal terms
no commercial launch
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```

