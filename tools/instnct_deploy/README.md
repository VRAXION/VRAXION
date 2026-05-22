# INSTNCT Deployment Harness

Status: 086 local/private bounded chat deployment harness integration.

`instnct_deploy.py` is the canonical local/private harness. It validates config,
runs the existing SDK smoke, then orchestrates the 085 bounded chat service API
alpha smoke as a child command. The harness does not implement bounded-chat
service logic or inference logic, and it does not call the 084 runtime directly
except through the 085 smoke path.

## Commands

```powershell
python tools/instnct_deploy/instnct_deploy.py validate-config --config tools/instnct_deploy/config/example.local.json
python tools/instnct_deploy/instnct_deploy.py healthcheck --config tools/instnct_deploy/config/example.local.json
python tools/instnct_deploy/instnct_deploy.py smoke --config tools/instnct_deploy/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/smoke
```

086 validation:

```powershell
python -m py_compile tools/instnct_deploy/instnct_deploy.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration_check.py
python scripts/probes/run_stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration_check.py --check-only
```

## Bounded Chat Service Child

The bounded chat service smoke is invoked exactly through the 085 service alpha:

```powershell
python tools/instnct_service_alpha/instnct_service_alpha.py smoke --config tools/instnct_service_alpha/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/service_smoke
```

The harness parses and rechecks `summary.json`, `service_metrics.json`,
`bounded_chat_request_response.json`, `child_runtime_manifest.json`, and
`audit_log.jsonl`. It writes `artifact_validation.json`, `rollback_pointer.json`,
`bounded_chat_service_manifest.json`, `bounded_chat_service_metrics.json`, and
`bounded_chat_request_response.json` into the 086 smoke output.

## Boundary

086 supports local/private deployment harness integration only.

Exact boundary tokens:

```text
no production deployment
no hosted SaaS
no public beta
no public API
no SDK release
no clinical use
no high-stakes education use
no production API readiness
no GPT-like assistant
no open-domain chat
no production chat
no safety alignment
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```

This is not production deployment, not hosted SaaS, not public beta, not public
API, not SDK release, not GPT-like assistant, not open-domain chat, not
production chat, and not safety alignment.
