# INSTNCT Service API Alpha

Status: 062 localhost-only service/API alpha.

`instnct_service_alpha.py` is the canonical alpha service wrapper. It is local
and private evaluation infrastructure only. It is no production API readiness,
no hosted SaaS, no public beta, no multi-tenant IAM, no clinical use, no
high-stakes education use, no commercial launch, no full VRAXION, no language
grounding, no consciousness, no biological/FlyWire equivalence, and no physical
quantum behavior.

085 adds one bounded chat service API alpha route:

```text
POST /v1/bounded-chat/infer
```

The route is service API alpha only and localhost/private only. It is not
deploy-ready service, not public API, not SDK surface, not GPT-like assistant,
not open-domain chat, not production chat, not safety alignment, and not public
beta / GA / hosted SaaS. The Python service does not duplicate bounded-chat
inference logic; it calls `cargo run -p instnct-core --example
phase_lane_bounded_chat_inference_runtime` and parses `single_inference.json`,
`runtime_metrics.json`, `summary.json`, `report.md`, and `audit_log.jsonl`.

## Commands

```powershell
python -m py_compile tools/instnct_service_alpha/instnct_service_alpha.py
python tools/instnct_service_alpha/instnct_service_alpha.py healthcheck --config tools/instnct_service_alpha/config/example.local.json
python tools/instnct_service_alpha/instnct_service_alpha.py smoke --config tools/instnct_service_alpha/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_062_service_api_alpha/smoke
python tools/instnct_service_alpha/instnct_service_alpha.py serve --config tools/instnct_service_alpha/config/example.local.json
python -m py_compile scripts/probes/run_stable_loop_phase_lock_085_bounded_chat_service_api_alpha_check.py
python tools/instnct_service_alpha/instnct_service_alpha.py smoke --config tools/instnct_service_alpha/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_085_bounded_chat_service_api_alpha/smoke
python scripts/probes/run_stable_loop_phase_lock_085_bounded_chat_service_api_alpha_check.py --check-only
```

PowerShell wrappers are convenience scripts. The Python service is the source
of truth.

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
