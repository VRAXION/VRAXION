# INSTNCT Service Alpha Runbook

Status: 062 service alpha runbook.

Run static checks:

```powershell
python -m py_compile tools/instnct_service_alpha/instnct_service_alpha.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py
python scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py --check-only
```

Run service smoke:

```powershell
python tools/instnct_service_alpha/instnct_service_alpha.py smoke --config tools/instnct_service_alpha/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_062_service_api_alpha/smoke
```

Run existing regressions:

```powershell
cargo check -p instnct-core --example instnct_sdk_candidate_smoke
cargo test -p instnct-core sdk_candidate
git diff --check
```

Start the local service manually:

```powershell
python tools/instnct_service_alpha/instnct_service_alpha.py serve --config tools/instnct_service_alpha/config/example.local.json
```

The manual service is localhost-only.

No production API readiness.
No production deployment.
No hosted SaaS.
No public beta.
No multi-tenant IAM.
No clinical use.
No high-stakes education use.
No commercial launch.
No full VRAXION.
No language grounding.
No consciousness.
No biological/FlyWire equivalence.
No physical quantum behavior.

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
