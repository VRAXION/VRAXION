# INSTNCT Ops Readiness Gate

Status: 064 local/private ops-readiness sanity tooling.

This tool records health signals, structured log samples, metrics, trace samples,
redaction checks, SLO/alerting boundary status, incident runbook checks, and a
synthetic backup/restore drill under `target/pilot_wave/...`.

Run:

```powershell
python tools/ops_readiness/instnct_ops_gate.py --out target/pilot_wave/stable_loop_phase_lock_064_observability_incident_backup_gate/smoke --heartbeat-sec 20
```

Required validation:

```powershell
python -m py_compile tools/ops_readiness/instnct_ops_gate.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_064_observability_incident_backup_check.py
python tools/ops_readiness/instnct_ops_gate.py --out target/pilot_wave/stable_loop_phase_lock_064_observability_incident_backup_gate/smoke --heartbeat-sec 20
python scripts/probes/run_stable_loop_phase_lock_064_observability_incident_backup_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_063_security_supply_chain_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py --check-only
git diff --check
```

No production deployment.
No hosted SaaS.
No public beta.
No production API readiness.
No production SRE readiness.
No SLA.
No disaster recovery guarantee.
No clinical use.
No high-stakes education use.
No PHI/student records.
No full VRAXION.
No language grounding.
No consciousness.
No biological/FlyWire equivalence.
No physical quantum behavior.
