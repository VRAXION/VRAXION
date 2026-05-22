# INSTNCT Security Supply-Chain Gate

Status: 063 source-only security and supply-chain gate.

`instnct_security_gate.py` is the canonical local gate. It writes internal
inventory and audit artifacts under `target/pilot_wave/...`.

## Commands

```powershell
python -m py_compile tools/security_supply_chain/instnct_security_gate.py
python tools/security_supply_chain/instnct_security_gate.py --out target/pilot_wave/stable_loop_phase_lock_063_security_supply_chain_gate/smoke --heartbeat-sec 20
```

PowerShell wrappers are convenience scripts. The Python script is the source of
truth.

## Boundary

Exact boundary tokens:

```text
no signed release
no CycloneDX compliance
no SPDX compliance
no SLSA compliance
no vulnerability-clean status
no production-ready security
no production readiness
no hosted SaaS readiness
no public beta
no clinical use
no high-stakes education use
no commercial launch
no final legal terms
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```
