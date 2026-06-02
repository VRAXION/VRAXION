# D52 Mutable ECF Controller Scale Confirm Result

This document is the checked-in result stub for the executable D52 probe.

The authoritative run artifacts are written to:

```text
target/pilot_wave/d52_mutable_ecf_controller_scale_confirm/smoke/
```

Required validation:

```text
python -m py_compile scripts/probes/run_d52_mutable_ecf_controller_scale_confirm.py
python -m py_compile scripts/probes/run_d52_mutable_ecf_controller_scale_confirm_check.py
python scripts/probes/run_d52_mutable_ecf_controller_scale_confirm.py --out target/pilot_wave/d52_mutable_ecf_controller_scale_confirm/smoke --seeds 10601,10602,10603,10604,10605 --train-rows-per-seed 800 --test-rows-per-seed 800 --ood-rows-per-seed 800 --generations 200 --population 80 --workers auto --cpu-target 50-75 --heartbeat-sec 20 --scale-mode scale_lite
python scripts/probes/run_d52_mutable_ecf_controller_scale_confirm_check.py --check-only --out target/pilot_wave/d52_mutable_ecf_controller_scale_confirm/smoke
git diff --check
```

The committed source defines the contract, runner, and checker. The live result values are intentionally kept in JSON artifacts so the repository does not carry generated target output.

Boundary: D52 only scale-confirms mutable control policy for controlled symbolic joint formula discovery. It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
