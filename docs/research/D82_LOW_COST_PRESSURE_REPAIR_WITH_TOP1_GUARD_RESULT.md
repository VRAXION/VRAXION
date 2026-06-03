# D82 Low-Cost Pressure Repair With Top1 Guard Result

## Status

Implemented in `scripts/probes/run_d82_low_cost_pressure_repair_with_top1_guard.py` with validation in `scripts/probes/run_d82_low_cost_pressure_repair_with_top1_guard_check.py`.

## Run command

```bash
python scripts/probes/run_d82_low_cost_pressure_repair_with_top1_guard.py \
  --out target/pilot_wave/d82_low_cost_pressure_repair_with_top1_guard \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 14101,14102,14103,14104,14105 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d82_low_cost_pressure_repair_with_top1_guard_check.py \
  --out target/pilot_wave/d82_low_cost_pressure_repair_with_top1_guard
```

## Expected decision

If the low-cost pressure breakpoint improves to at least `0.74` while preserving the top1 guard, D68, recall, safety, Rust, fallback, and failed-job gates, the expected decision is:

- `decision=low_cost_pressure_repair_confirmed`
- `next=D83_LOW_COST_PRESSURE_REPAIR_SCALE_CONFIRM`

## Boundary

D82 only repairs low-cost pressure while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
