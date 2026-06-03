# D83 Low-Cost Pressure Repair Scale Confirm Result

## Status

Implemented in `scripts/probes/run_d83_low_cost_pressure_repair_scale_confirm.py` with validation in `scripts/probes/run_d83_low_cost_pressure_repair_scale_confirm_check.py`.

## Run command

```bash
python scripts/probes/run_d83_low_cost_pressure_repair_scale_confirm.py \
  --out target/pilot_wave/d83_low_cost_pressure_repair_scale_confirm \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 14201,14202,14203,14204,14205,14206,14207,14208 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d83_low_cost_pressure_repair_scale_confirm_check.py \
  --out target/pilot_wave/d83_low_cost_pressure_repair_scale_confirm
```

## Expected decision

If the scaled low-cost pressure repair keeps `low_cost_pressure_breakpoint >= 0.74` while preserving the top1 guard, D68, recall, safety, Rust, fallback, and failed-job gates, the expected decision is:

- `decision=low_cost_pressure_repair_scale_confirmed`
- `next=D84_LOW_COST_PRESSURE_REPAIR_STRESS_MAP`

## Boundary

D83 only scale-confirms low-cost pressure repair while preserving top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
