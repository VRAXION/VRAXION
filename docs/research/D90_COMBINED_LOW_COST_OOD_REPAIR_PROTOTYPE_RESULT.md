# D90 Combined Low-Cost OOD Repair Prototype Result

## Status

Implemented in `scripts/probes/run_d90_combined_low_cost_ood_repair_prototype.py` with validation in `scripts/probes/run_d90_combined_low_cost_ood_repair_prototype_check.py`.

## Run command

```bash
python scripts/probes/run_d90_combined_low_cost_ood_repair_prototype.py \
  --out target/pilot_wave/d90_combined_low_cost_ood_repair_prototype \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 14701,14702,14703,14704,14705 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d90_combined_low_cost_ood_repair_prototype_check.py \
  --out target/pilot_wave/d90_combined_low_cost_ood_repair_prototype
```

## Expected decision

- `decision=combined_low_cost_ood_repair_confirmed`
- `next=D91_COMBINED_LOW_COST_OOD_SCALE_CONFIRM`
- `best_arm=COMBINED_LOW_COST_OOD_REPAIR_COST_AWARE`

## Boundary

D90 only repairs the combined low-cost + OOD support distribution shift breakpoint while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
