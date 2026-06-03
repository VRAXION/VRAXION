# D92 Combined Low-Cost OOD Stress Map Result

## Status

Implemented in `scripts/probes/run_d92_combined_low_cost_ood_stress_map.py` with validation in `scripts/probes/run_d92_combined_low_cost_ood_stress_map_check.py`.

## Run command

```bash
python scripts/probes/run_d92_combined_low_cost_ood_stress_map.py \
  --out target/pilot_wave/d92_combined_low_cost_ood_stress_map \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 14901,14902,14903,14904,14905 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d92_combined_low_cost_ood_stress_map_check.py \
  --out target/pilot_wave/d92_combined_low_cost_ood_stress_map
```

## Expected decision

- `decision=combined_low_cost_ood_stress_map_completed`
- `next=D93_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`
- `best_arm=D91_COMBINED_LOW_COST_OOD_REPLAY`
- `dominant_breakpoint=COMBINED_OOD_JOINT_BOUNDARY`

## Boundary

D92 only maps stress breakpoints after combined low-cost + OOD repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
