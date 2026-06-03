# D88 Combined Low-Cost Top1 Ambiguity Stress Map Result

## Status

Implemented in `scripts/probes/run_d88_combined_low_cost_top1_ambiguity_stress_map.py` with validation in `scripts/probes/run_d88_combined_low_cost_top1_ambiguity_stress_map_check.py`.

## Run command

```bash
python scripts/probes/run_d88_combined_low_cost_top1_ambiguity_stress_map.py \
  --out target/pilot_wave/d88_combined_low_cost_top1_ambiguity_stress_map \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 14601,14602,14603,14604,14605 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d88_combined_low_cost_top1_ambiguity_stress_map_check.py \
  --out target/pilot_wave/d88_combined_low_cost_top1_ambiguity_stress_map
```

## Expected decision

- `decision=combined_low_cost_top1_ambiguity_stress_map_completed`
- `next=D89_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`

## Boundary

D88 only maps stress breakpoints after combined low-cost + top1/top2 ambiguity repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
