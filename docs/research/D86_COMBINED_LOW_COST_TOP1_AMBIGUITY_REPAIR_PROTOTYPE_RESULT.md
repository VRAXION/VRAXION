# D86 Combined Low-Cost Top1 Ambiguity Repair Prototype Result

## Status

Implemented in `scripts/probes/run_d86_combined_low_cost_top1_ambiguity_repair_prototype.py` with validation in `scripts/probes/run_d86_combined_low_cost_top1_ambiguity_repair_prototype_check.py`.

## Run command

```bash
python scripts/probes/run_d86_combined_low_cost_top1_ambiguity_repair_prototype.py \
  --out target/pilot_wave/d86_combined_low_cost_top1_ambiguity_repair_prototype \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 14401,14402,14403,14404,14405 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d86_combined_low_cost_top1_ambiguity_repair_prototype_check.py \
  --out target/pilot_wave/d86_combined_low_cost_top1_ambiguity_repair_prototype
```

## Expected decision

- `decision=combined_low_cost_top1_ambiguity_repair_confirmed`
- `next=D87_COMBINED_LOW_COST_TOP1_AMBIGUITY_SCALE_CONFIRM`

## Boundary

D86 only repairs the combined low-cost + top1/top2 ambiguity breakpoint while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
