# D87 Combined Low-Cost Top1 Ambiguity Scale Confirm Result

## Status

Implemented in `scripts/probes/run_d87_combined_low_cost_top1_ambiguity_scale_confirm.py` with validation in `scripts/probes/run_d87_combined_low_cost_top1_ambiguity_scale_confirm_check.py`.

## Run command

```bash
python scripts/probes/run_d87_combined_low_cost_top1_ambiguity_scale_confirm.py \
  --out target/pilot_wave/d87_combined_low_cost_top1_ambiguity_scale_confirm \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 14501,14502,14503,14504,14505,14506,14507,14508 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d87_combined_low_cost_top1_ambiguity_scale_confirm_check.py \
  --out target/pilot_wave/d87_combined_low_cost_top1_ambiguity_scale_confirm
```

## Expected decision

- `decision=combined_low_cost_top1_ambiguity_scale_confirmed`
- `next=D88_COMBINED_LOW_COST_TOP1_AMBIGUITY_STRESS_MAP`

## Boundary

D87 only scale-confirms the combined low-cost + top1/top2 ambiguity repair while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
