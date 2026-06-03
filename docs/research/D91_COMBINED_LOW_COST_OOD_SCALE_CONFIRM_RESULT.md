# D91 Combined Low-Cost OOD Scale Confirm Result

## Status

Implemented in `scripts/probes/run_d91_combined_low_cost_ood_scale_confirm.py` with validation in `scripts/probes/run_d91_combined_low_cost_ood_scale_confirm_check.py`.

## Run command

```bash
python scripts/probes/run_d91_combined_low_cost_ood_scale_confirm.py \
  --out target/pilot_wave/d91_combined_low_cost_ood_scale_confirm \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 14801,14802,14803,14804,14805,14806,14807,14808 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d91_combined_low_cost_ood_scale_confirm_check.py \
  --out target/pilot_wave/d91_combined_low_cost_ood_scale_confirm
```

## Expected decision

- `decision=combined_low_cost_ood_scale_confirmed`
- `next=D92_COMBINED_LOW_COST_OOD_STRESS_MAP`
- `best_arm=D90_COMBINED_LOW_COST_OOD_REPAIR_REPLAY`

## Boundary

D91 only scale-confirms the combined low-cost + OOD support distribution repair while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
