# D95 Combined OOD Joint Boundary Scale Confirm Result

## Status

Implemented in `scripts/probes/run_d95_combined_ood_joint_boundary_scale_confirm.py` with validation in `scripts/probes/run_d95_combined_ood_joint_boundary_scale_confirm_check.py`.

## Run command

```bash
python scripts/probes/run_d95_combined_ood_joint_boundary_scale_confirm.py \
  --out target/pilot_wave/d95_combined_ood_joint_boundary_scale_confirm \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 16001,16002,16003,16004,16005,16006,16007,16008 \
  --train-rows-per-seed 360 \
  --test-rows-per-seed 360 \
  --ood-rows-per-seed 360
```

## Validation command

```bash
python scripts/probes/run_d95_combined_ood_joint_boundary_scale_confirm_check.py \
  --out target/pilot_wave/d95_combined_ood_joint_boundary_scale_confirm
```

## Expected decision

- `decision=combined_ood_joint_boundary_scale_confirmed`
- `next=D96_NEXT_BREAKPOINT_OR_GENERALIZATION_PLAN`
- `best_arm=COMBINED_OOD_JOINT_BOUNDARY_REPAIR_COST_AWARE_SCALE`
- `combined_ood_joint_boundary_breakpoint>=0.755`
- `min_seed_combined_ood_joint_boundary_breakpoint>=0.752`

## Boundary

D95 only scale-confirms the D94 combined OOD + joint-boundary repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
