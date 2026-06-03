# D94 Combined OOD Joint Boundary Repair Prototype Result

## Status

Implemented in `scripts/probes/run_d94_combined_ood_joint_boundary_repair_prototype.py` with validation in `scripts/probes/run_d94_combined_ood_joint_boundary_repair_prototype_check.py`.

## Run command

```bash
python scripts/probes/run_d94_combined_ood_joint_boundary_repair_prototype.py \
  --out target/pilot_wave/d94_combined_ood_joint_boundary_repair_prototype \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 15001,15002,15003,15004,15005 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d94_combined_ood_joint_boundary_repair_prototype_check.py \
  --out target/pilot_wave/d94_combined_ood_joint_boundary_repair_prototype
```

## Expected decision

- `decision=combined_ood_joint_boundary_repair_confirmed`
- `next=D95_COMBINED_OOD_JOINT_BOUNDARY_SCALE_CONFIRM`
- `best_arm=COMBINED_OOD_JOINT_BOUNDARY_REPAIR_COST_AWARE`
- `combined_ood_joint_boundary_breakpoint>=0.755`

## Boundary

D94 only repairs the combined OOD + joint-boundary breakpoint while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
