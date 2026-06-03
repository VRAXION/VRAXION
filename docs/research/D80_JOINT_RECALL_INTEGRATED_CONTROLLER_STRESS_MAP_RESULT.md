# D80 Joint-Recall Integrated Controller Stress Map Result

## Status

Implemented in `scripts/probes/run_d80_joint_recall_integrated_controller_stress_map.py` with validation in `scripts/probes/run_d80_joint_recall_integrated_controller_stress_map_check.py`.

## Run command

```bash
python scripts/probes/run_d80_joint_recall_integrated_controller_stress_map.py \
  --out target/pilot_wave/d80_joint_recall_integrated_controller_stress_map \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 14001,14002,14003,14004,14005 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d80_joint_recall_integrated_controller_stress_map_check.py \
  --out target/pilot_wave/d80_joint_recall_integrated_controller_stress_map
```

## Expected decision

If all stress axes are mapped, D79 remains stable under standard stress, top1 guard ablation remains visible, and hard gates pass, the expected decision is:

- `decision=integrated_joint_recall_stress_map_completed`
- `next=D81_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`

## Boundary

D80 only maps stress breakpoints of the integrated joint-recall counter-action router in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
