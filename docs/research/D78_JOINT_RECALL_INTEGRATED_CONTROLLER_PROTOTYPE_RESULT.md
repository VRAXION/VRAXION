# D78 Joint-Recall Integrated Controller Prototype Result

## Status

Implemented in `scripts/probes/run_d78_joint_recall_integrated_controller_prototype.py` with validation in `scripts/probes/run_d78_joint_recall_integrated_controller_prototype_check.py`.

## Run command

```bash
python scripts/probes/run_d78_joint_recall_integrated_controller_prototype.py \
  --out target/pilot_wave/d78_joint_recall_integrated_controller_prototype \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 13801,13802,13803,13804,13805 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d78_joint_recall_integrated_controller_prototype_check.py \
  --out target/pilot_wave/d78_joint_recall_integrated_controller_prototype
```

## Expected decision

The expected integrated fair arm is `INTEGRATED_JOINT_RECALL_ROUTER_COST_AWARE`. If the integrated router is invoked and all gates pass, the expected decision is:

- `decision=joint_recall_integrated_controller_prototype_confirmed`
- `next=D79_JOINT_RECALL_INTEGRATED_CONTROLLER_SCALE_CONFIRM`

## Boundary

D78 only tests integrated joint-recall counter-action routing inside the controlled symbolic ECF/IPF joint formula discovery stack. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
