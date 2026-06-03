# D79 Joint-Recall Integrated Controller Scale Confirm Result

## Status

Implemented in `scripts/probes/run_d79_joint_recall_integrated_controller_scale_confirm.py` with validation in `scripts/probes/run_d79_joint_recall_integrated_controller_scale_confirm_check.py`.

## Run command

```bash
python scripts/probes/run_d79_joint_recall_integrated_controller_scale_confirm.py \
  --out target/pilot_wave/d79_joint_recall_integrated_controller_scale_confirm \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 13901,13902,13903,13904,13905,13906,13907,13908 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d79_joint_recall_integrated_controller_scale_confirm_check.py \
  --out target/pilot_wave/d79_joint_recall_integrated_controller_scale_confirm
```

## Expected decision

The expected scaled integrated arm is `D78_INTEGRATED_ROUTER_COST_AWARE_REPLAY`. If the integrated router remains invoked, D76/D78 support and safety gates hold, and top1 sufficiency ablation remains worse, the expected decision is:

- `decision=joint_recall_integrated_controller_scale_confirmed`
- `next=D80_JOINT_RECALL_INTEGRATED_CONTROLLER_STRESS_MAP`

## Boundary

D79 only scale-confirms integrated joint-recall counter-action routing inside the controlled symbolic ECF/IPF joint formula discovery stack. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
