# D76 Joint-Recall Component Scale Confirm Result

## Status

Implemented in `scripts/probes/run_d76_joint_recall_component_scale_confirm.py` with validation in `scripts/probes/run_d76_joint_recall_component_scale_confirm_check.py`.

## Run command

```bash
python scripts/probes/run_d76_joint_recall_component_scale_confirm.py \
  --out target/pilot_wave/d76_joint_recall_component_scale_confirm \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 13701,13702,13703,13704,13705,13706,13707,13708 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d76_joint_recall_component_scale_confirm_check.py \
  --out target/pilot_wave/d76_joint_recall_component_scale_confirm
```

## Expected decision

If the D75 component remains inside all scale, oracle-gap, recall, D68, routing, safety, truth-leak, Rust, fallback, and failed-job gates, the expected decision is:

- `decision=joint_recall_component_scale_confirmed`
- `next=D77_JOINT_RECALL_COMPONENT_INTEGRATION_PLAN`

## Boundary

D76 only scale-confirms joint-recall component migration in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
