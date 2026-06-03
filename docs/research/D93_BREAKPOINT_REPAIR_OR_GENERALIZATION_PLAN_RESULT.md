# D93 Breakpoint Repair or Generalization Plan Result

## Status

Implemented in `scripts/probes/run_d93_breakpoint_repair_or_generalization_plan.py` with validation in `scripts/probes/run_d93_breakpoint_repair_or_generalization_plan_check.py`.

## Run command

```bash
python scripts/probes/run_d93_breakpoint_repair_or_generalization_plan.py \
  --out target/pilot_wave/d93_breakpoint_repair_or_generalization_plan \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20
```

## Validation command

```bash
python scripts/probes/run_d93_breakpoint_repair_or_generalization_plan_check.py \
  --out target/pilot_wave/d93_breakpoint_repair_or_generalization_plan
```

## Expected decision

- `decision=combined_ood_joint_boundary_plan_selected`
- `next=D94_COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PROTOTYPE`
- `selected_repair_path=COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PLAN`
- `dominant_breakpoint=COMBINED_OOD_JOINT_BOUNDARY`

## Boundary

D93 only plans repair/generalization after D92 stress mapping in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
