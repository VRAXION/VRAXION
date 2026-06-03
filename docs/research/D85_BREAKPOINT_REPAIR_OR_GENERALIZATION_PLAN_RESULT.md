# D85 Breakpoint Repair Or Generalization Plan Result

## Status

Implemented in `scripts/probes/run_d85_breakpoint_repair_or_generalization_plan.py` with validation in `scripts/probes/run_d85_breakpoint_repair_or_generalization_plan_check.py`.

## Run command

```bash
python scripts/probes/run_d85_breakpoint_repair_or_generalization_plan.py \
  --out target/pilot_wave/d85_breakpoint_repair_or_generalization_plan \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20
```

## Validation command

```bash
python scripts/probes/run_d85_breakpoint_repair_or_generalization_plan_check.py \
  --out target/pilot_wave/d85_breakpoint_repair_or_generalization_plan
```

## Expected decision

If the combined low-cost + top1 ambiguity target is selected while the top1 guard remains a hard invariant, the expected decision is:

- `decision=combined_low_cost_top1_ambiguity_plan_selected`
- `next=D86_COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PROTOTYPE`

## Boundary

D85 only plans repair/generalization after D84 stress mapping in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
