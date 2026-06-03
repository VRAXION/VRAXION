# D81 Breakpoint Repair Or Generalization Plan Result

## Status

Implemented in `scripts/probes/run_d81_breakpoint_repair_or_generalization_plan.py` with validation in `scripts/probes/run_d81_breakpoint_repair_or_generalization_plan_check.py`.

## Run command

```bash
python scripts/probes/run_d81_breakpoint_repair_or_generalization_plan.py \
  --out target/pilot_wave/d81_breakpoint_repair_or_generalization_plan \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20
```

## Validation command

```bash
python scripts/probes/run_d81_breakpoint_repair_or_generalization_plan_check.py \
  --out target/pilot_wave/d81_breakpoint_repair_or_generalization_plan
```

## Expected decision

D81 treats top1 guard corruption as both a hard invariant and a hardening target, but not as a disposable cost knob. For the first targeted D82 milestone, the expected selected repair path is `LOW_COST_PRESSURE_REPAIR_PLAN`, with the top1 guard retained as a required proof/ablation gate.

- `decision=low_cost_pressure_repair_plan_selected`
- `next=D82_LOW_COST_PRESSURE_REPAIR_WITH_TOP1_GUARD`

## Boundary

D81 only plans breakpoint repair/generalization after D80 stress mapping in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
