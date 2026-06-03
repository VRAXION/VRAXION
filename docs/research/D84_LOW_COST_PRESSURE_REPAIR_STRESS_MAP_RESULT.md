# D84 Low-Cost Pressure Repair Stress Map Result

## Status

Implemented in `scripts/probes/run_d84_low_cost_pressure_repair_stress_map.py` with validation in `scripts/probes/run_d84_low_cost_pressure_repair_stress_map_check.py`.

## Run command

```bash
python scripts/probes/run_d84_low_cost_pressure_repair_stress_map.py \
  --out target/pilot_wave/d84_low_cost_pressure_repair_stress_map \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 14301,14302,14303,14304,14305 \
  --train-rows-per-seed 240 \
  --test-rows-per-seed 240 \
  --ood-rows-per-seed 240
```

## Validation command

```bash
python scripts/probes/run_d84_low_cost_pressure_repair_stress_map_check.py \
  --out target/pilot_wave/d84_low_cost_pressure_repair_stress_map
```

## Expected decision

If the stress map is complete and D83 core gates hold, the expected decision is:

- `decision=low_cost_pressure_repair_stress_map_completed`
- `next=D85_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`

## Boundary

D84 only maps stress breakpoints after low-cost pressure repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
