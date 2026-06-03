# D99 Recurrent Routing Microcircuit Prototype Result

## Status

Implemented in `scripts/probes/run_d99_recurrent_routing_microcircuit_prototype.py` with validation in `scripts/probes/run_d99_recurrent_routing_microcircuit_prototype_check.py`.

## Run command

```bash
python scripts/probes/run_d99_recurrent_routing_microcircuit_prototype.py \
  --out target/pilot_wave/d99_recurrent_routing_microcircuit_prototype \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 20001,20002,20003,20004,20005,20006,20007,20008,20009,20010 \
  --train-rows-per-seed 480 \
  --test-rows-per-seed 480 \
  --ood-rows-per-seed 480 \
  --stress-seeds 20101,20102,20103,20104,20105,20106 \
  --stress-rows-per-seed 640
```

## Validation command

```bash
python scripts/probes/run_d99_recurrent_routing_microcircuit_prototype_check.py \
  --out target/pilot_wave/d99_recurrent_routing_microcircuit_prototype
```

## Expected healthy decision

- `decision=d99_recurrent_routing_microcircuit_prototype_confirmed`
- `next=D100_RECURRENT_ROUTING_MICROCIRCUIT_SCALE_CONFIRM`
- `best_fair_recurrent_arm=D99_RECURRENT_HALTING_CONFIDENCE_FAIR`

## Boundary

D99 is only a recurrent routing microcircuit prototype for controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
