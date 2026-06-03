# D100 Recurrent Routing Microcircuit Scale Confirm Result

## Status

Implemented in `scripts/probes/run_d100_recurrent_routing_microcircuit_scale_confirm.py` with validation in `scripts/probes/run_d100_recurrent_routing_microcircuit_scale_confirm_check.py`.

## Run command

```bash
python scripts/probes/run_d100_recurrent_routing_microcircuit_scale_confirm.py \
  --out target/pilot_wave/d100_recurrent_routing_microcircuit_scale_confirm \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 21001,21002,21003,21004,21005,21006,21007,21008,21009,21010,21011,21012 \
  --train-rows-per-seed 640 \
  --test-rows-per-seed 640 \
  --ood-rows-per-seed 640 \
  --stress-seeds 21101,21102,21103,21104,21105,21106,21107,21108 \
  --stress-rows-per-seed 820
```

## Validation command

```bash
python scripts/probes/run_d100_recurrent_routing_microcircuit_scale_confirm_check.py \
  --out target/pilot_wave/d100_recurrent_routing_microcircuit_scale_confirm
```

## Expected healthy decision

- `decision=d100_recurrent_routing_microcircuit_scale_confirmed`
- `next=D101_AUTO_ANNEAL_AND_SPARSE_STABILIZATION_PREP`
- `best_fair_recurrent_arm=D99_RECURRENT_HALTING_CONFIDENCE_FAIR_SCALE`

## Boundary

D100 is only a recurrent routing microcircuit scale-confirmation run for controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
