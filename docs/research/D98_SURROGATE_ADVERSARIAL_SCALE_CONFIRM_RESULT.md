# D98 Surrogate Adversarial Scale Confirm Result

## Status

Implemented in `scripts/probes/run_d98_surrogate_adversarial_scale_confirm.py` with validation in `scripts/probes/run_d98_surrogate_adversarial_scale_confirm_check.py`.

## Run command

```bash
python scripts/probes/run_d98_surrogate_adversarial_scale_confirm.py \
  --out target/pilot_wave/d98_surrogate_adversarial_scale_confirm \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 19001,19002,19003,19004,19005,19006,19007,19008,19009,19010,19011,19012 \
  --train-rows-per-seed 540 \
  --test-rows-per-seed 540 \
  --ood-rows-per-seed 540 \
  --stress-seeds 19101,19102,19103,19104,19105,19106 \
  --stress-rows-per-seed 720
```

## Validation command

```bash
python scripts/probes/run_d98_surrogate_adversarial_scale_confirm_check.py \
  --out target/pilot_wave/d98_surrogate_adversarial_scale_confirm
```

## Expected healthy decision

- `decision=d98_surrogate_adversarial_scale_confirmed`
- `next=D99_RECURRENT_ROUTING_MICROCIRCUIT_PROTOTYPE`
- `best_fair_surrogate_arm=SURROGATE_SMALL_MLP_FAIR`

## Boundary

D98 is only a surrogate adversarial scale-confirmation run for controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
