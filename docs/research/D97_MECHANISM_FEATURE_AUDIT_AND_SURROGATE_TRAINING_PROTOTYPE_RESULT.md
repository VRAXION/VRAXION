# D97 Mechanism Feature Audit and Surrogate Training Prototype Result

## Status

Implemented in `scripts/probes/run_d97_mechanism_feature_audit_and_surrogate_training_prototype.py` with validation in `scripts/probes/run_d97_mechanism_feature_audit_and_surrogate_training_prototype_check.py`.

## Run command

```bash
python scripts/probes/run_d97_mechanism_feature_audit_and_surrogate_training_prototype.py \
  --out target/pilot_wave/d97_mechanism_feature_audit_and_surrogate_training_prototype \
  --workers auto \
  --cpu-target 50-75 \
  --heartbeat-sec 20 \
  --seeds 18001,18002,18003,18004,18005,18006,18007,18008 \
  --train-rows-per-seed 420 \
  --test-rows-per-seed 420 \
  --ood-rows-per-seed 420 \
  --stress-seeds 18101,18102,18103,18104 \
  --stress-rows-per-seed 540
```

## Validation command

```bash
python scripts/probes/run_d97_mechanism_feature_audit_and_surrogate_training_prototype_check.py \
  --out target/pilot_wave/d97_mechanism_feature_audit_and_surrogate_training_prototype
```

## Expected healthy decision

- `decision=d97_surrogate_training_prototype_confirmed`
- `next=D98_SURROGATE_ADVERSARIAL_SCALE_CONFIRM`
- `surrogate_best_fair_arm=SURROGATE_SMALL_MLP_FAIR`

## Boundary

D97 is only a mechanism feature audit and surrogate training prototype for controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
