# DECK_LOCAL_LEARN_INFER_STRESS_002 Result

`DECK_LOCAL_LEARN_INFER_STRESS_002` extends the Deck-local learn/infer smoke with multi-seed stress runs across seed count, model capacity, data coverage, epoch budget, and harder heldout surface templates.

This is not the official `100/101` artifact-gated path. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not hosted SaaS, not deployment readiness, and not safety alignment.

## Question

```text
Does the Deck-local bounded learner reliably train and infer,
and where does it break under data/capacity/surface stress?
```

## Stage 1: Baseline And Coverage

Generated artifacts:

```text
target/pilot_wave/deck_local_learn_infer_stress_002/
```

Aggregate:

```text
default_neg_aug:
  count: 10
  positive: 10
  min_acc: 1.000
  mean_acc: 1.000
  min_margin_vs_random: 0.900
  min_margin_vs_static: 0.9166666666666666

default_no_neg:
  count: 5
  positive: 5
  min_acc: 0.875
  mean_acc: 0.875

h64_r4_e180_neg_aug:
  count: 5
  positive: 5
  min_acc: 0.9375
  mean_acc: 0.9525

h32_r4_e180_neg_aug:
  count: 5
  positive: 5
  min_acc: 0.9375
  mean_acc: 0.9663888888888889

h64_r4_e80_neg_aug:
  count: 5
  positive: 5
  min_acc: 0.9375
  mean_acc: 0.955

h64_r2_e180_neg_aug:
  count: 5
  positive: 2
  failed: 3
  min_acc: 0.7708333333333334
  mean_acc: 0.7963888888888889
```

Interpretation:

```text
The learner is stable with represented negation/distractor coverage.
The first clear breaking point is low data coverage: train_repeats=2.
Extra epochs do not compensate for missing coverage.
```

## Stage 2: Minimum Stable Band

Generated artifacts:

```text
target/pilot_wave/deck_local_learn_infer_stress_002_stage2/
```

Aggregate:

```text
h64_r3_e120_neg_aug:
  count: 10
  positive: 10
  min_acc: 0.8666666666666667
  mean_acc: 0.8820833333333333

h64_r3_e180_neg_aug:
  count: 5
  positive: 5
  min_acc: 0.8666666666666667
  mean_acc: 0.8830555555555556

h32_r3_e180_neg_aug:
  count: 5
  positive: 5
  min_acc: 0.8541666666666666
  mean_acc: 0.8875

h16_r4_e180_neg_aug:
  count: 5
  positive: 5
  min_acc: 0.9375
  mean_acc: 0.9683333333333334

h64_r4_e40_neg_aug:
  count: 5
  positive: 5
  min_acc: 0.9375
  mean_acc: 0.9508333333333333
```

Interpretation:

```text
train_repeats=3 is above the pass threshold but weaker.
train_repeats=4 is robust even with lower epoch budgets.
Small hidden sizes still learn the toy task when coverage is sufficient.
```

## Stage 3: Hard Surface Shift

Generated artifacts:

```text
target/pilot_wave/deck_local_learn_infer_stress_003_hard_surface/
```

Hard heldout templates added:

```text
TARGET={target}; DISTRACTOR={distractor}
selected_code:{target}; rejected_code:{distractor}
do not answer {distractor}; permitted answer is {target}
valid option {target}; invalid option {distractor}
```

Aggregate:

```text
neg_aug_hard_eval_no_hard_train:
  count: 5
  positive: 5
  min_acc: 0.9375
  mean_acc: 0.9375

neg_hard_train_hard_eval:
  count: 10
  positive: 10
  min_acc: 0.8541666666666666
  mean_acc: 0.8864583333333333

h64_r4_hard_train_hard_eval:
  count: 5
  positive: 2
  failed: 3
  min_acc: 0.7708333333333334
  mean_acc: 0.7976388888888889
```

Failure-map note:

```text
Without hard-surface train coverage, failures concentrated on hard_eval_valid_invalid.
Adding hard templates without enough repeat coverage diluted the base template coverage and hurt accuracy.
```

## Stage 4: Balanced Hard Surface Coverage

Generated artifacts:

```text
target/pilot_wave/deck_local_learn_infer_stress_004_hard_surface_balanced/
```

The script added a matching train template for `valid option {target}; invalid option {distractor}` and raised repeat coverage.

Aggregate:

```text
h128_r12_e260_hard_balanced:
  count: 10
  positive: 10
  min_acc: 0.9625
  mean_acc: 0.9809722222222222
  max_acc: 0.9930555555555556

h64_r12_e120_hard_balanced:
  count: 5
  positive: 5
  min_acc: 0.975
  mean_acc: 0.9783333333333333

h64_r8_e120_hard_balanced:
  count: 5
  positive: 5
  min_acc: 0.9166666666666666
  mean_acc: 0.9388888888888889
```

Interpretation:

```text
Balanced phenomenon coverage substantially repairs hard surface-shift performance.
Coverage quality mattered more than simply adding more templates.
With enough balanced coverage, the stress task is robust across seeds and capacity settings.
```

## Overall Verdict

```text
DECK_LOCAL_LEARN_INFER_STRESS_POSITIVE
MODEL_LEARNS_RELIABLY_ON_DECK
COVERAGE_GAP_IDENTIFIED
DATA_COVERAGE_MORE_IMPORTANT_THAN_EXTRA_EPOCHS
HARD_SURFACE_REPAIRED_BY_BALANCED_COVERAGE
CONTROLS_BEATEN
CLAIM_BOUNDARY_PRESERVED
```

## Boundary

This result supports a bounded Deck-local training/inference pipeline. It does not prove general language understanding, GPT-like assistant readiness, open-domain assistant readiness, production chat, hosted SaaS readiness, public API readiness, deployment readiness, or safety alignment.
