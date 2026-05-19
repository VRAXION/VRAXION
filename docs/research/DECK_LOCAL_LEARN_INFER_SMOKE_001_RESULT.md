# DECK_LOCAL_LEARN_INFER_SMOKE_001 Result

`DECK_LOCAL_LEARN_INFER_SMOKE_001` is a Deck-local train plus heldout inference smoke.

It is intentionally independent from the missing `099/100/101` target artifact chain. Its narrower question is:

```text
Can a fresh target model learn a bounded assistant slot task on this Deck,
change its checkpoint,
and pass heldout inference controls?
```

## Scope

This smoke is not the `100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE` gate and not the `101_FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP` gate.

It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not hosted SaaS, not deployment readiness, and not safety alignment.

## Runner

```text
scripts/probes/run_deck_local_learn_infer_smoke_001.py
```

Default output:

```text
target/pilot_wave/deck_local_learn_infer_smoke_001/smoke
```

## What It Tests

The runner creates a fresh bounded assistant slot dataset:

```text
prompt with active code + distractor
-> model predicts the active code label
-> generated text is built from the predicted label
```

The response is not selected from a response table. The neural model must infer the label from prompt features.

Required evidence:

```text
train_loss_final < train_loss_initial
eval_loss_after < eval_loss_before
checkpoint_hash_before != checkpoint_hash_after
heldout_inference_accuracy >= 0.80
train_eval_exact_text_overlap_count = 0
train_eval_pair_overlap_count = 0
static baseline beaten
random-label control beaten
collapse/repetition/copy gates pass
```

## Verdicts

Positive verdicts:

```text
DECK_LOCAL_LEARN_INFER_SMOKE_POSITIVE
MODEL_LEARNS_FROM_RANDOM_INIT
HELDOUT_INFERENCE_PASSES
CHECKPOINT_CHANGED
CONTROLS_BEATEN
COLLAPSE_REJECTED
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts:

```text
DECK_LOCAL_LEARN_INFER_SMOKE_FAILS
CHECKPOINT_DID_NOT_CHANGE
TRAIN_LOSS_DID_NOT_DECREASE
EVAL_LOSS_DID_NOT_DECREASE
HELDOUT_INFERENCE_WEAK
STATIC_BASELINE_TOO_CLOSE
RANDOM_LABEL_CONTROL_TOO_CLOSE
TRAIN_EVAL_OVERLAP_DETECTED
GENERATION_FORMAT_FAIL
COLLAPSE_DETECTED
```

## Deck Smoke Result

Command:

```bash
.venv/bin/python scripts/probes/run_deck_local_learn_infer_smoke_001.py \
  --out target/pilot_wave/deck_local_learn_infer_smoke_001/smoke \
  --seed 2026 \
  --epochs 260 \
  --hidden 128 \
  --lr 0.015 \
  --train-repeats 6
```

Status:

```text
positive
```

Verdicts:

```text
DECK_LOCAL_LEARN_INFER_SMOKE_POSITIVE
MODEL_LEARNS_FROM_RANDOM_INIT
HELDOUT_INFERENCE_PASSES
CHECKPOINT_CHANGED
CONTROLS_BEATEN
COLLAPSE_REJECTED
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_CHAT_NOT_CLAIMED
```

Key metrics:

```text
train_loss_initial:                 2.4854655265808105
train_loss_final:                   0.0000003998962938567274
eval_loss_before:                   2.4860787391662598
eval_loss_after:                    0.21733388304710388
heldout_inference_accuracy:         0.875
random_label_control_accuracy:      0.125
static_baseline_accuracy:           0.06666666666666667
checkpoint_changed:                 true
train_eval_exact_text_overlap_count: 0
train_eval_pair_overlap_count:       0
nonempty_generation_rate:            1.0
utf8_valid_generation_rate:          1.0
static_output_rate:                  0.125
repetition_rate:                     0.0
copy_prompt_rate:                    0.0
wall_clock_sec:                      11.457
```

Interpretation:

```text
The Deck runtime can train a fresh bounded target model from random init.
The model changed its checkpoint, reduced train/eval loss, and beat static and random-label controls.
Heldout inference passed at 0.875 with no train/eval text or pair overlap.
```

## Multi-Seed Confirmation

The same smoke was run for five seeds:

```text
2026, 2027, 2028, 2029, 2030
```

Aggregate:

```text
all_positive:          true
min_accuracy:          0.875
mean_accuracy:         0.875
max_accuracy:          0.875
min_margin_vs_random:  0.750
min_margin_vs_static:  0.7916666666666666
checkpoint_changed:    true for all seeds
train/eval text overlap: 0 for all seeds
train/eval pair overlap: 0 for all seeds
max_static_output_rate: 0.125
max_repetition_rate:   0.0
max_copy_prompt_rate:  0.0
```

Failure map:

```text
all failures were in heldout_bounded
pattern: "local bounded answer uses {target}; do not use distractor {distractor}"
```

This isolated the residual 0.875 ceiling to one heldout scope/template family.

## Negation/Distractor Coverage Diagnostic

A second five-seed diagnostic added explicit negation/distractor training templates:

```text
--include-negation-templates
```

The gates and heldout split stayed the same.

Aggregate:

```text
all_positive:          true
min_accuracy:          1.000
mean_accuracy:         1.000
max_accuracy:          1.000
min_margin_vs_random:  0.925
min_margin_vs_static:  0.9166666666666666
```

Interpretation:

```text
The remaining heldout failure was a scope/template coverage gap.
Adding targeted negation/distractor evidence closed the gap across five seeds.
This supports the narrower conclusion that the Deck-local learner can acquire
the bounded slot rule and the distractor-negation variant when that phenomenon
is represented in training.
```

## Boundary

This is a local proof that the Deck runtime can train a fresh bounded model and run heldout inference. It does not replace the official 100/101 artifact-gated path.
