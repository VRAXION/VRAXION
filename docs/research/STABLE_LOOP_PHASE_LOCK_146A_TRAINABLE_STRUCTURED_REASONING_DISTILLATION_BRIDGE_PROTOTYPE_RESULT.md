# STABLE_LOOP_PHASE_LOCK_146A_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE Result

146A is the first trainable structured reasoning distillation model-facing bridge after the structured helper stack was scale-confirmed through 145H and selected by 145Z.

Boundary: 146A is constrained model-facing distillation evidence only with canonical structured prompts only; not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Expected Decision

```text
decision = trainable_structured_reasoning_distillation_bridge_prototype_positive
verdict = INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE_POSITIVE
next = 146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM
```

## What 146A Tests

146A tests whether a small deterministic local predictor can imitate the confirmed structured helper scaffold on raw canonical structured text.

The intended bridge is:

```text
confirmed structured helper scaffold
-> controlled supervised curriculum
-> raw canonical structured text model input
-> trainable selected-pocket predictor
-> final value copied from predicted pocket candidate line
```

The target is not free-form language understanding. The prototype does not use natural-language rule prompts and does not claim GPT-like/Gemma-like assistant behavior.

## Raw Input Requirement

The model-facing input is raw canonical structured text only. It may contain rule blocks, priority line, and pocket candidate lines. It must not contain:

```text
selected_pocket_id
winner=pocket_*
final_selected
derived_selected
answer value
gold value
target value
resolved output
expected output
teacher trace fields
per-row oracle metadata
ANSWER=
GOLD=
TARGET=
EXPECTED=
```

Teacher traces are allowed as artifacts, but not as model input features.

## Required Evidence

146A writes curriculum split files, teacher trace manifest, training config, model input audit, value token leakage report, dataset split audit, shortcut scanner report, baseline report, ablation report, evaluation report, oracle shortcut audit, model artifact audit, aggregate metrics, decision, summary, and report.

Key required metrics:

```text
teacher_label_reproduction_accuracy
selected_pocket_prediction_accuracy
final_value_prediction_accuracy
final_value_from_predicted_pocket_accuracy
heldout_template_accuracy
ood_composition_accuracy
candidate_value_permutation_accuracy
oracle_ablation_accuracy
shortcut_scanner_violation_count
train_validation_leakage_count
test_template_overlap_rate
deterministic_replay_passed
```

The model must beat non-oracle baselines by margin and must fail the shuffled-label control.

## Interpretation

A positive 146A means a local trainable predictor can reproduce the structured scaffold's selected-pocket behavior under controlled canonical inputs, with leakage and shortcut audits passing.

It still does not prove natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
