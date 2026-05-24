# STABLE_LOOP_PHASE_LOCK_146A_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE Contract

146A is the first trainable structured reasoning distillation bridge prototype after 145Z.

Boundary: 146A is constrained model-facing distillation evidence only with canonical structured prompts only; not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Expected Route

```text
decision = trainable_structured_reasoning_distillation_bridge_prototype_positive
verdict = INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE_POSITIVE
next = 146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM
```

## Scope

146A is executable prototype work, not another planning milestone. It must not modify `scripts/probes/shared_raw_generation_helper.py`.

The prototype uses the confirmed structured helper scaffold as a deterministic teacher to produce a controlled supervised curriculum, then trains and evaluates a small local predictor on raw canonical structured text.

The model-facing input policy is strict:

```text
input = raw canonical structured text
features = hashed character n-grams and token n-grams only
model = deterministic local classifier
primary target = selected_pocket_label
final value = copy candidate value from predicted pocket line
```

No parsed rule blocks, teacher trace, final selected pocket, selected pocket id, expected answer, scorer metadata, or derived helper fields may be passed to the model as features.

## Required Artifacts

Outputs must stay under:

```text
target/pilot_wave/stable_loop_phase_lock_146a_trainable_structured_reasoning_distillation_bridge_prototype/
```

Required artifacts:

```text
queue.json
progress.jsonl
analysis_config.json
upstream_145z_manifest.json
curriculum_train.jsonl
curriculum_validation.jsonl
curriculum_test.jsonl
curriculum_ood_test.jsonl
teacher_trace_manifest.json
training_config.json
model_input_audit.json
value_token_leakage_report.json
dataset_split_audit.json
shortcut_scanner_report.json
baseline_report.json
ablation_report.json
evaluation_report.json
oracle_shortcut_audit.json
model_artifact_audit.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

The runner must write `queue.json` immediately and append `progress.jsonl` throughout curriculum generation, training, and evaluation.

## Data And Leakage Policy

The curriculum must use canonical structured mixed-rule composition prompts with rule blocks, priority line, and candidate value lines. Candidate values must be opaque random tokens that do not encode pocket id, rule type, priority, row id, split id, or answer.

Splits:

```text
train
validation
test
ood_test
```

Required split guarantees:

```text
no row_id overlap
no exact prompt overlap
heldout templates only in test/OOD
OOD contains heldout priority/template/composition patterns
train_validation_leakage_count = 0
test_template_overlap_rate <= 0.05
```

Forbidden in model-facing inputs:

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

Teacher trace fields may appear only in teacher trace and scoring artifacts, never in model input.

## Gates

Required positive gates:

```text
teacher_label_reproduction_accuracy >= 0.80
selected_pocket_prediction_accuracy >= 0.80
final_value_prediction_accuracy >= 0.80
final_value_from_predicted_pocket_accuracy >= 0.80
heldout_template_accuracy >= 0.70
ood_composition_accuracy >= 0.60
candidate_value_permutation_accuracy >= 0.70
oracle_ablation_accuracy <= 0.20
no_priority_ablation_accuracy <= 0.35
shuffled_priority_ablation_accuracy <= 0.35
no_rule_blocks_ablation_accuracy <= 0.35
candidate_value_shuffle_consistency >= 0.70
shortcut_scanner_violation_count = 0
train_validation_leakage_count = 0
test_template_overlap_rate <= 0.05
value_token_contains_pocket_id_rate = 0.0
value_token_contains_rule_type_rate = 0.0
value_token_overlap_train_test_rate = 0.0
deterministic_replay_passed = true
```

The model must beat all non-oracle baselines by at least `0.10` on selected-pocket prediction, and shuffled-label control must fail.

## Claim Limit

A positive 146A proves only limited supervised imitation of the structured helper scaffold under controlled canonical inputs. It does not prove natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
