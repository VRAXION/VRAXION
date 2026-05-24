# STABLE_LOOP_PHASE_LOCK_146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM Contract

146H is a scale-confirm milestone after the positive 146A trainable structured reasoning distillation bridge prototype.

Boundary: 146H is constrained model-facing distillation evidence only with canonical structured prompts only; not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Expected Route

```text
decision = trainable_structured_reasoning_distillation_bridge_scale_confirmed
verdict = INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRMED
next = 146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN
```

## Scope

146H must not modify `shared_raw_generation_helper.py`, introduce a new model architecture, call external APIs, download models, mutate checkpoints, or extend helper behavior. It reuses the accepted 146A raw-text model family:

```text
input = raw canonical structured text
features = hashed character n-grams and token n-grams only
model = deterministic local classifier
primary target = selected_pocket_label
final value = copy candidate value from predicted pocket line
```

Teacher traces and parsed symbolic fields may exist in teacher/scoring artifacts, but they are forbidden as model-facing inputs and forbidden as feature-extractor inputs.

## Upstream Requirement

146H requires 146A prototype-positive evidence:

```text
decision = trainable_structured_reasoning_distillation_bridge_prototype_positive
verdict = INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE_POSITIVE
next = 146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM
selected_pocket_prediction_accuracy >= 0.90
final_value_from_predicted_pocket_accuracy >= 0.90
heldout_template_accuracy = 1.0
ood_composition_accuracy >= 0.70
shortcut_scanner_violation_count = 0
train_validation_leakage_count = 0
value_token_overlap_train_test_rate = 0.0
deterministic_replay_passed = true
```

## Scale Defaults

```text
seeds = 5501,5502,5503,5504
train_rows_per_seed = 2400
validation_rows_per_seed = 600
test_rows_per_seed = 600
ood_rows_per_seed = 600
heartbeat_sec = 20
```

The runner writes `queue.json` immediately and appends `progress.jsonl` during startup, upstream verification, per-seed curriculum generation, training, evaluation, combined writeout, and final decision.

## Required Artifacts

```text
model_feature_audit.json
feature_path_audit.json
same_model_family_audit.json
baseline_margin_report.json
per_seed_gate_report.json
per_family_ood_report.json
split_stability_report.json
model_input_audit.json
value_token_leakage_report.json
dataset_split_audit.json
shortcut_scanner_report.json
baseline_report.json
ablation_report.json
evaluation_report.json
oracle_shortcut_audit.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

The runner may also write combined curriculum and teacher trace artifacts for auditability.

## Feature Path Audit

`feature_path_audit.json` must prove:

```text
feature_extractor_function_name = token_features
feature_extractor_input_field = model_input
feature_extractor_uses_only_model_input = true
feature_extractor_reads_teacher_trace = false
feature_extractor_reads_selected_pocket_label = false
feature_extractor_reads_final_value_label = false
feature_extractor_reads_candidate_values_as_labels = false
train_X_source_field = model_input
validation_X_source_field = model_input
test_X_source_field = model_input
ood_X_source_field = model_input
```

## Acceptance Gates

```text
selected_pocket_prediction_accuracy >= 0.88
final_value_from_predicted_pocket_accuracy >= 0.88
heldout_template_accuracy >= 0.85
ood_composition_accuracy >= 0.70
minimum_ood_family_accuracy >= 0.50
margin_over_best_baseline >= 0.15
test_margin_over_best_baseline >= 0.15
ood_margin_over_best_baseline >= 0.10
shuffled_label_control_accuracy <= 0.35
shortcut_scanner_violation_count = 0
train_validation_leakage_count = 0
test_template_overlap_rate <= 0.05
value_token_contains_pocket_id_rate = 0.0
value_token_contains_rule_type_rate = 0.0
value_token_overlap_train_test_rate = 0.0
value_token_overlap_train_ood_rate = 0.0
oracle_ablation_accuracy <= 0.20
deterministic_replay_passed = true
```

Per-seed gates require selected-pocket and final-value accuracy at or above `0.85`, OOD accuracy at or above `0.65`, margin over best baseline at or above `0.10`, no shortcut scanner violations, and zero train/test value-token overlap.

Per-family OOD gates require `minimum_ood_family_accuracy >= 0.50`, `collapsed_ood_family_count = 0`, and `no_ood_family_below_minimum = true`.

## Clean Negative Routes

```text
curriculum_generation_failure -> 146B_CURRICULUM_GENERATION_FAILURE_ANALYSIS
train_eval_leakage_detected -> 146C_TRAIN_EVAL_LEAKAGE_ANALYSIS
feature_audit_failure -> 146D_MODEL_SHORTCUT_ANALYSIS
model_shortcut_detected -> 146D_MODEL_SHORTCUT_ANALYSIS
baseline_margin_failure -> 146D_MODEL_SHORTCUT_ANALYSIS
teacher_label_reproduction_failure -> 146E_TEACHER_DISTILLATION_FAILURE_ANALYSIS
ood_generalization_scale_failure -> 146F_OOD_COMPOSITION_FAILURE_ANALYSIS
ood_seed_variance_failure -> 146G_OOD_SEED_VARIANCE_ANALYSIS
helper_stack_regression -> 145B_MIXED_RULE_BLOCK_PARSE_FAILURE_ANALYSIS
```

## Interpretation

A positive 146H proves only that the 146A raw-text trainable distillation bridge remains stable across more seeds with stronger OOD, feature-path, leakage, and baseline-margin audits. It does not prove natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
