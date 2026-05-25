# STABLE_LOOP_PHASE_LOCK_148A_FULL_SELECTED_LINE_GENERATION_PROTOTYPE Contract

148A is the executable prototype after positive 147Z.

Expected route:

```text
decision = full_selected_line_generation_prototype_positive
verdict = INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_POSITIVE
next = 148H_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRM
```

## Scope

148A proves only bounded full `SELECTED=<label>` line generation from canonical structured prompts.
In plain checker wording, this is full SELECTED=<label> line generation.

Boundary: constrained model-facing distillation evidence only, canonical structured prompts only, not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

147Z remains accepted as the planning milestone. 148A must not retrofit 147Z or 147H.

## Generation Requirement

Evaluation generation input must end exactly at:

```text
<OUTPUT>
```

The model raw continuation must generate exactly one full selected line:

```text
SELECTED=A
SELECTED=B
SELECTED=C
SELECTED=fallback
```

The runner must not provide `SELECTED=`, wrap a predicted label byte, repair malformed output, or score a repaired/generated substring. Final value remains deterministic copy from the generated selected line. Direct opaque value-token generation remains out of scope.

## Required Audits

`generation_prefix_audit.json` must prove:

```text
eval_generation_input_ends_with_output_delimiter = true
eval_generation_input_contains_selected_prefix = false
runner_prepends_selected_prefix = false
deterministic_selected_line_wrapper_used = false
model_generates_selected_prefix = true
model_generates_full_selected_line = true
```

`raw_generation_audit.json` must prove:

```text
raw_generated_text_stored = true
schema_scored_from_raw_generated_text = true
post_generation_repair_used = false
selected_line_extracted_from_substring = false
casing_repair_used = false
prefix_repair_used = false
label_repair_used = false
```

`decoding_audit.json` must prove:

```text
autoregressive_generation_used = true
full_selected_line_target_used = true
first_byte_only_training_used = false
forced_selected_prefix_used = false
constrained_label_only_decoding_used = false
stop_on_newline_or_max_len = true
max_new_bytes present
```

Training target must cover the full selected line:

```text
train_target_sequence = SELECTED=<label>\n
```

not only the first label byte.

## Gates

Core positive gates:

```text
selected_prefix_generation_accuracy >= 0.70
selected_label_generation_accuracy >= 0.70
full_selected_line_exact_match_rate >= 0.70
selected_label_extracted_from_full_line_accuracy >= 0.70
final_value_from_generated_line_accuracy >= 0.70
generated_output_schema_valid_rate >= 0.80
ood_full_line_accuracy >= 0.50
full_line_generation_accuracy >= best_baseline_accuracy + 0.10
```

No wrapper/repair gates:

```text
eval_generation_input_contains_selected_prefix = false
runner_prepends_selected_prefix = false
deterministic_selected_line_wrapper_used = false
post_generation_repair_used = false
selected_line_extracted_from_substring = false
casing_repair_used = false
prefix_repair_used = false
label_repair_used = false
```

Per-label and leakage gates:

```text
every label A/B/C/fallback appears in train, validation, test, and OOD
fallback_full_line_accuracy >= 0.40
minimum_per_label_full_line_accuracy >= 0.40
answer_value_generation_rate = 0.0
selected_pocket_id_generation_rate = 0.0
multiple_selected_line_rate = 0.0
extra_text_generation_rate <= 0.20
shuffled_target_control_accuracy <= 0.35
shortcut_scanner_violation_count = 0
train_eval_prompt_overlap_count = 0
train_ood_prompt_overlap_count = 0
value_token_overlap_train_test_rate = 0.0
generation_deterministic_replay_passed = true
```
