# STABLE_LOOP_PHASE_LOCK_148H_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRM Contract

148H is a scale-confirm milestone after positive 148A. It is a scale confirm only, not a new feature, and must use the same 148A runner-local PyTorch byte-level autoregressive full-line generation setup.

Expected route:

```text
decision = full_selected_line_generation_scale_confirmed
verdict = INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRMED
next = 148Z_FULL_SELECTED_LINE_GENERATION_NEXT_DECISION_PLAN
```

Boundary: constrained model-facing distillation evidence only, canonical structured prompts only, bounded full SELECTED=<label> line generation only; not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Scope

148H must not introduce a new model architecture, helper modification, external model, external API, model download, pretrained weights, natural-language input, or opaque value-token generation requirement. It must not modify `shared_raw_generation_helper.py`.

Generated outputs must stay under:

```text
target/pilot_wave/stable_loop_phase_lock_148h_full_selected_line_generation_scale_confirm/
```

## Required Behavior

The model behavior remains:

```text
canonical structured prompt
-> generation input ends exactly at <OUTPUT>
-> model raw continuation generates SELECTED=<label>
-> strict schema validation from raw generated text
-> deterministic candidate-value copy from generated selected line
```

Forbidden:

```text
fixed SELECTED= prefix in generation input
runner prefix injection
deterministic selected-line wrapper
post-generation repair
substring extraction
casing repair
label repair
constrained label-only decoding
first-byte-only training
```

## Gates

Main gates:

```text
full_selected_line_exact_match_rate >= 0.95
full_line_generation_accuracy >= 0.95
selected_prefix_generation_accuracy >= 0.95
selected_label_generation_accuracy >= 0.95
selected_label_extracted_from_full_line_accuracy >= 0.95
final_value_from_generated_line_accuracy >= 0.95
fallback_full_line_accuracy >= 0.90
generated_output_schema_valid_rate >= 0.95
ood_full_line_accuracy >= 0.85
minimum_ood_family_accuracy >= 0.75
generation_deterministic_replay_passed = true
```

Hardening gates:

```text
eval_generation_input_contains_selected_prefix = false
runner_prepends_selected_prefix = false
deterministic_selected_line_wrapper_used = false
raw_generated_text_stored = true
schema_scored_from_raw_generated_text = true
post_generation_repair_used = false
selected_line_extracted_from_substring = false
casing_repair_used = false
prefix_repair_used = false
label_repair_used = false
extra_text_ignored_for_schema = false
first_byte_only_training_used = false
forced_selected_prefix_used = false
constrained_label_only_decoding_used = false
```
