# STABLE_LOOP_PHASE_LOCK_147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN Contract

147Z is a planning-only, artifact-only next-decision milestone after positive 147H.

Expected route:

```text
decision = full_selected_line_generation_prototype_plan_recommended
selected_option = full_selected_line_generation_prototype
next = 148A_FULL_SELECTED_LINE_GENERATION_PROTOTYPE
```

## Scope

147Z must not train, call `raw_generate`, import `shared_raw_generation_helper`, run torch forward passes, mutate checkpoints, modify helper/runtime/product surfaces, or claim broad capability.

Generated artifacts must stay under:

```text
target/pilot_wave/stable_loop_phase_lock_147z_lm_style_canonical_structured_text_distillation_next_decision_plan/
```

Boundary: constrained model-facing distillation evidence only, canonical structured prompts only, not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Upstream 147H Evidence

147Z must verify the accepted 147H scale-confirm evidence exactly:

```text
decision = lm_style_canonical_structured_text_distillation_scale_confirmed
verdict = INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRMED
next = 147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN
selected_label_generation_accuracy = 1.0
selected_label_byte_accuracy = 1.0
final_value_from_generated_label_accuracy = 1.0
generated_output_schema_valid_rate = 1.0
ood_selected_accuracy = 1.0
minimum_ood_family_accuracy = 1.0
minimum_ood_family_row_count = 140
shuffled_target_control_accuracy = 0.0
generation_deterministic_replay_passed = true
model_generates_full_selected_line = false
schema_prefix_fixed_by_runner = true
selected_line_wrapper_deterministic = true
```

## Decision

147Z compares:

```text
full_selected_line_generation_prototype
multi_token_schema_generation_prototype
controlled_natural_language_wrapper_later_plan
scale_selected_byte_bridge_further
stop_at_selected_label_byte_generation
```

The selected option is `full_selected_line_generation_prototype`, because 147H scale-confirmed only selected label byte/token prediction after a fixed `SELECTED=` prefix. The next honest bridge is bounded raw full `SELECTED=<label>` line generation.

## Target 148A Requirements

148A must be implementation-ready for:

```text
148A_FULL_SELECTED_LINE_GENERATION_PROTOTYPE
```

148A intended primitive:

```text
canonical structured prompt
-> runner-local PyTorch byte-level autoregressive model
-> raw model continuation generates full SELECTED=<label> line
-> strict schema validation from raw generated text
-> deterministic candidate-value copy
```

148A must remove the fixed `SELECTED=` prefix from generation input. The generation input must end exactly with:

```text
<OUTPUT>
```

Forbidden generation input:

```text
<OUTPUT>
SELECTED=
```

The runner must not prepend `SELECTED=`, must not deterministically wrap a label byte, and must not repair malformed output. It may only parse the raw generated continuation and strip a trailing newline.

Valid raw generated output is exactly one line:

```text
SELECTED=A
SELECTED=B
SELECTED=C
SELECTED=fallback
```

Final value remains deterministic copy from the generated selected line. Direct opaque value-token generation remains out of scope.

Required 148A reports include:

```text
generation_prefix_audit.json
raw_generation_audit.json
decoding_audit.json
full_line_generation_report.json
generated_schema_report.json
generation_input_audit.json
anti_memorization_report.json
ood_generation_family_report.json
baseline_margin_report.json
shortcut_scanner_report.json
leakage_audit.json
model_artifact_audit.json
deterministic_replay_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

Key 148A hard gates:

```text
eval_generation_input_ends_with_output_delimiter = true
eval_generation_input_contains_selected_prefix = false
runner_prepends_selected_prefix = false
model_generates_selected_prefix = true
model_generates_full_selected_line = true
deterministic_selected_line_wrapper_used = false
autoregressive_generation_used = true
forced_selected_prefix_used = false
constrained_label_only_decoding_used = false
raw_generated_text_stored = true
schema_scored_from_raw_generated_text = true
post_generation_repair_used = false
selected_line_extracted_from_substring = false
casing_repair_used = false
prefix_repair_used = false
```

## Required Artifacts

147Z must write:

```text
queue.json
progress.jsonl
analysis_config.json
upstream_147h_manifest.json
evidence_chain_summary.json
lm_style_distillation_state_report.json
full_line_generation_gap_analysis.json
next_decision_matrix.json
target_148a_milestone_plan.json
anti_oracle_requirements.json
risk_register.json
decision.json
summary.json
report.md
```
