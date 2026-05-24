# STABLE_LOOP_PHASE_LOCK_146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN Contract

146Z is a planning-only, artifact-only next-decision milestone after positive 146H.

Boundary: 146Z is constrained model-facing distillation evidence only with canonical structured prompts only; not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Expected Route

```text
decision = lm_style_canonical_structured_text_distillation_prototype_plan_recommended
selected_option = lm_style_canonical_structured_text_distillation_prototype
next = 147A_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE
```

146Z does not retroactively modify 146H. 146H remains accepted, and its hardening requirements are carried forward into the target 147A executable plan.

## Scope

146Z must not train, call `raw_generate`, import `shared_raw_generation_helper`, run torch forward passes, mutate checkpoints, modify helper/runtime/product surfaces, or claim broad capability.

Generated outputs must stay under:

```text
target/pilot_wave/stable_loop_phase_lock_146z_trainable_structured_reasoning_distillation_next_decision_plan/
```

## Upstream Requirement

146Z verifies positive 146H exactly:

```text
decision = trainable_structured_reasoning_distillation_bridge_scale_confirmed
verdict = INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRMED
next = 146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN
selected_pocket_prediction_accuracy = 1.0
final_value_from_predicted_pocket_accuracy = 1.0
heldout_template_accuracy = 1.0
ood_composition_accuracy = 1.0
minimum_ood_family_accuracy = 1.0
margin_over_best_baseline >= 0.58
shortcut_scanner_violation_count = 0
value_token_overlap_train_test_rate = 0.0
value_token_overlap_train_ood_rate = 0.0
deterministic_replay_passed = true
```

## Decision Options

146Z compares:

```text
lm_style_canonical_structured_text_distillation_prototype
scale_raw_text_perceptron_curriculum_further
natural_language_wrapper_before_sequence_model
helper_engine_extension_after_distillation
stop_at_raw_text_distillation_bridge
```

It recommends `lm_style_canonical_structured_text_distillation_prototype` because 146H already scale-confirmed the raw-text perceptron bridge, while LM-style sequence generation of selected labels remains untested.

## Target 147A Contract

`target_147a_milestone_plan.json` must define an implementation-ready plan for:

```text
147A_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE
```

147A intended primitive:

```text
confirmed structured teacher curriculum
-> canonical structured prompt/output text pairs
-> runner-local PyTorch byte-level causal next-byte model
-> generated SELECTED=<label>
-> deterministic candidate-value copy from generated selected label
-> ANSWER=<value>
```

147A uses local PyTorch only. It must not use external APIs, external model downloads, helper modifications, or natural-language input.

Valid generated output schema is exactly one line:

```text
SELECTED=A
SELECTED=B
SELECTED=C
SELECTED=fallback
```

Invalid outputs include:

```text
SELECTED=pocket_a
ANSWER=...
selected_pocket_id=...
winner=pocket_*
extra text before/after the SELECTED line
malformed labels
multiple SELECTED lines
```

147A must not require closed-vocabulary generation of unseen opaque value tokens. Primary learned output is `SELECTED=<A|B|C|fallback>`, and final value accuracy is computed by deterministic candidate-value copy from the input.

## Required 147A Audits

```text
feature_path_audit.json
model_artifact_audit.json
model_input_audit.json
ood_split_definition_report.json
generated_schema_report.json
anti_memorization_report.json
baseline_margin_report.json
shortcut_scanner_report.json
leakage_audit.json
```

`generated_schema_report.json` must track:

```text
generated_output_schema_valid_rate
multiple_selected_line_rate
answer_value_generation_rate
malformed_selected_label_rate
```

`anti_memorization_report.json` must track prompt overlap and heldout template overlap. `ood_split_definition_report.json` must prove heldout templates and relevant priority/block/composition patterns remain OOD.

## 147A Positive Route

```text
decision = lm_style_canonical_structured_text_distillation_prototype_positive
verdict = INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_POSITIVE
next = 147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM
```

## Interpretation

A positive 147A would prove only runner-local byte-level LM-style selected-label generation on canonical structured prompts. It would not prove natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
