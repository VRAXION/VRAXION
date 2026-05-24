# STABLE_LOOP_PHASE_LOCK_146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM Result

146H scale-confirms the 146A trainable structured reasoning distillation bridge using the same raw canonical structured text model family.

Boundary: 146H is constrained model-facing distillation evidence only with canonical structured prompts only; not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority.

## Expected Decision

```text
decision = trainable_structured_reasoning_distillation_bridge_scale_confirmed
verdict = INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRMED
next = 146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN
```

## What 146H Tests

146H does not add helper behavior and does not add a new model architecture. It repeats the 146A bridge at scale:

```text
canonical structured prompt
-> raw text n-gram features
-> deterministic local selected-pocket predictor
-> final value copied from the predicted pocket candidate line
```

The scale run uses multiple seeds and stronger audits for OOD family stability, feature path integrity, leakage, and baseline margin.

## Required Evidence

146H writes the following decision and audit reports:

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

`feature_path_audit.json` must show that `token_features` receives only `model_input`. Parsed rule blocks, teacher traces, labels, final values, and candidate values as labels are not allowed as feature inputs.

## Acceptance Meaning

A positive result means:

```text
raw canonical structured text predictor
-> selected-pocket imitation
-> final-value copy from predicted pocket
```

is stable across the scale seeds and passes OOD, baseline, leakage, shortcut, and feature-path controls.

## Non-Claims

146H does not claim:

```text
natural-language rule reasoning
open-ended arbitration
GPT-like/Gemma-like assistant capability
production readiness
architecture superiority
```

These limits are part of the result contract and must appear in decision, summary, report, and docs.
