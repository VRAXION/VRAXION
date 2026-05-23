# STABLE_LOOP_PHASE_LOCK_141F_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRM

141F is the scale confirm after the positive 141A multi-field transfer smoke.
It confirms multi-field final selection under controlled helper manifest.
141F is executable helper-only evaluation, so the runner may call
`shared_raw_generation_helper.raw_generate`; the checker must not call
`raw_generate`.

This is not a new capability design, not open-ended reasoning, not general
composition, not GPT-like readiness, not open-domain reasoning, not broad
assistant capability, not production/public API/deployment/safety readiness, and
not general architecture superiority.

Boundary phrases: not broad assistant capability; not general composition.

Required upstream:

```text
141A decision = instnct_pocket_gated_multi_field_transfer_probe_positive
141A next = 141F_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRM
141A main_final_answer_accuracy = 1.0
141A main_multi_field_binding_accuracy = 1.0
141A main_pocket_writeback_rate = 1.0
141A ablation_final_answer_accuracy = 0.0
141A pocket_ablation_delta = 1.0
141A shortcut and bypass rates = 0.0
141A deterministic_replay_passed = true
```

Scale defaults:

```text
seeds = 4201,4202,4203,4204
families = 6
groups_per_family = 24
group_size = 5
eval_rows = 2880
scaffold_variant_count >= 72
max_new_tokens = 96
```

Canonical aggregate metric names:

```text
direct_pocket_value_marker_rate
pocket_ablation_delta_final_answer_accuracy
main_final_answer_accuracy
main_multi_field_binding_accuracy
main_pocket_writeback_rate
priority_conflict_wrong_field_rate
```

Accepted aliases must be normalized into canonical fields:

```text
direct_POCKET_VALUE_rate -> direct_pocket_value_marker_rate
pocket_ablation_delta -> pocket_ablation_delta_final_answer_accuracy
main_final_accuracy -> main_final_answer_accuracy
main_binding_accuracy -> main_multi_field_binding_accuracy
```

Positive requires all metric gates, infrastructure gates, and per-seed gates to
pass. Each family must also independently pass final accuracy, multi-field
binding, writeback, and zero single-field shortcut gates. The single-field
shortcut rate must stay zero. Required controls must fail:

- FIELD_A_ONLY_CONTROL
- FIELD_B_ONLY_CONTROL
- INTERMEDIATE_COPY_CONTROL
- VISIBLE_TARGET_BYPASS_CONTROL
- NOISY_DISTRACTOR_CONTROL
- CLOSED_POCKET_ABLATION_CONTROL
- SINGLE_FIELD_SHORTCUT_CONTROL
- PRIORITY_CONFLICT_WRONG_FIELD_CONTROL
- PREFIX_ONLY_CONTROL

If positive:

```text
decision = instnct_pocket_gated_multi_field_transfer_scale_confirmed
verdict = INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRMED
next = 141Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_NEXT_DECISION_PLAN
```

Clean negative routes:

- single_field_shortcut_detected -> 141B_SINGLE_FIELD_SHORTCUT_ANALYSIS
- multi_field_binding_scale_failure -> 141C_MULTI_FIELD_BINDING_FAILURE_ANALYSIS
- pocket_ablation_not_decision_critical -> 141D_POCKET_CAUSALITY_FAILURE_ANALYSIS
- priority_conflict_failure -> 141E_PRIORITY_CONFLICT_FAILURE_ANALYSIS
- scale_instability_detected -> 141FS_MULTI_FIELD_SCALE_INSTABILITY_ANALYSIS
- helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL

Boundaries:

- no training
- no source checkpoint mutation
- no helper/backend modification
- no public request-key change
- runner helper requests use only prompt, checkpoint_path, checkpoint_hash, seed, max_new_tokens, generation_config
- checker does not call raw_generate
- no runtime/release/product/deploy changes
- no root `LICENSE` change
- all broad capability/readiness flags remain false

Required hardening artifacts:

- helper_request_audit.json
- canonical_metric_alias_report.json
- per_seed_gate_report.json
- per_family_gate_report.json
