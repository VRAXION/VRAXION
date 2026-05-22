# STABLE_LOOP_PHASE_LOCK_123_MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM Contract

123 is an eval-only scale confirmation for the 122 multi-turn state repair. It evaluates the 122 repaired raw checkpoint read-only on fresh, larger, multi-seed multi-turn state rows. It performs no training, no repair, no checkpoint mutation, no service startup, no deployment smoke, and no runtime/product/release integration. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Required upstreams

Require positive upstream artifacts:

- 122 `MULTI_TURN_STATE_REPAIR_POSITIVE`
- 121 `TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_POSITIVE`
- 120 `POST_REASONING_CEILING_AND_GAP_REMAP_POSITIVE`
- 119 `REASONING_REPAIR_SCALE_CONFIRM_POSITIVE`
- 118 `REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

122 showed:

```text
post_multi_turn_state_accuracy = 0.9900568181818182
raw_state_accuracy_improvement = 0.33380681818181823
depth_8_state_accuracy = 0.9886363636363636
tier4_reasoning_accuracy = 1.0
tier8_reasoning_combo_accuracy = 1.0
reasoning_failure_rate = 0.0
```

## Required run

The positive result requires the full configured run:

```text
seeds = 2161,2162,2163,2164,2165
eval_rows_per_family = 96
multi_turn_depths = 2,4,6,8
diagnostic_depths = 10,12
state_update_variants = 12
stale_decoy_count = 8
table_rows = 48
multi_doc_count = 6
long_context_chars = 16384
noise_blocks = 16
format_variants = 8
```

No tiny or dev substitute may emit `MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_POSITIVE`.

## Target and controls

Positive-scored arm:

```text
POST_122_MULTI_TURN_STATE_REPAIRED_RAW_SCALE_CONFIRM
```

Diagnostic/control arms:

```text
PRE_122_POST_REASONING_RAW_BASELINE
PRE_STATE_REPAIR_RAW_BASELINE
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
RANDOM_STATE_CONTROL
STALE_STATE_COPY_CONTROL
RANDOM_SLOT_CONTROL
```

Controls must fail:

```text
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
RANDOM_STATE_CONTROL
STALE_STATE_COPY_CONTROL
RANDOM_SLOT_CONTROL
```

## Required safeguards

The 122 repaired checkpoint must be loaded read-only from the 122 target manifest. The runner must record:

```text
repaired_checkpoint_path
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
```

Final eval must be pure raw generation. These must be recorded false:

```text
integrated_policy_used_during_final_eval
decoder_reference_used_during_final_eval
teacher_forcing_used_during_final_eval
expected_answer_used_during_eval
oracle_rerank_used
verifier_rerank_used
llm_judge_used
```

The leakage audit must compare against 112-122 artifacts and require:

```text
exact_prompt_overlap = 0
exact_expected_output_overlap = 0 except counted standard refusal templates
near_duplicate_prompt_count = 0 at token_jaccard >= 0.90
```

## Required gates

Every seed must pass independently. Mean-only, best-seed, and 4-of-5 passes are rejected.

Per-seed state gates:

```text
multi_turn_state_accuracy >= 0.95
multi_turn_state_accuracy >= PRE_122_POST_REASONING_RAW_BASELINE + 0.20
state_tracking_accuracy >= 0.95
multi_turn_correction_accuracy >= 0.95
active_vs_stale_tracking_accuracy >= 0.95
override_chain_accuracy >= 0.92
slot_update_sequence_accuracy >= 0.95
stale_state_rejection_accuracy >= 0.95
active_slot_after_update_accuracy >= 0.95
tier4_multi_turn_breakpoint_accuracy >= 0.95
tier7_state_combo_accuracy >= 0.88
tier8_combined_state_accuracy >= 0.88
state_failure_rate <= 0.05
```

Depth gates:

```text
depth_2_state_accuracy >= 0.95
depth_4_state_accuracy >= 0.95
depth_6_state_accuracy >= 0.95
depth_8_state_accuracy >= 0.90
diagnostic_depth_10_state_accuracy recorded
diagnostic_depth_12_state_accuracy recorded
```

Diagnostic depth 10/12 does not fail positive unless it triggers collapse, retention regression, overclaim, namespace leakage, or stale-state leakage.

Reasoning preservation gates:

```text
tier4_reasoning_accuracy >= 0.97
tier8_reasoning_combo_accuracy >= 0.90
reasoning_failure_rate <= 0.05
```

Retention, namespace, stale-state, collapse, and boundary gates:

```text
bounded_chat_slot_binding_accuracy >= 0.90
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_retention_accuracy >= 0.80
namespace_leak_rate <= 0.03
teacher_namespace_copy_rate <= 0.03
case_id_drift_rate <= 0.05
stale_state_copy_rate <= 0.05
stale_decoy_leak_rate <= 0.05
empty_output_rate <= 0.02
static_output_rate <= 0.10
repetition_rate <= 0.20
copy_prompt_rate <= 0.15
nonempty_generation_rate >= 0.98
utf8_valid_generation_rate >= 0.80
artifact_exfiltration_count = 0
all overclaim counts = 0
```

## Required artifacts

The runner must write:

```text
queue.json
progress.jsonl
eval_config.json
upstream_122_manifest.json
upstream_121_manifest.json
upstream_120_manifest.json
upstream_119_manifest.json
upstream_118_manifest.json
upstream_112_manifest.json
upstream_099_manifest.json
checkpoint_integrity_manifest.json
bounded_release_integrity_manifest.json
state_scale_dataset.jsonl
eval_row_hashes.json
freshness_leakage_audit.json
raw_generation_results.jsonl
control_results.jsonl
per_family_metrics.json
per_seed_metrics.jsonl
depth_metrics.json
aggregate_metrics.json
state_scale_metrics.json
state_depth_report.json
reasoning_preservation_report.json
retention_report.json
collapse_metrics.json
namespace_audit.json
overclaim_exfiltration_report.json
control_arm_report.json
human_readable_samples.jsonl
failure_case_samples.jsonl
decision.json
summary.json
report.md
```

Expected positive decision:

```text
decision = multi_turn_state_repair_scale_confirmed
next = 124_POST_STATE_REPAIR_CEILING_AND_GAP_REMAP
```
