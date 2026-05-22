# STABLE_LOOP_PHASE_LOCK_122_MULTI_TURN_STATE_REPAIR Contract

122 is targeted research repair only. It repairs the post-reasoning multi-turn state breakpoint with raw-only final evaluation. It is not generic SFT, not deploy polish, not an architecture pivot, not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Required upstreams

Require positive upstream artifacts:

- 121 `TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_POSITIVE`
- 120 `POST_REASONING_CEILING_AND_GAP_REMAP_POSITIVE`
- 119 `REASONING_REPAIR_SCALE_CONFIRM_POSITIVE`
- 118 `REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

121 selected:

```text
selected_next_milestone = 122_MULTI_TURN_STATE_REPAIR
selected_repair_target = multi_turn_state_first
first_breakpoint_tier = TIER_4_MULTI_TURN_STATE_UPDATE
primary_next_repair_target = multi_turn_state_failure
reasoning_regression_rejected = true
reasoning_failure_rate = 0.0
```

## Required run

The positive result requires the full configured run:

```text
seeds = 2151,2152,2153
steps = 12000
batch_size = 64
seq_len = 256
train_examples = 120000
eval_rows_per_family = 64
fineweb_replay_tokens = 1000000
rollout_eval_every = 50
multi_turn_depths = 2,4,6,8
state_update_variants = 8
stale_decoy_count = 6
```

No tiny or dev substitute may emit `MULTI_TURN_STATE_REPAIR_POSITIVE`.

## Target and controls

Positive-scored arm:

```text
POST_122_MULTI_TURN_STATE_REPAIRED_RAW
```

Comparison/control arms:

```text
PRE_122_POST_REASONING_RAW_BASELINE
NO_ROLLOUT_OBJECTIVE_CONTROL
GENERAL_SFT_ONLY_CONTROL
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
RANDOM_STATE_CONTROL
STALE_STATE_COPY_CONTROL
```

Controls must fail:

```text
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
RANDOM_STATE_CONTROL
STALE_STATE_COPY_CONTROL
```

## Required safeguards

Final eval must be pure raw autoregressive generation. During final eval these are forbidden and must be recorded false:

```text
integrated_policy_used_during_final_eval
decoder_reference_used_during_final_eval
oracle_rerank_used
expected_answer_used_during_eval
teacher_forcing_used_during_final_eval
verifier_rerank_used
llm_judge_used
```

Scheduled sampling or rollout objective must actually run:

```text
scheduled_sampling_batch_count > 0 OR rollout_loss_batch_count > 0
```

The train/eval and final eval leakage audit must compare against 112-121 artifacts and require:

```text
exact_prompt_overlap = 0
exact_expected_output_overlap = 0 except counted standard refusal templates
near_duplicate_prompt_count = 0 at token_jaccard >= 0.90
```

Source checkpoints and bounded release artifacts remain frozen. Only the target 122 checkpoint under `target/` may change.

## Required gates

Positive requires actual repair:

```text
train_step_count > 0
optimizer_step_count > 0
target_122_checkpoint_changed = true
source_100_checkpoint_unchanged = true
source_102_checkpoint_unchanged = true
bounded_release_artifact_unchanged = true
packaged_winner_hash_unchanged = true
train_loss_final < train_loss_initial
raw rollout state metrics improve
```

Baseline gap must be reproduced:

```text
pre_multi_turn_state_accuracy recorded
post_multi_turn_state_accuracy recorded
raw_state_accuracy_improvement recorded
```

Multi-turn gates:

```text
post_multi_turn_state_accuracy >= 0.95
post_multi_turn_state_accuracy >= pre_multi_turn_state_accuracy + 0.10
post_state_tracking_accuracy >= 0.95
post_multi_turn_correction_accuracy >= 0.95
stale_state_rejection_accuracy >= 0.95
override_chain_accuracy >= 0.92
active_slot_after_update_accuracy >= 0.95
tier4_multi_turn_breakpoint_accuracy >= 0.95
tier7_state_combo_accuracy >= 0.88
tier8_combined_state_accuracy >= 0.88
multi_turn_state_failure_count_post <= 25% of pre
```

Depth gates:

```text
depth_2_state_accuracy recorded
depth_4_state_accuracy recorded
depth_6_state_accuracy recorded
depth_8_state_accuracy recorded
depth_8_state_accuracy >= 0.88
```

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
artifact_exfiltration_count = 0
all overclaim counts = 0
controls_failed = true
```

## Required artifacts

The runner must write:

```text
queue.json
progress.jsonl
repair_config.json
upstream_121_manifest.json
upstream_120_manifest.json
upstream_119_manifest.json
upstream_118_manifest.json
upstream_112_manifest.json
upstream_099_manifest.json
checkpoint_integrity_manifest.json
bounded_release_integrity_manifest.json
train_dataset_manifest.json
eval_dataset_manifest.json
state_repair_dataset.jsonl
eval_row_hashes.json
freshness_leakage_audit.json
namespace_audit.json
arm_training_metrics.jsonl
rollout_eval_metrics.jsonl
raw_generation_results.jsonl
control_results.jsonl
per_family_metrics.json
depth_metrics.json
state_repair_metrics.json
reasoning_preservation_report.json
retention_report.json
collapse_metrics.json
overclaim_exfiltration_report.json
control_arm_report.json
human_readable_samples.jsonl
failure_case_samples.jsonl
decision.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be refreshed at startup, upstream verification, dataset build, leakage audit, each seed train start, training heartbeat, rollout eval heartbeat, each seed final eval, aggregate analysis, decision writing, and final verdict.

