# STABLE_LOOP_PHASE_LOCK_122_MULTI_TURN_STATE_REPAIR Result

122 is targeted research repair only. It repairs the post-reasoning multi-turn state breakpoint with raw-only final evaluation. It is not generic SFT, not deploy polish, not an architecture pivot, not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Expected positive result

Expected positive verdicts:

```text
MULTI_TURN_STATE_REPAIR_POSITIVE
UPSTREAM_121_PLAN_VERIFIED
MULTI_TURN_STATE_BREAKPOINT_IMPROVED
RAW_STATE_ROLLOUT_IMPROVED
DEPTH_8_STATE_TRACKING_PASSES
REASONING_REPAIR_PRESERVED
RETENTION_PRESERVED
COLLAPSE_REJECTED
NAMESPACE_MEMORIZATION_REJECTED
STALE_STATE_MEMORIZATION_REJECTED
CONTROLS_FAILED
LEAKAGE_REJECTED
BOUNDED_RELEASE_UNCHANGED
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

Expected decision:

```text
decision = multi_turn_state_repair_success
next = 123_MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM
```

## Required evidence

The result must show:

```text
train_step_count > 0
optimizer_step_count > 0
target_122_checkpoint_changed = true
source_100_checkpoint_unchanged = true
source_102_checkpoint_unchanged = true
bounded_release_artifact_unchanged = true
packaged_winner_hash_unchanged = true
train_loss_final < train_loss_initial
scheduled_sampling_batch_count > 0 OR rollout_loss_batch_count > 0
raw rollout state metrics improve
```

Baseline gap must be reproduced:

```text
pre_multi_turn_state_accuracy recorded
post_multi_turn_state_accuracy recorded
raw_state_accuracy_improvement recorded
```

If the pre baseline is unexpectedly high, the runner must route to:

```text
122A_MULTI_TURN_TARGET_REVALIDATION
```

## Required gates

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

State taxonomy must be separated:

```text
multi_turn_correction_accuracy
active_vs_stale_tracking_accuracy
override_chain_accuracy
slot_update_sequence_accuracy
stale_state_rejection_accuracy
active_slot_after_update_accuracy
```

Reasoning preservation:

```text
tier4_reasoning_accuracy >= 0.97
tier8_reasoning_combo_accuracy >= 0.90
reasoning_failure_rate <= 0.05
```

Retention, namespace, stale-state, collapse, and boundary:

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

Controls must fail:

```text
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
RANDOM_STATE_CONTROL
STALE_STATE_COPY_CONTROL
```

## Failure routes

Failure verdicts include:

```text
MULTI_TURN_STATE_REPAIR_FAILS
UPSTREAM_121_ARTIFACT_MISSING
UPSTREAM_121_NOT_POSITIVE
TRAINING_HELPER_MISSING
TRAIN_EVAL_LEAKAGE_DETECTED
NAMESPACE_MEMORIZATION_DETECTED
STALE_STATE_MEMORIZATION_DETECTED
TEACHER_FORCING_ONLY_SUCCESS_DETECTED
REASONING_REGRESSION_DETECTED
RETENTION_REGRESSION_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
CONTROL_UNEXPECTED_PASS
TASK_TOO_EASY_OR_SCORER_WEAK
OVERCLAIM_DETECTED
ARTIFACT_EXFILTRATION_DETECTED
CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
ROOT_LICENSE_CHANGED
```

## Validation

Run:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_122_multi_turn_state_repair.py
python scripts/probes/run_stable_loop_phase_lock_122_multi_turn_state_repair.py --out target/pilot_wave/stable_loop_phase_lock_122_multi_turn_state_repair/smoke --upstream-121-root target/pilot_wave/stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan/smoke --upstream-120-root target/pilot_wave/stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap/smoke --upstream-119-root target/pilot_wave/stable_loop_phase_lock_119_reasoning_repair_scale_confirm/smoke --upstream-118-root target/pilot_wave/stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair/smoke --upstream-112-root target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/smoke --upstream-099-root target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke --seeds 2151,2152,2153 --steps 12000 --batch-size 64 --seq-len 256 --train-examples 120000 --fineweb-replay-tokens 1000000 --eval-rows-per-family 64 --rollout-eval-every 50 --multi-turn-depths 2,4,6,8 --state-update-variants 8 --stale-decoy-count 6 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_122_multi_turn_state_repair_check.py
python scripts/probes/run_stable_loop_phase_lock_122_multi_turn_state_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap_check.py --check-only
git diff --check
```

