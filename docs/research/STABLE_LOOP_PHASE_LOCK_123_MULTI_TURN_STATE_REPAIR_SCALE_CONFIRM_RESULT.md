# STABLE_LOOP_PHASE_LOCK_123_MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM Result

123 is an eval-only scale confirmation for the 122 multi-turn state repair. It evaluates the 122 repaired raw checkpoint read-only on fresh, larger, multi-seed multi-turn state rows. It performs no training, no repair, no checkpoint mutation, no service startup, no deployment smoke, and no runtime/product/release integration. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Expected positive result

Expected positive verdicts:

```text
MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_POSITIVE
UPSTREAM_122_STATE_REPAIR_VERIFIED
MULTI_TURN_STATE_REPAIR_GENERALIZES
DEPTH_8_STATE_TRACKING_CONFIRMED
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
decision = multi_turn_state_repair_scale_confirmed
next = 124_POST_STATE_REPAIR_CEILING_AND_GAP_REMAP
```

## Required evidence

The result must show that the full configured run was used:

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

The result must also show 122 checkpoint read-only provenance:

```text
repaired_checkpoint_path
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
```

## Required gates

Every seed must pass independently:

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
nonempty_generation_rate >= 0.98
utf8_valid_generation_rate >= 0.80
artifact_exfiltration_count = 0
all overclaim counts = 0
```

Controls must fail:

```text
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
RANDOM_STATE_CONTROL
STALE_STATE_COPY_CONTROL
RANDOM_SLOT_CONTROL
```

The final eval must be pure raw generation. These must be false:

```text
integrated_policy_used_during_final_eval
decoder_reference_used_during_final_eval
teacher_forcing_used_during_final_eval
expected_answer_used_during_eval
oracle_rerank_used
verifier_rerank_used
llm_judge_used
```

## Failure routes

Failure verdicts include:

```text
MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_FAILS
UPSTREAM_122_ARTIFACT_MISSING
UPSTREAM_122_NOT_POSITIVE
CHECKPOINT_PROVENANCE_MISSING
CHECKPOINT_MUTATION_DETECTED
MULTI_SEED_STATE_INSTABILITY_DETECTED
STATE_REPAIR_DOES_NOT_GENERALIZE
DEPTH_8_STATE_REGRESSION_DETECTED
REASONING_REGRESSION_DETECTED
RETENTION_REGRESSION_DETECTED
NAMESPACE_MEMORIZATION_DETECTED
STALE_STATE_MEMORIZATION_DETECTED
CONTROL_UNEXPECTED_PASS
TASK_TOO_EASY_OR_SCORER_WEAK
STATE_EVAL_LEAKAGE_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
EMPTY_OUTPUT_COLLAPSE_DETECTED
OVERCLAIM_DETECTED
ARTIFACT_EXFILTRATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
LLM_JUDGE_USED
ORACLE_SHORTCUT_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
ROOT_LICENSE_CHANGED
```

## Validation

Run:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm.py
python scripts/probes/run_stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm.py --out target/pilot_wave/stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm/smoke --upstream-122-root target/pilot_wave/stable_loop_phase_lock_122_multi_turn_state_repair/smoke --upstream-121-root target/pilot_wave/stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan/smoke --upstream-120-root target/pilot_wave/stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap/smoke --upstream-119-root target/pilot_wave/stable_loop_phase_lock_119_reasoning_repair_scale_confirm/smoke --upstream-118-root target/pilot_wave/stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair/smoke --upstream-112-root target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/smoke --upstream-099-root target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke --seeds 2161,2162,2163,2164,2165 --eval-rows-per-family 96 --multi-turn-depths 2,4,6,8 --diagnostic-depths 10,12 --state-update-variants 12 --stale-decoy-count 8 --table-rows 48 --multi-doc-count 6 --long-context-chars 16384 --noise-blocks 16 --format-variants 8 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_122_multi_turn_state_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan_check.py --check-only
git diff --check
```
