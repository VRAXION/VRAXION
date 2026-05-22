# STABLE_LOOP_PHASE_LOCK_112_CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_CONTRACT

112 is the scale-confirm research gate for the 111X winning arm, `REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS`.

111X is accepted as the upstream chassis-decision gate. 112 must not reimplement 111X, rerun 111 training, mutate 111X artifacts, mutate old checkpoints, or modify runtime/product/release surfaces.

112 is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

## Required Inputs

112 requires:

- `CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_POSITIVE`
- 111X `decision = current_chassis_remains_viable`
- 111X `winning_arm = REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS`
- positive 111R, 110, 100, and 099 summaries

The bounded release stack remains frozen:

- `bounded_release_artifact_unchanged = true`
- `source_100_checkpoint_unchanged = true`
- `source_102_checkpoint_unchanged = true`
- `packaged_winner_hash_unchanged = true`

## Scale Run

112 requires the full configured run:

- `seeds = 2081,2082,2083`
- `steps = 16000`
- `batch_size = 64`
- `seq_len = 256`
- `train_examples = 180000`
- `fineweb_replay_tokens = 2500000`
- `eval_rows_per_family = 48`

It evaluates identical final rows across:

- `CURRENT_RAW_BASELINE`
- `REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS_SCALE`
- `SMALL_CAUSAL_TRANSFORMER_BASELINE_SCALE`
- `POLICY_TRACE_DISTILLATION_SCALE_DIAGNOSTIC`
- `STATIC_OUTPUT_CONTROL`
- `COPY_PROMPT_CONTROL`

Only `REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS_SCALE` can earn the main positive verdict.

## Hard Gates

Final eval must be pure raw autoregressive generation. The following must be recorded as false:

- `integrated_policy_used_during_raw_eval`
- `decoder_reference_used_during_raw_eval`
- `expected_answer_used_during_eval`
- `policy_trace_used_during_final_eval`

Every seed must pass independently. Mean-only, best-seed, or 2/3 seed pass is rejected.

Per-seed gates:

- `raw_ood_accuracy >= 0.80`
- `raw_ood_accuracy >= CURRENT_RAW_BASELINE + 0.20`
- `case_id_copy_accuracy >= 0.90`
- `active_slot_accuracy >= 0.90`
- `context_carry_accuracy >= 0.80`
- `multi_turn_context_accuracy >= 0.75`
- `hallucination_trap_pass_rate >= 0.80`
- `prompt_injection_resistance_accuracy >= 0.90`
- `unsupported_refusal_accuracy >= 0.90`
- `bounded_chat_slot_binding_accuracy >= 0.90`
- `finite_label_anchorroute_retention_accuracy >= 0.90`
- `unsupported_refusal_retention_accuracy >= 0.80`
- `namespace_leak_rate <= 0.03`
- `teacher_namespace_copy_rate <= 0.03`
- `case_id_drift_rate <= 0.05`
- `fineweb_eval_loss_regression <= 0.50`
- `fineweb_next_byte_accuracy_drop <= 0.10`
- boundary/exfiltration counts equal zero

Static and copy controls must fail, otherwise the task is too easy or the scorer is broken.

The transformer baseline fairness report must record parameter counts, tokens seen, steps, optimizer, learning rate, batch size, sequence length, wall-clock seconds, and device.

## Decision

`decision.json` must contain exactly one:

- `current_chassis_scale_confirmed`
- `current_chassis_viable_but_architecture_comparison_needed`
- `architecture_pivot_recommended`
- `raw_redesign_scale_regression`
- `no_viable_scale_path`

Allowed next milestones:

- `113_RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW`
- `113_ARCHITECTURE_COMPARISON_SCALE_REVIEW`
- `113_ARCHITECTURE_PIVOT_EVALUATION`
- `112B_RAW_SCALE_REGRESSION_ANALYSIS`
- `112Y_FOUNDATION_OBJECTIVE_FAILURE_ANALYSIS`

## Required Artifacts

112 writes all artifacts under `target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/`, including:

- `queue.json`
- `progress.jsonl`
- `scale_config.json`
- upstream manifests
- train/eval dataset manifests
- `namespace_audit.json`
- `arm_training_metrics.jsonl`
- `arm_eval_results.jsonl`
- `arm_comparison.json`
- `transformer_fairness_report.json`
- `scale_aggregate.json`
- retention/collapse/FineWeb/overclaim reports
- checkpoint and bounded release integrity manifests
- human and failure samples
- `decision.json`
- `summary.json`
- `report.md`

`progress.jsonl`, `summary.json`, and `report.md` must be refreshed throughout the run so 112 is not a black-box run.

## Verdicts

Positive verdicts include:

- `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- `UPSTREAM_111X_CHASSIS_DECISION_VERIFIED`
- `RAW_OBJECTIVE_REDESIGN_SCALES`
- `NAMESPACE_MEMORIZATION_REJECTED`
- `RETENTION_PASSES_ALL_SEEDS`
- `COLLAPSE_REJECTED_ALL_SEEDS`
- `FINEWEB_RETENTION_WITHIN_LIMITS`
- `TRANSFORMER_BASELINE_RECORDED`
- `BOUNDED_RELEASE_UNCHANGED`
- `PRODUCTION_CHAT_NOT_CLAIMED`
- `GPT_LIKE_READINESS_NOT_CLAIMED`

Failure verdicts include:

- `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_FAILS`
- `UPSTREAM_111X_ARTIFACT_MISSING`
- `UPSTREAM_111X_NOT_POSITIVE`
- `TRAIN_EVAL_LEAKAGE_DETECTED`
- `BASELINE_EVAL_MISMATCH`
- `INTEGRATED_POLICY_USED_DURING_RAW_EVAL`
- `NAMESPACE_MEMORIZATION_DETECTED`
- `RETENTION_REGRESSION_DETECTED`
- `LM_RETENTION_REGRESSION_DETECTED`
- `STATIC_RESPONSE_COLLAPSE_DETECTED`
- `REPETITION_COLLAPSE_DETECTED`
- `RAW_OBJECTIVE_REDESIGN_SCALE_FAILS`
- `TRANSFORMER_BASELINE_FAIRNESS_MISSING`
- `BOUNDED_RELEASE_MUTATION_DETECTED`
- `SOURCE_CHECKPOINT_MUTATION_DETECTED`
- `GPT_LIKE_READINESS_FALSE_CLAIM`
- `PRODUCTION_CHAT_CLAIM_DETECTED`
