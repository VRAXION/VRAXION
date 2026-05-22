# STABLE_LOOP_PHASE_LOCK_111X_CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_CONTRACT

111X is a chassis decision gate after the failed 111 target distillation run and the positive 111R failure analysis.

It answers a narrow architecture question: should the current raw chassis be scaled toward Gemma-like local autoregressive behavior, should it continue only with policy-trace supervision, should it be compared against a transformer baseline before scaling, or should an architecture pivot be evaluated?

111X is not a repair-smoke, not deploy, not productization, not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

## Required Inputs

111X requires:

- `RETENTION_OR_LM_REGRESSION_ANALYSIS_POSITIVE` from 111R.
- Failed 111 artifacts with `RAW_OOD_ACCURACY_NOT_IMPROVED`.
- Positive 110 integrated decoder-policy OOD confirm.
- Positive 109 decoder policy integration.
- Positive 100 open-vocab assistant capability scale.
- Positive 099 bounded local/private clean deploy-ready gate.

The failed 111 target checkpoint is not a release candidate. The bounded release stack and existing checkpoints remain frozen.

## Arms

111X evaluates identical final eval rows across:

- `CURRENT_RAW_BASELINE`
- `FAILED_111_STANDARD_REPLAY_DIAGNOSTIC`
- `REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS`
- `DECODER_POLICY_TRACE_DISTILLATION`
- `SMALL_CAUSAL_TRANSFORMER_BASELINE`
- `INTEGRATED_DECODER_POLICY_REFERENCE`
- `STATIC_OUTPUT_CONTROL`
- `COPY_PROMPT_CONTROL`

All arms must record `eval_row_hash`, `eval_prompt_hash`, and `eval_count`. Mismatch fails as `BASELINE_EVAL_MISMATCH`.

## Hard Guards

Current-chassis final raw eval must not use integrated decoder policy, decoder reference, expected-answer metadata, oracle rerank, policy-stage repair, or a response table. The runner records:

- `integrated_policy_used_during_raw_eval = false`
- `decoder_reference_used_during_raw_eval = false`
- `expected_answer_used_during_eval = false`

The transformer baseline must record parameter count, current-chassis parameter count, tokens seen, steps, optimizer, learning rate, batch size, sequence length, wall-clock seconds, and device. If parameter counts differ by more than 2x, the comparison must be normalized and architecture superiority must not be claimed from raw accuracy alone.

Namespace memorization is a hard gate:

- `namespace_leak_rate <= 0.05`
- `teacher_namespace_copy_rate <= 0.05`
- `case_id_drift_rate <= 0.10`

Retention and LM retention are also gates:

- `bounded_chat_slot_binding_accuracy >= 0.90`
- `finite_label_anchorroute_retention_accuracy >= 0.90`
- `unsupported_refusal_accuracy >= 0.80`
- `fineweb_eval_loss_regression <= 0.50`
- `fineweb_next_byte_accuracy_drop <= 0.10`

`REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS` must record scheduled sampling or rollout-loss usage. `DECODER_POLICY_TRACE_DISTILLATION` may train on policy trace labels, but final eval must be raw autoregressive output and must not use policy trace as an inference oracle.

`STATIC_OUTPUT_CONTROL` and `COPY_PROMPT_CONTROL` must fail; otherwise the eval is too weak.

## Decision Contract

`architecture_decision.json` must include:

- `decision`
- `winning_arm`
- `current_chassis_viable`
- `transformer_baseline_stronger`
- `policy_trace_required`
- `raw_generation_scaling_recommended`
- `architecture_pivot_recommended`
- `next_milestone`
- `evidence_summary`

Allowed decisions:

- `current_chassis_remains_viable`
- `architecture_comparison_needed_before_scaling`
- `current_chassis_viable_only_with_policy_trace`
- `architecture_pivot_recommended`
- `no_viable_raw_chassis_found`

Allowed next milestones:

- `112_CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM`
- `112_ARCHITECTURE_BASELINE_COMPARISON_SCALE`
- `112_POLICY_TRACE_DISTILLATION_SCALE_CONFIRM`
- `112_ARCHITECTURE_PIVOT_EVALUATION`
- `111Y_FOUNDATION_OBJECTIVE_FAILURE_ANALYSIS`

Partial improvement is not success unless the architecture decision is explicit and evidence-linked.

## Required Artifacts

111X writes under `target/pilot_wave/stable_loop_phase_lock_111x_chassis_decision_raw_generation_redesign_gate/`:

- `queue.json`
- `progress.jsonl`
- `decision_config.json`
- upstream manifests
- `bounded_release_integrity_manifest.json`
- `train_dataset_manifest.json`
- `eval_dataset_manifest.json`
- `namespace_audit.json`
- `arm_training_metrics.jsonl`
- `arm_eval_results.jsonl`
- `arm_comparison.json`
- `architecture_decision.json`
- `retention_report.json`
- `collapse_metrics.json`
- `fineweb_retention_report.json`
- `transformer_baseline_fairness.json`
- `human_readable_samples.jsonl`
- `failure_case_samples.jsonl`
- `summary.json`
- `report.md`

`progress.jsonl`, `summary.json`, `report.md`, `arm_training_metrics.jsonl`, and `arm_eval_results.jsonl` are refreshed throughout the run so the gate is never a black-box run.

## Verdicts

Positive verdicts include:

- `CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_POSITIVE`
- `UPSTREAM_111R_ANALYSIS_VERIFIED`
- `NAMESPACE_MEMORIZATION_REJECTED`
- `RAW_OBJECTIVE_REDESIGN_EVALUATED`
- `POLICY_TRACE_DISTILLATION_EVALUATED`
- `SMALL_TRANSFORMER_BASELINE_EVALUATED`
- `ARCHITECTURE_DECISION_WRITTEN`
- `RETENTION_PASSES`
- `COLLAPSE_REJECTED`
- `BOUNDED_RELEASE_UNCHANGED`
- `PRODUCTION_CHAT_NOT_CLAIMED`
- `GPT_LIKE_READINESS_NOT_CLAIMED`

Failure verdicts include:

- `CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_FAILS`
- `UPSTREAM_111R_ARTIFACT_MISSING`
- `UPSTREAM_STACK_NOT_POSITIVE`
- `TRAIN_EVAL_LEAKAGE_DETECTED`
- `NAMESPACE_MEMORIZATION_DETECTED`
- `RAW_OBJECTIVE_REDESIGN_INCOMPLETE`
- `POLICY_TRACE_USED_AS_INFERENCE_ORACLE`
- `STATIC_OR_COPY_CONTROL_UNEXPECTED_PASS`
- `TASK_TOO_EASY`
- `RETENTION_REGRESSION_DETECTED`
- `LM_RETENTION_REGRESSION_DETECTED`
- `BOUNDED_RELEASE_MUTATION_DETECTED`
- `SOURCE_CHECKPOINT_MUTATION_DETECTED`
- `ARCHITECTURE_BASELINE_FAIRNESS_MISSING`
- `ARCHITECTURE_DECISION_MISSING`
- `ORACLE_SHORTCUT_DETECTED`
- `INTEGRATED_POLICY_USED_DURING_RAW_EVAL`
- `GPT_LIKE_READINESS_FALSE_CLAIM`
- `PRODUCTION_CHAT_CLAIM_DETECTED`
