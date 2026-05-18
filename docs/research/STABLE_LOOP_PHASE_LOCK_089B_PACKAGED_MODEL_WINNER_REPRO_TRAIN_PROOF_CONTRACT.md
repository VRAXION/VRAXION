# STABLE_LOOP_PHASE_LOCK_089B_PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF Contract

## Summary

089B is a bounded winner reproducibility proof for the packaged bounded-chat model. It verifies that the 089 packaged checkpoint is hash-bound to the validated 080/083 winner, that the packaged checkpoint passes a fresh eval-only check, that the 080 diversity training path reproduces from 078, that winner arms beat controls on identical eval rows, and that tamper/leakage/shortcut controls fail as expected.

This is bounded winner reproducibility proof only, not GPT-like assistant readiness, not open-domain chat, not full English LM, not production deployment, not safety alignment, and not public release.

## Key Changes

Add only:

```text
scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof.py
scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_089B_PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_089B_PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_RESULT.md
```

Generated outputs go only under:

```text
target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/
```

Do not modify `instnct-core/`, service API, deploy harness, SDK/public exports, `docs/product/`, `docs/releases/`, root `LICENSE`, existing checkpoints, or existing model artifact packages.

## Required Upstreams

Require positive upstreams:

```text
078 CHAT_COMPOSITION_REPAIR_POSITIVE
080 CHAT_COMPOSITION_DIVERSITY_REPAIR_POSITIVE
081 CHAT_DIVERSITY_FRESH_CONFIRM_POSITIVE
082 CHAT_DIVERSITY_MULTI_SEED_CONFIRM_POSITIVE
083 CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE
089 PRIVATE_EVALUATION_RC_PACKAGE_POSITIVE
```

Failure verdicts:

```text
UPSTREAM_ARTIFACT_MISSING
UPSTREAM_STACK_NOT_POSITIVE
```

## Gates

Package hash binding:

```text
083 source_checkpoint_sha256 == 083 packaged_checkpoint_sha256
083 packaged checkpoint file hash == 080 checkpoint file hash
089 packaged_083_artifact_zip_sha256 == 083 artifact_package_zip_sha256
checkpoint bytes unchanged before/after 089B
```

Packaged checkpoint eval:

```text
PACKAGED_089_RC_CHECKPOINT is eval-only
packaged_train_step_count = 0
packaged_checkpoint_hash_before recorded
packaged_checkpoint_hash_after recorded
packaged_checkpoint_hash_unchanged = true
novel_response_rate >= 0.65
template_copy_rate <= 0.25
response_skeleton_reuse_rate <= 0.50
response_skeleton_diversity >= 0.50
slot_binding_accuracy >= 0.90
finite_label_retention_accuracy >= 0.90
empty_output_rate = 0
static_response_rate <= 0.15
repetition_rate <= 0.20
```

Repro training:

```text
fresh repro_child output path
repro_child_started_after_089b_start = true
repro_child_summary_newer_than_089b_start = true
repro_child_report_newer_than_089b_start = true
child_command recorded exactly
child_train_step_count > 0
child_token_train_step_count > 0
child_checkpoint_after_hash != child_checkpoint_before_hash
child_checkpoint_after_hash == upstream 080 model payload hash
full reproduced checkpoint file hash == upstream 080 checkpoint file hash
token_loss_final < token_loss_initial
checkpoint_save_load_pass = true
resume_from_checkpoint_pass = true
```

If the full checkpoint hash does not match, emit `WINNER_NOT_REPRODUCED` and write `deterministic_mismatch_analysis.json` with:

```text
metadata_hash
model_payload_hash
checkpoint_schema_version
timestamp_fields_present
float_serialization_check
key_order_check
payload_hash_matches
metadata_only_mismatch
full_checkpoint_file_sha256
normalized_model_payload_sha256
```

Winner arms:

```text
PACKAGED_089_RC_CHECKPOINT
REPRODUCED_080_DIVERSITY_REPAIR
NO_REPAIR_078_BASELINE
RESPONSE_TABLE_ONLY_CONTROL
ONE_TARGET_PER_PROMPT_CONTROL
NO_SKELETON_DROPOUT_CONTROL
NO_LEXICAL_DROPOUT_CONTROL
NO_CLAUSE_RANDOMIZATION_CONTROL
RANDOM_LABEL_CONTROL
COPY_PROMPT_CONTROL
```

For every arm record:

```text
eval_row_hash
eval_row_count
eval_dataset_sha256
novelty/template/skeleton/slot/retention/collapse metrics
```

Positive requires:

```text
all eval_row_hash identical
all eval_row_count identical
both winner arms pass winner gates
both winners beat RESPONSE_TABLE_ONLY_CONTROL on novel_response_rate by >= 0.30
both winners beat ONE_TARGET_PER_PROMPT_CONTROL on novel_response_rate by >= 0.15
both winners beat NO_SKELETON_DROPOUT_CONTROL on skeleton reuse reduction by >= 0.10
RANDOM_LABEL_CONTROL and COPY_PROMPT_CONTROL fail
```

Tamper and leakage controls must record:

```text
artifact_path
mutation_type
expected_failure
observed_failure
detector_used
```

Required negative controls:

```text
corrupt packaged checkpoint by flipping one byte
wrong 083 artifact hash
exact train/eval prompt overlap fixture
response-table shortcut evidence
```

Failure verdicts:

```text
PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_FAILS
PACKAGE_CHECKPOINT_HASH_MISMATCH
PACKAGED_CHECKPOINT_FRESH_EVAL_FAILS
REPRO_TRAINING_FAILS
TOKEN_OBJECTIVE_NOT_LEARNED
WINNER_NOT_REPRODUCED
STALE_REPRO_ARTIFACT_USED
BASELINE_EVAL_MISMATCH
CONTROL_DELTA_INSUFFICIENT
RANDOM_OR_COPY_CONTROL_UNEXPECTED_PASS
CORRUPTED_CHECKPOINT_UNEXPECTEDLY_ACCEPTED
WRONG_HASH_UNEXPECTEDLY_ACCEPTED
LEAKAGE_CONTROL_UNEXPECTEDLY_ACCEPTED
RESPONSE_TABLE_SHORTCUT_UNEXPECTEDLY_ACCEPTED
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_ON_PACKAGED_CHECKPOINT
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
```

Positive verdicts:

```text
PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_POSITIVE
PACKAGE_HASH_BINDING_VERIFIED
PACKAGED_CHECKPOINT_FRESH_EVAL_PASSES
DETERMINISTIC_REPRO_TRAINING_PASSES
TOKEN_OBJECTIVE_LEARNED
WINNER_BEATS_CONTROLS
BASELINE_EVAL_ROWS_MATCH
TAMPER_CONTROLS_FAIL_AS_EXPECTED
LEAKAGE_CONTROLS_FAIL_AS_EXPECTED
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
NO_TRAINING_ON_PACKAGED_CHECKPOINT
CHECKPOINT_PIPELINE_PASSES
PRODUCTION_CHAT_NOT_CLAIMED
```

## Required Artifacts

Write:

```text
queue.json
progress.jsonl
winner_proof_config.json
upstream_manifest.json
package_hash_binding.json
packaged_checkpoint_eval.json
repro_training_manifest.json
repro_training_metrics.jsonl
deterministic_mismatch_analysis.json
arm_comparison.json
control_delta_report.json
tamper_control_report.json
leakage_control_report.json
eval_row_hashes.json
human_readable_samples.jsonl
failure_case_samples.jsonl
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be written from start and refreshed after upstream verification, package hash binding, packaged checkpoint eval, repro training heartbeat/status, arm comparison, tamper/leakage controls, and final verdict.

`human_readable_samples.jsonl` must include rows from packaged winner, reproduced winner, no-repair baseline, response-table-only control, copy-prompt control, and random-label control.

## Validation

Run:

```text
python -m py_compile scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof.py
python scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof.py --out target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-080-root target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-081-root target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke --upstream-082-root target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke --upstream-083-root target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke --upstream-089-root target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke --seed 2026 --chat-examples 80000 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof_check.py
python scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_088_bounded_chat_long_run_concurrency_resource_stability_check.py --check-only
git diff --check
```

## Assumptions

089B reuses existing 080 and 081 Rust examples as child runners. Full file hash equality remains the hard winner reproduction gate; payload hash only explains failures and does not weaken the gate. If 089B passes, the next strategic milestone is `091_OPEN_VOCAB_CHAT_LM_FOUNDATION`, while `090_BOUNDED_LOCAL_PRIVATE_DEPLOY_READY_GATE` remains a later release-closure gate. If 089B fails, run `089C_WINNER_PROOF_FAILURE_ANALYSIS`.
