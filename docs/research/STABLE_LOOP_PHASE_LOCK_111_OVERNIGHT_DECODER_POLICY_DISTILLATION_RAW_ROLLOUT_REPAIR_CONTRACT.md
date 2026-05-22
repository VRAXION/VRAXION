# STABLE_LOOP_PHASE_LOCK_111_OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR

## Summary

111 is target-only overnight research training after positive 110.

110 showed:

```text
raw_ood_stress_accuracy = 0.55648
integrated_ood_stress_accuracy = 1.0
decoder_reference_ood_stress_accuracy = 1.0
decoder_reference_used_rate = 0.0
repair_stage_trace_rate = 0.44352
retention = 1.0
checkpoint unchanged = true
bounded release unchanged = true
```

111 trains new 111 target checkpoint copies from the 102 repair checkpoint so raw generation can internalize the integrated decoder-policy teacher behavior.

111 is not eval-only. It is target-only overnight research training. It does not mutate the bounded release stack, existing checkpoints, service/runtime/deploy surfaces, SDK/public exports, product/release docs, or root `LICENSE`.

111 is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

## Inputs And Scope

Add only:

```text
scripts/probes/run_stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair.py
scripts/probes/run_stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_111_OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_111_OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR_RESULT.md
```

Generated outputs must stay under:

```text
target/pilot_wave/stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair/
```

Require positive upstreams:

```text
110 INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE
109 DECODER_POLICY_INTEGRATION_POSITIVE
108A RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_POSITIVE
100 OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE
099 BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE
```

Record:

```text
human_selected_training_before_package_review = true
```

because 110 mechanically recommended package/boundary review, while 111 is a deliberate human-selected raw training route.

## Default Run

```powershell
python scripts/probes/run_stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair.py --out target/pilot_wave/stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair/smoke --upstream-110-root target/pilot_wave/stable_loop_phase_lock_110_integrated_decoder_policy_ood_confirm_batch/smoke --upstream-109-root target/pilot_wave/stable_loop_phase_lock_109_decoder_policy_integration/smoke --upstream-108a-root target/pilot_wave/stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis/smoke --upstream-100-root target/pilot_wave/stable_loop_phase_lock_100_open_vocab_assistant_capability_scale/smoke --upstream-099-root target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke --teacher-seeds 2052,2053,2054,2055,2056 --train-seeds 2061,2062 --teacher-rows-per-family 48 --eval-rows-per-family 24 --fineweb-replay-tokens 5000000 --distill-examples 250000 --seq-len 256 --batch-size 64 --steps 40000 --min-runtime-minutes 360 --max-runtime-minutes 540 --heartbeat-sec 20
```

CUDA must be used when available. Local expected GPU:

```text
NVIDIA GeForce RTX 4070 Ti SUPER
```

If CUDA is available but not selected without a recorded hardware/runtime error:

```text
CUDA_AVAILABLE_BUT_NOT_USED
```

If median GPU utilization is below 15% without a CPU fallback:

```text
RESOURCE_UNDERUTILIZATION_DETECTED
```

## Training Contract

Use the 109/110 integrated decoder-policy path as a teacher only for target-local training data.

Training mix:

```text
45% integrated decoder-policy teacher outputs
20% FineWeb/FineWeb-Edu replay
15% short instruction / explanation / provided-fact QA
10% bounded chat retention
5% finite-label AnchorRoute retention
5% refusal / boundary / prompt-injection safe outputs
```

Final eval arms use identical eval rows:

```text
PRE_111_RAW_BASELINE
POST_111_RAW_DISTILLED
INTEGRATED_TEACHER_REFERENCE
NO_FINEWEB_REPLAY_CONTROL
NO_RETENTION_MIX_CONTROL
SFT_ONLY_NO_TEACHER_CONTROL
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
```

Final positive eval must score `POST_111_RAW_DISTILLED` as pure raw autoregressive generation. It must not use integrated policy, decoder reference, expected-answer metadata, teacher forcing, oracle rerank, or response-table lookup.

Failures:

```text
INTEGRATED_POLICY_USED_DURING_RAW_EVAL
ORACLE_SHORTCUT_DETECTED
BASELINE_EVAL_MISMATCH
```

## Overnight Runtime

Positive requires a real overnight run:

```text
wall_clock_minutes >= min_runtime_minutes
min_runtime_minutes >= 360
early_finish_prevented = true if initial plan finishes early
extra_batches_launched_if_needed = true if under min runtime
```

Do not satisfy runtime by sleeping. If initial jobs finish early, launch additional target-only distillation batches from the latest 111 checkpoint.

Failure:

```text
OVERNIGHT_RUNTIME_UNDERUSED
```

Required heartbeat artifacts:

```text
progress.jsonl
summary.json
report.md
training_metrics.jsonl
resource_metrics.jsonl
```

Refresh them from start and after upstream verification, dataset build, checkpoint load, training start, every heartbeat, every additional overnight batch, eval start, decision, and final verdict.

## Gates

Positive requires:

```text
target_111_checkpoint_changed = true
train_step_count > 0
optimizer_step_count > 0
train_loss_final < train_loss_initial
post_111_raw_ood_accuracy >= 0.80
post_111_raw_ood_accuracy >= pre_111_raw_ood_accuracy + 0.20
post_111_raw_accuracy_gap_to_integrated_teacher <= 0.15
post_111_raw_per_family_min_accuracy >= 0.65 excluding Hungarian diagnostic
bounded_chat_slot_binding_accuracy >= 0.90
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_retention_accuracy >= 0.80
fineweb_eval_loss_regression <= 0.50
fineweb_next_byte_accuracy_drop <= 0.10
artifact_exfiltration_count = 0
all overclaim counts = 0
collapse rejected
```

Integrity:

```text
source_102_checkpoint_unchanged = true
source_100_checkpoint_unchanged = true
packaged_winner_hash_unchanged = true
bounded_release_artifact_unchanged = true
```

Only the 111 target checkpoint may change.

Leakage:

```text
teacher_eval_exact_prompt_overlap_count = 0
train_eval_exact_prompt_overlap_count = 0
train_eval_exact_response_overlap_count = 0
max_train_eval_prompt_jaccard < 0.90
max_teacher_eval_prompt_jaccard < 0.90
```

Failures include:

```text
TEACHER_DATASET_LEAKAGE_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
RAW_OOD_ACCURACY_NOT_IMPROVED
RAW_TO_INTEGRATED_GAP_REMAINS_HIGH
FINEWEB_RETENTION_REGRESSION_DETECTED
BOUNDED_RETENTION_REGRESSION_DETECTED
SOURCE_CHECKPOINT_MUTATION_DETECTED
PACKAGED_CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
```

## Required Artifacts

Required artifacts:

```text
queue.json
progress.jsonl
overnight_config.json
upstream_110_manifest.json
upstream_109_manifest.json
upstream_108a_manifest.json
upstream_100_manifest.json
upstream_099_manifest.json
bounded_release_integrity_manifest.json
source_checkpoint_manifest.json
teacher_dataset_manifest.json
train_dataset_manifest.json
eval_dataset_manifest.json
train_examples_sample.jsonl
eval_examples_sample.jsonl
training_metrics.jsonl
resource_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
generation_results_pre_raw.jsonl
generation_results_post_raw.jsonl
generation_results_integrated_teacher.jsonl
arm_comparison.json
fineweb_retention_metrics.json
bounded_retention_metrics.json
collapse_metrics.json
overclaim_metrics.json
resource_report.json
human_readable_samples.jsonl
failure_case_samples.jsonl
decision.json
summary.json
report.md
```

`human_readable_samples.jsonl` must include paired rows for:

```text
PRE_111_RAW_BASELINE
POST_111_RAW_DISTILLED
INTEGRATED_TEACHER_REFERENCE
```

Across every final eval family and at least 3 seeds/variants.

## Decision

```text
all gates pass -> 112_RAW_ASSISTANT_MULTI_SEED_OOD_CONFIRM
raw improves but below gates -> 111B_DISTILLATION_PARTIAL_FAILURE_ANALYSIS
FineWeb or bounded retention regresses -> 111R_RETENTION_OR_LM_REGRESSION_ANALYSIS
overnight under-utilizes compute -> 111H_OVERNIGHT_HARNESS_UTILIZATION_FIX
overclaim/exfiltration occurs -> 111C_BOUNDARY_FAILURE_ANALYSIS
```

Positive verdict:

```text
OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR_POSITIVE
```

Positive 111 means a target-only raw checkpoint improved toward the integrated teacher under the 111 gates. It does not mean GPT-like assistant readiness, open-domain assistant readiness, production chat, public API, deployment readiness, or safety alignment.
