# STABLE_LOOP_PHASE_LOCK_110_INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_RESULT

## Status

Pending local execution.

The expected smoke output root is:

```text
target/pilot_wave/stable_loop_phase_lock_110_integrated_decoder_policy_ood_confirm_batch/smoke
```

## Result Contract

The 110 result is valid only when `summary.json` emits:

```text
INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE
```

The result must report separate path metrics for:

```text
RAW_FREE_GENERATION
DECODER_REPAIRED_REFERENCE
INTEGRATED_DECODER_POLICY_GENERATION
```

`PATH_METRICS_MERGED` is invalid. The result must keep raw, decoder-reference, and integrated metrics separate:

```text
raw_ood_stress_accuracy
decoder_reference_ood_stress_accuracy
integrated_ood_stress_accuracy
raw_vs_integrated_gap
integrated_vs_decoder_reference_gap
```

## Required Positive Evidence

The positive result must show:

```text
all seeds passed independently
integrated_ood_stress_accuracy >= 0.90 on every seed
mean_integrated_ood_stress_accuracy >= 0.95
integrated_per_family_min_accuracy >= 0.75 excluding Hungarian diagnostic
integrated_ood_stress_accuracy >= raw_ood_stress_accuracy + 0.25 on every seed
integrated_vs_decoder_reference_gap <= 0.10 on every seed
decoder_reference_used_rate <= 0.10
repair_stage_trace_rate > 0.20
bounded_chat_slot_binding_accuracy >= 0.90
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_retention_accuracy >= 0.80
artifact_exfiltration_count = 0
all overclaim counts = 0
collapse rejected
checkpoint_hash_unchanged = true
bounded_release_artifact_unchanged = true
train_step_count = 0
optimizer_step_count = 0
llm_judge_used = false
prediction_oracle_used = false
```

The run must reject:

```text
MULTI_SEED_INTEGRATED_INSTABILITY_DETECTED
FAMILY_SPECIFIC_INTEGRATED_REGRESSION_DETECTED
DECODER_REFERENCE_DEPENDENCE_TOO_HIGH
POLICY_TRACE_MISSING
EVAL_LEAKAGE_DETECTED
EVAL_ROW_MISMATCH
RETENTION_REGRESSION_DETECTED
OVERCLAIM_DETECTED
ARTIFACT_EXFILTRATION_DETECTED
```

## Decision

`decision.json` must choose one conservative next value:

```text
111_INTEGRATED_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW
110B_DECODER_REFERENCE_DEPENDENCE_ANALYSIS
110B_INTEGRATED_OOD_CONFIRM_FAILURE_ANALYSIS
110R_RETENTION_REGRESSION_ANALYSIS
110C_BOUNDARY_OVERCLAIM_OR_EXFILTRATION_FAILURE_ANALYSIS
```

## Boundary

110 is eval-only research confirm. It does not train, repair, mutate checkpoints, or integrate the path into service/runtime/deploy/product surfaces.

110 is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.
