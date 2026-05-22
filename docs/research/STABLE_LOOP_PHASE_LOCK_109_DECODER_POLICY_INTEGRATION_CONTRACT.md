# STABLE_LOOP_PHASE_LOCK_109_DECODER_POLICY_INTEGRATION_CONTRACT

## Summary

109 is a research-harness decoder policy integration milestone after positive 108A.

108A diagnosed:

```text
raw_decoder_disagreement_count = 108
decoder_success_on_raw_fail_rate = 1.0
raw_vs_decoder_ood_gap = 0.4737
primary_failure_mechanism = context_carry_failure
recommended_next = 109_DECODER_POLICY_INTEGRATION
```

109 tests whether a traceable integrated generation path can close the raw OOD gap by using the decoder policy that already succeeds.

109 performs no training, no checkpoint mutation, no service/runtime/deploy integration, and no product/API changes. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

## Inputs And Paths

Add only:

```text
scripts/probes/run_stable_loop_phase_lock_109_decoder_policy_integration.py
scripts/probes/run_stable_loop_phase_lock_109_decoder_policy_integration_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_109_DECODER_POLICY_INTEGRATION_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_109_DECODER_POLICY_INTEGRATION_RESULT.md
```

Generated outputs must stay under:

```text
target/pilot_wave/stable_loop_phase_lock_109_decoder_policy_integration/
```

Require positive upstreams:

```text
108A RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_POSITIVE
108 OPEN_DOMAIN_ASSISTANT_OOD_STRESS_AND_FAILURE_MAP_BATCH_POSITIVE
107 OPEN_DOMAIN_MULTI_SEED_ASSISTANT_CONFIRM_POSITIVE
099 BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE
```

Use the 102 repair checkpoint provenance recorded through 108/108A as read-only and record before/after hashes.

## Evaluation Contract

Run seeds:

```text
2044,2045,2046
```

Build fresh OOD rows with no exact or near-duplicate overlap against 108/108A rows.

Evaluate identical rows through separate paths:

```text
RAW_FREE_GENERATION
DECODER_REPAIRED_REFERENCE
INTEGRATED_DECODER_POLICY_GENERATION
```

Do not average or merge these metrics. Required reported metrics:

```text
raw_ood_stress_accuracy
decoder_reference_ood_stress_accuracy
integrated_ood_stress_accuracy
raw_vs_integrated_gap
integrated_vs_decoder_reference_gap
```

Every integrated row must include policy trace fields:

```text
context_carry_repair_used
instruction_boundary_repair_used
wrong_language_repair_used
prompt_format_repair_used
fallback_to_raw_used
decoder_reference_used
policy_trace_reason
```

If `decoder_reference_used` dominates nearly all rows, emit `DECODER_REFERENCE_DOMINATES_INTEGRATION`. This may still pass, but the report must not claim independent raw generation repair.

## Required Artifacts

Required artifacts:

```text
queue.json
progress.jsonl
integration_config.json
upstream_108a_manifest.json
upstream_108_manifest.json
checkpoint_integrity_manifest.json
fresh_integration_eval_dataset.jsonl
eval_row_hashes.json
raw_generation_results.jsonl
decoder_reference_results.jsonl
integrated_generation_results.jsonl
policy_trace_results.jsonl
family_metrics.json
seed_metrics.jsonl
multi_seed_aggregate.json
raw_vs_integrated_gap.json
integrated_vs_decoder_reference_gap.json
context_carry_repair_report.json
instruction_boundary_repair_report.json
language_repair_report.json
prompt_format_repair_report.json
retention_report.json
collapse_metrics.json
overclaim_metrics.json
human_readable_samples.jsonl
failure_case_samples.jsonl
decision.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be written from start and refreshed after upstream verification, checkpoint integrity, dataset build, every seed eval, aggregate analysis, decision writing, and final verdict.

## Positive Gate

Every seed must satisfy:

```text
integrated_ood_stress_accuracy >= 0.90
integrated_ood_stress_accuracy >= raw_ood_stress_accuracy + 0.25
integrated_vs_decoder_reference_gap <= 0.10
eval row hashes/counts match across all three paths
bounded_chat_slot_binding_accuracy >= 0.90
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_retention_accuracy >= 0.80
artifact_exfiltration_count = 0
all overclaim counts = 0
empty/static/repetition/copy collapse rejected
train_step_count = 0
optimizer_step_count = 0
checkpoint_hash_unchanged = true
bounded_release_artifact_unchanged = true
llm_judge_used = false
prediction_oracle_used = false
```

Positive verdict:

```text
DECODER_POLICY_INTEGRATION_POSITIVE
```

## Decision

Decision rules:

```text
retention fails -> 109R_RETENTION_REGRESSION_ANALYSIS
overclaim/exfiltration occurs -> 109C_BOUNDARY_OVERCLAIM_OR_EXFILTRATION_FAILURE_ANALYSIS
integrated improves but remains below gate -> 109B_DECODER_POLICY_INTEGRATION_FAILURE_ANALYSIS
integrated passes and decoder_reference_used dominates -> 110_INTEGRATED_PATH_PRODUCTIZATION_BOUNDARY_REVIEW
integrated passes with meaningful repair-stage traces -> 110_INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH
```

Positive 109 means a research integrated generation path closes most or all of the 108 raw OOD gap. It does not mean GPT-like readiness, open-domain assistant readiness, production chat, public API, service/runtime integration, deployment readiness, or safety alignment.
