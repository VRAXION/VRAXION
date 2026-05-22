# STABLE_LOOP_PHASE_LOCK_109_DECODER_POLICY_INTEGRATION_RESULT

## Status

Result document for `STABLE_LOOP_PHASE_LOCK_109_DECODER_POLICY_INTEGRATION`.

The canonical machine-readable result is generated under:

```text
target/pilot_wave/stable_loop_phase_lock_109_decoder_policy_integration/smoke/
```

109 is research-harness integration only. It does not train, mutate checkpoints, change service/runtime/deploy code, or create production integration.

It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

## Expected Positive Meaning

A positive 109 means:

```text
DECODER_POLICY_INTEGRATION_POSITIVE
UPSTREAM_108A_RAW_OOD_ANALYSIS_VERIFIED
INTEGRATED_DECODER_POLICY_GENERATION_EVALUATED
RAW_DECODER_INTEGRATED_PATHS_REPORTED_SEPARATELY
POLICY_TRACE_WRITTEN
RAW_OOD_GAP_CLOSED
RETENTION_PASSES
COLLAPSE_REJECTED
OVERCLAIM_REJECTED
CHECKPOINT_UNCHANGED
NO_TRAINING_PERFORMED
NO_RUNTIME_SURFACE_MUTATION
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_CHAT_NOT_CLAIMED
```

It means the integrated research generation path closed the raw OOD stress gap on the 109 eval batch while retaining hard integrity, retention, and boundary gates.

It does not mean the model was improved or that the service runtime was changed.

## Required Result Fields

`summary.json` and `multi_seed_aggregate.json` must record:

```text
raw_ood_stress_accuracy
decoder_reference_ood_stress_accuracy
integrated_ood_stress_accuracy
raw_vs_integrated_gap
integrated_vs_decoder_reference_gap
decoder_reference_used_rate
repair_stage_trace_rate
decoder_reference_dominates_integration
bounded_chat_slot_binding_accuracy
finite_label_anchorroute_retention_accuracy
unsupported_refusal_retention_accuracy
checkpoint_hash_unchanged
bounded_release_artifact_unchanged
train_step_count
optimizer_step_count
next
```

`policy_trace_results.jsonl` must show which integrated policy stages fired for every row.

## Validation

Expected validation commands:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_109_decoder_policy_integration.py
python scripts/probes/run_stable_loop_phase_lock_109_decoder_policy_integration.py --out target/pilot_wave/stable_loop_phase_lock_109_decoder_policy_integration/smoke --upstream-108a-root target/pilot_wave/stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis/smoke --upstream-108-root target/pilot_wave/stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch/smoke --upstream-107-root target/pilot_wave/stable_loop_phase_lock_107_open_domain_multi_seed_assistant_confirm/smoke --upstream-099-root target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke --seeds 2044,2045,2046 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_109_decoder_policy_integration_check.py
python scripts/probes/run_stable_loop_phase_lock_109_decoder_policy_integration_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch_check.py --check-only
git diff --check
```

The generated artifacts are authoritative; this static result document records the contract and expected interpretation only.
