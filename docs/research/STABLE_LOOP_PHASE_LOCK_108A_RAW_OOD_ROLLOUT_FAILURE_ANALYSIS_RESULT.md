# STABLE_LOOP_PHASE_LOCK_108A_RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_RESULT

## Status

Result document for `STABLE_LOOP_PHASE_LOCK_108A_RAW_OOD_ROLLOUT_FAILURE_ANALYSIS`.

The canonical machine-readable result is written by the runner under:

```text
target/pilot_wave/stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis/smoke/
```

108A is analysis-only. It does not train, repair, mutate checkpoints, change runtime/service/deploy code, or improve model capability.

It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

## Expected Positive Meaning

A positive 108A means:

```text
RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_POSITIVE
UPSTREAM_108_STRESS_MAP_VERIFIED
RAW_DECODER_GAP_CONFIRMED
RAW_FAILURES_ATTRIBUTED
PREFIX_SURVIVAL_ANALYZED
ROLLOUT_DRIFT_ANALYZED
STOP_CONDITION_ANALYZED
REPAIR_PLAN_WRITTEN
NO_TRAINING_PERFORMED
CHECKPOINT_UNCHANGED
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_CHAT_NOT_CLAIMED
```

It means the raw OOD failures from 108 were paired against decoder-passing outputs, attributed deterministically, and converted into a machine-readable 109 recommendation.

It does not mean the model improved.

## Required Result Fields

The `summary.json` metrics must record:

```text
raw_failure_count
raw_decoder_disagreement_count
decoder_success_on_raw_fail_rate
raw_rollout_drift_rate
unknown_raw_failure_rate
first_wrong_token_position_mean
first_wrong_token_position_median
gold_prefix_survival_rate_mean
gold_prefix_survival_rate_min
case_id_drift_rate
slot_drift_rate
distractor_leak_rate
stale_value_rate
hallucinated_fact_rate
over_refusal_rate
under_refusal_rate
prompt_injection_follow_rate
stop_condition_failure_rate
repetition_rate
utf8_valid_rate
checkpoint_hash_unchanged
bounded_release_artifact_unchanged
train_step_count
optimizer_step_count
recommended_next
primary_failure_mechanism
```

`recommended_repair_plan.json` must include:

```text
next
secondary_next_if_decoder_integration_fails
primary_failure_mechanism
secondary_failure_mechanisms
evidence_counts
evidence_rates
recommended_scope_for_109
```

## Validation

Expected validation commands:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis.py
python scripts/probes/run_stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis.py --out target/pilot_wave/stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis/smoke --upstream-108-root target/pilot_wave/stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch/smoke --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis_check.py
python scripts/probes/run_stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_107_open_domain_multi_seed_assistant_confirm_check.py --check-only
git diff --check
```

This result page intentionally stays static; the generated `summary.json`, `recommended_repair_plan.json`, and `report.md` are the authoritative run result.
