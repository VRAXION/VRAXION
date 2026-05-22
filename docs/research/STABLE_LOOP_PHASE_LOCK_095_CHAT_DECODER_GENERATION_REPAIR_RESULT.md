# STABLE_LOOP_PHASE_LOCK_095_CHAT_DECODER_GENERATION_REPAIR Result

## Status

095 implements the target-only decoder generation repair PoC selected by 094B.

Expected smoke root:

```text
target/pilot_wave/stable_loop_phase_lock_095_chat_decoder_generation_repair/smoke
```

This is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not deployment, not public release, and not safety alignment.

## Expected Artifacts

```text
queue.json
progress.jsonl
repair_config.json
upstream_094b_manifest.json
checkpoint_integrity_manifest.json
eval_row_manifest.json
decoder_policy_manifest.json
repaired_generation_results.jsonl
baseline_vs_repaired_report.json
family_metrics.json
stop_condition_report.json
human_readable_samples.jsonl
failure_case_samples.jsonl
summary.json
report.md
```

## Positive Verdicts

```text
CHAT_DECODER_GENERATION_REPAIR_POSITIVE
UPSTREAM_094B_GAP_ANALYSIS_VERIFIED
TARGET_ONLY_DECODER_REPAIR_WRITTEN
GENERATION_ACCURACY_REPAIRED
STOP_CONDITION_REPAIRED
FINITE_LABEL_OUTPUT_REPAIRED
CHECKPOINTS_UNCHANGED
NO_TRAINING_PERFORMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

## Failure Verdicts

```text
CHAT_DECODER_GENERATION_REPAIR_FAILS
UPSTREAM_094B_ARTIFACT_MISSING
UPSTREAM_094B_NOT_POSITIVE
DECODER_REPAIR_INSUFFICIENT
DECODER_REPAIR_FAMILY_REGRESSION
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_RELEASE_CLAIM_DETECTED
```

## Validation

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_095_chat_decoder_generation_repair.py
python scripts/probes/run_stable_loop_phase_lock_095_chat_decoder_generation_repair.py --out target/pilot_wave/stable_loop_phase_lock_095_chat_decoder_generation_repair/smoke --upstream-094b-root target/pilot_wave/stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis/smoke --seed 2028 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_095_chat_decoder_generation_repair_check.py
python scripts/probes/run_stable_loop_phase_lock_095_chat_decoder_generation_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis_check.py --check-only
git diff --check
```

## Next

If 095 passes, proceed to `096_FRESH_CHAT_GENERATION_EVAL`.
