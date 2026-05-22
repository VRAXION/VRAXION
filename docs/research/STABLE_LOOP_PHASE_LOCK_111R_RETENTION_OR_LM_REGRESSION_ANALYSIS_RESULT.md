# STABLE_LOOP_PHASE_LOCK_111R_RETENTION_OR_LM_REGRESSION_ANALYSIS Result

111R is expected to write its canonical result under:

```text
target/pilot_wave/stable_loop_phase_lock_111r_retention_or_lm_regression_analysis/smoke/
```

The expected positive result is:

```text
RETENTION_OR_LM_REGRESSION_ANALYSIS_POSITIVE
```

Positive means the failed 111 standard run was analyzed and the root cause was classified. It does not mean the 111 target checkpoint improved, and it does not make readiness or production claims.

## Expected Current Finding

Based on the failed 111 artifacts, the likely classification is:

```text
primary_root_cause = MIXED_CAUSE
recommended_next = 111X_COMBINED_RAW_DISTILLATION_REDESIGN
```

Expected contributing causes:

- `EVAL_PATH_MISMATCH`
- `NAMESPACE_MEMORIZATION`
- `TEACHER_FORCING_ROLLOUT_GAP`
- `RETENTION_MIX_UNDERPOWERED`
- `TARGET_CHECKPOINT_COLLAPSE`
- `DATA_BALANCE_FAILURE`

## Validation

Run:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_111r_retention_or_lm_regression_analysis.py
python scripts/probes/run_stable_loop_phase_lock_111r_retention_or_lm_regression_analysis.py --out target/pilot_wave/stable_loop_phase_lock_111r_retention_or_lm_regression_analysis/smoke --upstream-111-root target/pilot_wave/stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair/smoke --upstream-110-root target/pilot_wave/stable_loop_phase_lock_110_integrated_decoder_policy_ood_confirm_batch/smoke --upstream-109-root target/pilot_wave/stable_loop_phase_lock_109_decoder_policy_integration/smoke --upstream-108a-root target/pilot_wave/stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis/smoke --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_111r_retention_or_lm_regression_analysis_check.py
python scripts/probes/run_stable_loop_phase_lock_111r_retention_or_lm_regression_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_110_integrated_decoder_policy_ood_confirm_batch_check.py --check-only
git diff --check
```

Do not require the positive 111 checker for 111R because 111R intentionally consumes a failed 111 root.
