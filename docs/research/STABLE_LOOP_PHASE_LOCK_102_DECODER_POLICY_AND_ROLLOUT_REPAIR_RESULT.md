# STABLE_LOOP_PHASE_LOCK_102_DECODER_POLICY_AND_ROLLOUT_REPAIR Result

## Status

Pending local smoke execution.

102 is a target-only research repair. It trains only a new 102 checkpoint copy and must keep 100, 101, the 099 bounded release stack, packaged winner artifacts, runtime/service/deploy code, SDK/public exports, product/release docs, and root `LICENSE` unchanged.

This is raw generation repair only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not public release, and not safety alignment.

## Expected Positive Meaning

If positive, 102 means:

```text
DECODER_POLICY_AND_ROLLOUT_REPAIR_POSITIVE
```

The expected improvement is:

```text
raw generation improves over 101
case ID drift is reduced by at least 50 percent relative
slot drift remains low
decoder-assisted reference stays >= 0.90
bounded/AnchorRoute retention survives
collapse is rejected
source 100 checkpoint remains unchanged
bounded release artifacts remain unchanged
```

## Required Evidence

The smoke root must include:

```text
queue.json
progress.jsonl
repair_config.json
upstream_101_manifest.json
source_checkpoint_manifest.json
repair_dataset_manifest.json
repair_train_examples_sample.jsonl
repair_eval_examples_sample.jsonl
training_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
pre_repair_eval_metrics.json
post_repair_eval_metrics.json
arm_comparison.json
control_delta_report.json
raw_rollout_drift_report.json
case_id_anchor_report.json
slot_pinning_report.json
language_guard_report.json
lm_retention_report.json
retention_report.json
collapse_metrics.json
generation_samples.jsonl
human_readable_samples.jsonl
failure_case_samples.jsonl
summary.json
report.md
```

## Validation

Run:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_102_decoder_policy_and_rollout_repair.py
python scripts/probes/run_stable_loop_phase_lock_102_decoder_policy_and_rollout_repair.py --out target/pilot_wave/stable_loop_phase_lock_102_decoder_policy_and_rollout_repair/smoke --upstream-101-root target/pilot_wave/stable_loop_phase_lock_101_fresh_assistant_eval_and_raw_decoder_frontier_map/smoke --upstream-100-root target/pilot_wave/stable_loop_phase_lock_100_open_vocab_assistant_capability_scale/smoke --seed 2026 --repair-examples 24000 --lm-replay-tokens 200000 --seq-len 128 --batch-size 32 --steps 2000 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_102_decoder_policy_and_rollout_repair_check.py
python scripts/probes/run_stable_loop_phase_lock_102_decoder_policy_and_rollout_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_101_fresh_assistant_eval_and_raw_decoder_frontier_map_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_100_open_vocab_assistant_capability_scale_check.py --check-only
git diff --check
```
