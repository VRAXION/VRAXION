# STABLE_LOOP_PHASE_LOCK_068_REAL_TEXT_ANCHORCELL_CONFIRM_SCALE Result

Status: implementation result for the 068 real-text AnchorCell confirm scale
gate.

068 adds a static orchestrator/checker package for confirming the 067
real-text AnchorCell training PoC with larger but still controlled scale. It
does not run full 10B training, does not claim production training, and does
not claim a full English model.

The aggregate decision focuses on the child `MIXED_WITH_ROUTE_GRAMMAR_ON` arm.

no full 10B training
no production training
no full English model
no language grounding
no GA
no public beta
no hosted SaaS
no clinical use
no high-stakes education use
no full VRAXION
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior

## Implementation Summary

The 067 runner now supports `--mode confirm` with `--fineweb-source` and
`--anchorcell-examples`. Smoke mode remains bounded to `fineweb_edu_30m.txt`
and the original 10-50 MiB smoke cap.

The 068 orchestrator:

```text
creates or reuses a FineWeb confirm snapshot under target/
records FineWeb source hash, size, and modified timestamp before and after
records confirm snapshot hash before and after child runs
records the extraction command and extraction timestamp
runs child 067 confirm jobs for seeds 2026,2027,2028
records each child command exactly
rejects stale child artifacts
rechecks child 067 hard gates independently
requires all three seeds to pass
aggregates baseline knockout metrics per seed and cross seed
writes failure_case_samples.jsonl even when empty
records budget facts without performance claims
```

The 068 static checker validates committed files only. It does not run the
confirm jobs and does not validate generated `target/` artifacts.

## Observed Confirm Run

The completed confirm run used a 268435456-byte target-local FineWeb snapshot
derived from one recorded FineWeb source file and then rechecked the source and
snapshot hashes after all child runs. The configured confirm example budget was
`anchorcell_examples = 100000`; the generated artifacts record the actual
synthetic/control row count as `synthetic_examples_used = 21000`.

The three fresh child seeds all passed:

```text
seed 2026: heldout = 1.0, ood = 0.9999333333333333, family_min = 0.9988888888888889
seed 2027: heldout = 1.0, ood = 1.0, family_min = 1.0
seed 2028: heldout = 1.0, ood = 1.0, family_min = 1.0
```

Cross-seed baseline knockout remained positive:

```text
min_delta_vs_unigram = 0.30079999999999996
min_delta_vs_bigram = 0.3022666666666667
min_delta_vs_trigram = 0.3025
```

The confirm run recorded no production performance claim, no training
throughput claim, and no full-corpus claim.

## Required Commands

```powershell
cargo check -p instnct-core --example phase_lane_real_text_anchorcell_training_poc
python -m py_compile scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py
python scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale.py --out target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm --fineweb-root "S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B" --fineweb-bytes 268435456 --seeds 2026,2027,2028 --anchorcell-examples 100000 --heartbeat-sec 20
python scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_067_real_text_anchorcell_training_poc_check.py --check-only
cargo test -p instnct-core sdk_candidate
git diff --check
```

## Guardrail Outcomes

Fresh child-run validation is explicit:

```text
child_run_started = true
child_run_completed = true
child_exit_code = 0
child_summary_newer_than_068_start = true
child_report_newer_than_068_start = true
child_command recorded exactly
```

Confirm snapshot immutability is explicit:

```text
confirm snapshot hash before child runs
confirm snapshot hash after child runs
CONFIRM_SNAPSHOT_MUTATION_DETECTED on mismatch
```

FineWeb root immutability is explicit:

```text
source file hashes before/after
source file sizes before/after
source file modified timestamps before/after
FINEWEB_INPUT_MUTATION_DETECTED on mismatch
```

Scale limits are explicit:

```text
fineweb_bytes <= 1 GiB
anchorcell_examples <= 250000
no full parquet sweep
no all-shard training
no silent fallback to full corpus
```

## 067 Gate Recheck

For every seed, the orchestrator rechecks:

```text
REAL_TEXT_ANCHORCELL_TRAINING_POC_POSITIVE
prediction_oracle_used = false
baseline_eval_mismatch = false
train_eval_exact_input_overlap_count = 0
train_ood_exact_input_overlap_count = 0
checkpoint_save_load_pass = true
rollback_success = true
resume_from_checkpoint_pass = true
collapse_detected = false
```

## Artifact Contract

The orchestrator writes:

```text
queue.json
progress.jsonl
confirm_config.json
fineweb_confirm_source_manifest.json
fineweb_extraction_manifest.json
seed_metrics.jsonl
aggregate_metrics.json
multi_seed_stability.json
training_curve_report.json
baseline_knockout_aggregate.json
failure_case_samples.jsonl
checkpoint_pipeline_report.json
summary.json
report.md
```

Generated artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/
```

They are not committed.

## Verdicts

Positive verdicts include:

```text
REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_POSITIVE
FRESH_CHILD_RUNS_CONFIRMED
CONFIRM_SNAPSHOT_IMMUTABILITY_PASSES
CHILD_067_GATES_RECHECKED
MULTI_SEED_MIN_GATE_PASSES
CONFIRM_SCALE_LIMIT_ENFORCED
FAILURE_CASE_REPORT_WRITTEN
BASELINE_KNOCKOUT_STABLE
CHECKPOINT_PIPELINE_MULTI_SEED_PASS
PRODUCTION_TRAINING_NOT_CLAIMED
```

Failure verdicts include:

```text
REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_FAILS
STALE_CHILD_ARTIFACT_USED
CONFIRM_SNAPSHOT_MUTATION_DETECTED
FINEWEB_INPUT_MUTATION_DETECTED
FULL_CORPUS_TRAINING_ATTEMPTED
CONFIRM_SCALE_LIMIT_EXCEEDED
MULTI_SEED_INSTABILITY_DETECTED
CHILD_GATE_RECHECK_FAILS
BASELINE_KNOCKOUT_REGRESSION
FAILURE_CASE_REPORT_MISSING
PERFORMANCE_CLAIM_OVERREACH
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Boundary

068 is a confirm-scale research gate only. It does not produce a production
training artifact, does not create a release, does not authorize hosted SaaS,
does not authorize public beta, and does not make a language grounding claim.
The result is useful only as a controlled larger-scale check of the 067
real-text AnchorCell training PoC.
