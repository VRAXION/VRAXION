# STABLE_LOOP_PHASE_LOCK_068_REAL_TEXT_ANCHORCELL_CONFIRM_SCALE Contract

Status: contract for the 068 real-text AnchorCell confirm scale gate.

068 confirms the 067 real-text AnchorCell training PoC at controlled larger
scale. It runs three fresh child 067 confirm jobs over a target-local FineWeb
confirm snapshot and verifies that the 067 positive smoke result is not a
single-seed or stale-artifact outcome.

The key arm remains `MIXED_WITH_ROUTE_GRAMMAR_ON`.

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

## Scope

068 extends only the 067 real-text AnchorCell training PoC runner with
`--mode confirm`, `--fineweb-source`, and `--anchorcell-examples`.
The 067 smoke behavior remains unchanged.

The orchestrator is:

```text
scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale.py
```

The static checker is:

```text
scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py
```

068 does not modify the service/API alpha, deployment harness, public crate
exports, release docs, or root `LICENSE`.

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

## Confirm Snapshot

The FineWeb root remains read-only:

```text
S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B
```

068 creates or reuses a FineWeb confirm snapshot under:

```text
target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/confirm_snapshot/
```

The snapshot is a bounded FineWeb confirm snapshot, not a full corpus extract.
The orchestrator records:

```text
source file hashes
source file sizes
source file modified timestamps
snapshot hash before child runs
snapshot hash after child runs
snapshot byte count
extraction command
extraction timestamp
```

If the snapshot changes during the run, emit
`CONFIRM_SNAPSHOT_MUTATION_DETECTED`. If any FineWeb source file actually read
changes during the run, emit `FINEWEB_INPUT_MUTATION_DETECTED`.

## Confirm Limits

The limits are locked:

```text
fineweb_bytes <= 1 GiB
default fineweb_bytes = 268435456
anchorcell_examples <= 250000
default anchorcell_examples = 100000
seeds = 2026,2027,2028
```

In confirm mode, `anchorcell_examples` is the configured confirm example budget.
The child runner records the actual FineWeb, AnchorCell, counterfactual,
context-carry, and control row counts in `dataset_manifest.json`; the 068
orchestrator records the actual synthetic/control row count in the aggregate
budget facts.

The runner and orchestrator must reject full parquet sweep, all-shard training,
silent fallback to full corpus, and missing `--mode confirm`. Failure emits
`FULL_CORPUS_TRAINING_ATTEMPTED` or `CONFIRM_SCALE_LIMIT_EXCEEDED`.

## Fresh Child Runs

For each seed, the orchestrator records:

```text
child_run_started = true
child_run_completed = true
child_exit_code = 0
child_summary_newer_than_068_start = true
child_report_newer_than_068_start = true
child_command recorded exactly
```

If stale child artifacts are used, emit `STALE_CHILD_ARTIFACT_USED`.

## Child 067 Gate Recheck

For every seed, 068 independently parses child artifacts and rechecks:

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

Failure emits `CHILD_GATE_RECHECK_FAILS`.

## Multi-Seed Gate

Positive verdict requires all three seeds to pass. Mean-only pass, best-seed
pass, and two-of-three pass are not allowed. Failure emits
`MULTI_SEED_INSTABILITY_DETECTED`.

## Baseline And Collapse Aggregation

The aggregate report must include:

```text
per_seed_delta_vs_unigram
per_seed_delta_vs_bigram
per_seed_delta_vs_trigram
min_delta_vs_unigram
min_delta_vs_bigram
min_delta_vs_trigram
stddev_delta_vs_unigram
stddev_delta_vs_bigram
stddev_delta_vs_trigram
heldout_exact_accuracy
ood_exact_accuracy
context_carry_accuracy
paired_counterfactual_accuracy
family_min_accuracy
collapse metrics
```

If baseline knockout regresses, emit `BASELINE_KNOCKOUT_REGRESSION`.

## Failure Cases And Budget Honesty

The orchestrator always writes `failure_case_samples.jsonl`, even when empty.
If any family accuracy is below 1.0, rows should include:

```text
input
expected
predicted
seed
task_family
arm
```

If the failure case report is missing, emit `FAILURE_CASE_REPORT_MISSING`.

Budget facts must include elapsed seconds per seed, total elapsed seconds,
peak memory if available, disk usage under target outdir, FineWeb bytes used,
and synthetic examples used.

The report must state:

```text
no production performance claim
no training throughput claim
no full-corpus claim
```

If it overclaims performance, emit `PERFORMANCE_CLAIM_OVERREACH`.

## Required Artifacts

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
