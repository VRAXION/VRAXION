# STABLE_LOOP_PHASE_LOCK_067_REAL_TEXT_ANCHORCELL_TRAINING_POC Contract

Status: contract for the 067 bounded real-text + AnchorCell training PoC.

067 asks whether a small FineWeb-Edu carrier slice plus synthetic AnchorCell
trace/final-answer examples and counterfactual traps can produce learned,
input-conditioned behavior that beats frequency/static/copy baselines. It is a
training sanity gate, not a product release or production training milestone.

no production training
no full-corpus training
no GA
no public beta
no hosted SaaS
no clinical use
no high-stakes education use
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior

## Required Commands

```powershell
cargo check -p instnct-core --example phase_lane_real_text_anchorcell_training_poc
cargo run -p instnct-core --example phase_lane_real_text_anchorcell_training_poc -- --out target/pilot_wave/stable_loop_phase_lock_067_real_text_anchorcell_training_poc/smoke --fineweb-root "S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B" --mode smoke --seed 2026 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_067_real_text_anchorcell_training_poc_check.py
python scripts/probes/run_stable_loop_phase_lock_067_real_text_anchorcell_training_poc_check.py --check-only
cargo test -p instnct-core sdk_candidate
python scripts/probes/run_stable_loop_phase_lock_066_core_ga_private_readiness_check.py --check-only
git diff --check
```

## Input Boundary

Default smoke source:

```text
S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt
```

The runner must use `fineweb_edu_30m.txt`. If it is missing, emit
`FINEWEB_SMOKE_SOURCE_MISSING` and stop. It must not silently switch to parquet
shards.

The FineWeb root is read-only input. The runner must record source hash, size,
and modified timestamp before and after the run. Any change emits
`FINEWEB_INPUT_MUTATION_DETECTED`.

Smoke mode caps FineWeb bytes to 10-50 MiB. Any attempt to run all parquet
shards or full corpus scale emits `FULL_CORPUS_TRAINING_ATTEMPTED`.

## Required Artifacts

```text
queue.json
progress.jsonl
dataset_manifest.json
fineweb_file_manifest.json
fineweb_sample_offsets.jsonl
train_examples_sample.jsonl
heldout_examples_sample.jsonl
ood_examples_sample.jsonl
anchorcell_examples_sample.jsonl
baseline_metrics.json
training_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
reload_eval_report.json
rollback_report.json
resume_report.json
inference_samples.jsonl
collapse_metrics.json
baseline_knockout_report.json
per_family_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be written from the
start and refreshed at heartbeat cadence. The default heartbeat is 20 seconds.

## Hard Guardrails

Learned arms must show actual parameter updates:

```text
train_step_count > 0
updated_parameter_count > 0
checkpoint_after_hash != checkpoint_before_hash
```

If this fails, emit `NO_ACTUAL_TRAINING_UPDATE_DETECTED`.

Prediction must not use a parser/oracle/label generator. Dataset generation and
audit baselines may use labels, but model prediction must record:

```text
prediction_oracle_used = false
```

If violated, emit `ORACLE_SHORTCUT_DETECTED`.

Split leakage audit must record:

```text
train_eval_exact_input_overlap_count
train_eval_exact_label_overlap_count
train_ood_exact_input_overlap_count
heldout_ood_template_overlap_count
key_value_pair_overlap_count
```

Any exact input overlap into heldout or OOD emits
`TRAIN_EVAL_LEAKAGE_DETECTED`.

All baselines must evaluate on the same heldout/OOD rows as learned arms.
The baseline report must record:

```text
baseline_eval_mismatch = false
```

If not, emit `BASELINE_EVAL_MISMATCH`.

Positive cannot be aggregate-only. Per-family metrics are required for:

```text
FINEWEB_RAW_CONTINUATION
ANCHORCELL_TRACE_BINDING
ANCHORCELL_FINAL_ANSWER_ONLY
COUNTERFACTUAL_KEY_VALUE_PAIRS
CONTEXT_CARRY_QUERY_ANSWER
NON_ROUTE_TEXT_CONTROL
```

If the family-min gate fails, emit `FAMILY_MIN_GATE_FAILS`.

Collapse metrics must be global and per-family:

```text
top_output_rate
space_output_rate
empty_output_rate
unique_output_count
output_entropy
repetition_rate
copy_last_token_rate
```

If collapse or copy shortcut is detected, emit
`STATIC_OUTPUT_COLLAPSE_DETECTED` or `COPY_SHORTCUT_DETECTED`.

Checkpoint pipeline strictness requires:

```text
checkpoint_save_load_pass = true
eval_after_reload_matches_before = true
rollback_success = true
resume_from_checkpoint_pass = true
resumed_checkpoint_hash_changed = true
```

Failure emits `CHECKPOINT_RELOAD_FAILS`, `ROLLBACK_REHEARSAL_FAILS`, or
`RESUME_FROM_CHECKPOINT_FAILS`.

## Positive Gate

The positive gate is:

```text
heldout_exact_accuracy >= 0.85
ood_exact_accuracy >= 0.75
context_carry_accuracy >= 0.85
paired_counterfactual_accuracy >= 0.90
family_min_accuracy >= 0.70
delta_vs_unigram > 0.10
delta_vs_bigram > 0.05
delta_vs_trigram > 0.03
top_output_rate <= 0.35
space_output_rate <= 0.02
empty_output_rate <= 0.02
collapse_detected = false
checkpoint_save_load_pass = true
rollback_success = true
resume_from_checkpoint_pass = true
eval_after_reload_matches_before = true
```

If any hard gate fails, the runner must not emit
`REAL_TEXT_ANCHORCELL_TRAINING_POC_POSITIVE`. Honest failure is acceptable.

## Verdicts

Positive verdicts include:

```text
REAL_TEXT_ANCHORCELL_TRAINING_POC_POSITIVE
FINEWEB_INPUT_IMMUTABILITY_PASSES
FINEWEB_CARRIER_TRAINING_WORKS
ANCHORCELL_TRACE_SUPERVISION_WORKS
MIXED_DATASET_BEATS_BASELINES
FREQUENCY_BASELINE_REJECTED
BIGRAM_TRIGRAM_BASELINE_REJECTED
STATIC_OUTPUT_COLLAPSE_REJECTED
COPY_SHORTCUT_REJECTED
TRAIN_EVAL_LEAKAGE_REJECTED
ORACLE_SHORTCUT_REJECTED
PER_FAMILY_GATES_PASS
CHECKPOINT_PIPELINE_STRICT_PASS
PRODUCTION_TRAINING_NOT_CLAIMED
```

Failure verdicts include:

```text
REAL_TEXT_ANCHORCELL_TRAINING_POC_FAILS
FINEWEB_SMOKE_SOURCE_MISSING
FINEWEB_INPUT_MUTATION_DETECTED
FULL_CORPUS_TRAINING_ATTEMPTED
NO_ACTUAL_TRAINING_UPDATE_DETECTED
ORACLE_SHORTCUT_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
FAMILY_MIN_GATE_FAILS
STATIC_OUTPUT_COLLAPSE_DETECTED
COPY_SHORTCUT_DETECTED
CHECKPOINT_RELOAD_FAILS
ROLLBACK_REHEARSAL_FAILS
RESUME_FROM_CHECKPOINT_FAILS
```
