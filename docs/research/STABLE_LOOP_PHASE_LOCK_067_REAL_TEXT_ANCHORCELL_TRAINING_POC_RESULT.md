# STABLE_LOOP_PHASE_LOCK_067_REAL_TEXT_ANCHORCELL_TRAINING_POC Result

Status: implementation result for the 067 bounded real-text + AnchorCell
training PoC.

067 adds a research runner and static checker for the first mixed FineWeb-Edu
carrier plus AnchorCell/counterfactual training sanity gate. The checker
validates committed files only. The runner writes generated artifacts only
under `target/pilot_wave/stable_loop_phase_lock_067_real_text_anchorcell_training_poc/`.

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

## Implementation Summary

The runner:

```text
uses fineweb_edu_30m.txt as explicit smoke source
does not fall back to full parquet shards
hashes FineWeb input before and after run
records source size and modified timestamp
keeps FineWeb root read-only
uses deterministic seed 2026
records dataset offsets and split counts
trains learned arms with real parameter updates
records checkpoint_before_hash and checkpoint_after_hash
requires checkpoint_after_hash != checkpoint_before_hash for learned arms
records prediction_oracle_used = false
checks split leakage
evaluates baselines on the same heldout/OOD rows
records baseline_eval_mismatch = false
records global and per-family collapse metrics
checks checkpoint reload, rollback, and resume
```

The smoke source is:

```text
S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt
```

Missing source fails with `FINEWEB_SMOKE_SOURCE_MISSING`. Input mutation fails
with `FINEWEB_INPUT_MUTATION_DETECTED`. Full corpus attempts fail with
`FULL_CORPUS_TRAINING_ATTEMPTED`.

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

## Artifact Contract

The runner writes:

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

The runner refreshes `progress.jsonl`, `summary.json`, and `report.md` during
execution and at completion.

## Guardrail Outcomes

The result path is designed to emit these positive verdicts when the hard gates
pass:

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

It emits failure verdicts instead when appropriate:

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

## Boundary

067 does not prove a full English model, and it makes no language grounding or
open-ended reasoning claim. It also makes no clinical readiness claim,
no high-stakes education readiness claim, no hosted SaaS readiness claim, and
no production training readiness claim. It is only a bounded smoke gate for mixed
real-text carrier training, AnchorCell trace supervision, baseline knockout,
anti-collapse metrics, and checkpoint pipeline correctness.
