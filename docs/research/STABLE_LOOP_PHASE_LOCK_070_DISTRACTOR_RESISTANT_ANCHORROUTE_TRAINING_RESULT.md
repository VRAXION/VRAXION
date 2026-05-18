# STABLE_LOOP_PHASE_LOCK_070_DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING Result

Status: implementation result document for targeted hard-distractor
AnchorRoute repair training.

070 adds a bounded finite-label repair runner and static checker. It compares a
fresh targeted curriculum with a warm-start fine-tune from the 068
`MIXED_WITH_ROUTE_GRAMMAR_ON` checkpoint. It is designed to repair the 069 weak
families while preserving the previously strong finite-label families.

This milestone is finite-label repair training only.

no open-ended assistant
no free-form generation
no perplexity
no full English LM
no language grounding
no production training
no production-scale training
no GA
no public beta
no hosted SaaS
no clinical use
no high-stakes education use
no service API change
no deployment harness change
no public crate export change
no root LICENSE change

## Implementation Summary

Runner:

```text
instnct-core/examples/phase_lane_distractor_resistant_anchorroute_training.rs
```

Static checker:

```text
scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py
```

The runner loads the 068 checkpoint read-only, records upstream hash before and
after the run, loads the 069 benchmark artifacts as reference, builds targeted
hard-distractor training rows, compares fresh and fine-tune arms on identical
evaluation rows, and writes checkpoint, retention, arm comparison, human sample,
failure sample, collapse, and summary artifacts.

It records:

```text
warm_start_hash
initialized_hash
final_hash
prediction_oracle_used = false
overlap_with_069_eval_count
best_arm
fresh_pass
finetune_pass
fresh_vs_finetune_delta
recommended_next_strategy
```

## Required Commands

```powershell
cargo check -p instnct-core --example phase_lane_distractor_resistant_anchorroute_training
cargo run -p instnct-core --example phase_lane_distractor_resistant_anchorroute_training -- --out target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --upstream-checkpoint target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/seed_2028/checkpoints/MIXED_WITH_ROUTE_GRAMMAR_ON/model_checkpoint.json --upstream-summary target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/summary.json --benchmark-069-root target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke --targeted-examples 120000 --seed 2026 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py
python scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py --check-only
cargo test -p instnct-core sdk_candidate
git diff --check
```

## Required Artifacts

The runner writes generated artifacts only under:

```text
target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/
```

Required artifact names:

```text
queue.json
progress.jsonl
training_config.json
upstream_068_manifest.json
baseline_069_reference.json
targeted_dataset_manifest.json
train_examples_sample.jsonl
heldout_examples_sample.jsonl
ood_examples_sample.jsonl
training_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
post_training_capability_metrics.json
per_family_metrics.json
retention_metrics.json
regression_report.json
baseline_knockout_report.json
arm_comparison.json
human_readable_samples.jsonl
failure_case_samples.jsonl
collapse_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from the start and
then refreshed after major phases, so the run is not a black box.

## Arms And Families

Required arms:

```text
NO_TRAIN_069_BASELINE
FRESH_TARGETED_MIX_TRAINING
FINETUNE_068_TARGETED_REPAIR
FRESH_SHUFFLED_LABEL_CONTROL
FINETUNE_SHUFFLED_LABEL_CONTROL
NO_ROUTE_FEATURE_CONTROL
CHECKPOINT_RELOAD_EVAL
ROLLBACK_REHEARSAL
RESUME_FROM_CHECKPOINT
```

Required training families:

```text
HARD_DISTRACTOR_ANCHOR_BINDING
NEAR_MISS_ANCHOR_SELECTION
SAME_KEY_DIFFERENT_CONTEXT
LONG_CONTEXT_NEEDLE_RETRIEVAL
IRRELEVANT_POCKET_SUPPRESSION
NEGATIVE_ROUTE_REJECTION
ANSWER_ONLY_HARD_BINDING
TRACE_MIXED_HARD_BINDING
RETENTION_FINEWEB_CONTINUATION
RETENTION_NON_ROUTE_CONTROL
```

## Boundaries Preserved

The runner does not modify:

```text
tools/instnct_service_alpha/
tools/instnct_deploy/
docs/releases/
root LICENSE
public crate exports
```

Generated 070 checkpoints are under `target/` only and are not committed.

## Verdict Semantics

Positive result requires both learned arms to train, exact 069 benchmark
training overlap to remain zero, the upstream 068 checkpoint to remain
unchanged, the retention gate to pass, the arm comparison to be written, the
best learned arm to pass checkpoint reload/rollback/resume, and the capability
thresholds to pass without collapse.

Failure is acceptable and must be preserved with failure verdicts. If any hard
gate fails, the runner does not emit
`DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_POSITIVE`.

## Observed Smoke Result

The required 120000-example smoke run completed with:

```text
DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_POSITIVE
FRESH_TARGETED_MIX_TRAINING_COMPLETED
FINETUNE_068_TARGETED_REPAIR_COMPLETED
DISTRACTOR_RESISTANCE_IMPROVED
CONTEXT_ENTITY_EXTRACTION_IMPROVED
COUNTERFACTUAL_BINDING_IMPROVED
LONG_CONTEXT_NEEDLE_IMPROVED
RETENTION_GATE_PASSES
RETENTION_REGRESSION_REJECTED
ARM_COMPARISON_WRITTEN
BEST_ARM_SELECTED
TRAIN_BENCHMARK_LEAKAGE_REJECTED
UPSTREAM_068_CHECKPOINT_UNCHANGED
BASELINE_KNOCKOUT_STABLE
CHECKPOINT_PIPELINE_STRICT_PASS
ORACLE_SHORTCUT_REJECTED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Best arm:

```text
FINETUNE_068_TARGETED_REPAIR
```

Observed best-arm metrics:

```text
supported_accuracy = 0.961139896373057
family_min_accuracy = 0.8333333333333334
context_entity_extraction_accuracy = 0.9791666666666666
counterfactual_binding_accuracy = 0.9583333333333334
distractor_resistance_accuracy = 0.9375
long_context_needle_accuracy = 0.9791666666666666
instruction_following_closed_accuracy = 1.0
multi_hop_key_value_accuracy = 0.8333333333333334
symbolic_rule_closed_choice_accuracy = 1.0
non_route_text_control_accuracy = 1.0
```

Arm comparison:

```text
fresh_pass = true
finetune_pass = true
fresh_vs_finetune_delta = 0.0625
recommended_next_strategy = continue targeted repair from 068-style checkpoint with retention checks
```

Integrity observations:

```text
overlap_with_069_eval_count = 0
upstream 068 checkpoint unchanged = true
checkpoint reload / rollback / resume = true
prediction_oracle_used = false
```

Expected positive verdicts include:

```text
DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_POSITIVE
FRESH_TARGETED_MIX_TRAINING_COMPLETED
FINETUNE_068_TARGETED_REPAIR_COMPLETED
DISTRACTOR_RESISTANCE_IMPROVED
CONTEXT_ENTITY_EXTRACTION_IMPROVED
COUNTERFACTUAL_BINDING_IMPROVED
LONG_CONTEXT_NEEDLE_IMPROVED
RETENTION_GATE_PASSES
RETENTION_REGRESSION_REJECTED
ARM_COMPARISON_WRITTEN
BEST_ARM_SELECTED
TRAIN_BENCHMARK_LEAKAGE_REJECTED
UPSTREAM_068_CHECKPOINT_UNCHANGED
BASELINE_KNOCKOUT_STABLE
CHECKPOINT_PIPELINE_STRICT_PASS
ORACLE_SHORTCUT_REJECTED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Expected failure verdicts include:

```text
DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_FAILS
UPSTREAM_068_ARTIFACT_MISSING
FULL_CORPUS_TRAINING_ATTEMPTED
TARGETED_SCALE_LIMIT_EXCEEDED
CHECKPOINT_MUTATION_DETECTED
NO_ACTUAL_TRAINING_UPDATE_DETECTED
ORACLE_SHORTCUT_DETECTED
BASELINE_EVAL_MISMATCH
TRAIN_BENCHMARK_LEAKAGE_DETECTED
DISTRACTOR_RESISTANCE_STILL_FAILS
CONTEXT_ENTITY_EXTRACTION_STILL_FAILS
COUNTERFACTUAL_BINDING_STILL_FAILS
LONG_CONTEXT_NEEDLE_STILL_FAILS
RETENTION_REGRESSION_DETECTED
ARM_COMPARISON_MISSING
OPEN_ENDED_CLAIM_DETECTED
FAILURE_CASE_REPORT_MISSING
STATIC_OUTPUT_COLLAPSE_DETECTED
CHECKPOINT_RELOAD_FAILS
ROLLBACK_REHEARSAL_FAILS
RESUME_FROM_CHECKPOINT_FAILS
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```
