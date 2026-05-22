# STABLE_LOOP_PHASE_LOCK_071_REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM Result

Status: implementation result document for eval-only fresh confirmation of the
070 repaired checkpoint.

071 benchmarks only the 070 `FINETUNE_068_TARGETED_REPAIR` checkpoint. It uses
fresh hard-distractor, counterfactual, near-miss, pocket-suppression,
negative-route, and long-context rows to check whether the 070 repair
generalizes beyond the 070 training/eval shape.

This is eval-only.

no training
no checkpoint repair
no open-ended assistant
no free-form generation
no perplexity
no full English LM
no language grounding
no production training
no GA
no public beta
no hosted SaaS
no service API change
no deployment harness change
no release docs change
no public crate export change
no root LICENSE change

## Implementation Summary

Runner:

```text
instnct-core/examples/phase_lane_repaired_checkpoint_capability_confirm.rs
```

Static checker:

```text
scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py
```

The runner verifies the 070 summary, checkpoint manifest, arm comparison, and
069 benchmark root. It loads the best-arm checkpoint, records
`checkpoint_hash_before` and `checkpoint_hash_after`, evaluates model,
baselines, and no-route control on identical supported rows, writes
human-readable samples, and records limitation honesty.

It records:

```text
train_step_count = 0
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
prediction_oracle_used = false
open_ended_generation_supported = false
free_form_answering_supported = false
perplexity_supported = false
finite_label_surface = true
closed-label success does not imply language grounding
this is not an open-ended assistant
overlap_with_069_samples_count
overlap_with_070_samples_count
upstream_exact_overlap_audit_limited
eval_row_hash_model
eval_row_hash_baselines
eval_row_hash_no_route_control
baseline_eval_mismatch = false
delta_vs_no_route_control
```

## Required Commands

```powershell
cargo check -p instnct-core --example phase_lane_repaired_checkpoint_capability_confirm
cargo run -p instnct-core --example phase_lane_repaired_checkpoint_capability_confirm -- --out target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoints/finetune_068_targeted_repair/model_checkpoint.json --upstream-070-root target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --benchmark-069-root target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke --seed 2027 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py --check-only
cargo test -p instnct-core sdk_candidate
git diff --check
```

## Required Artifacts

Generated artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/
```

Required artifact names:

```text
queue.json
progress.jsonl
benchmark_config.json
upstream_070_manifest.json
checkpoint_manifest.json
capability_dataset_manifest.json
benchmark_examples_sample.jsonl
baseline_metrics.json
no_route_feature_control_metrics.json
capability_metrics.json
per_family_metrics.json
limitation_report.json
human_readable_samples.jsonl
failure_case_samples.jsonl
collapse_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from the start, so
071 is not a black-box run.

## Benchmark Families

```text
FRESH_CONTEXT_ENTITY_EXTRACTION
FRESH_COUNTERFACTUAL_BINDING
FRESH_DISTRACTOR_RESISTANCE
FRESH_LONG_CONTEXT_NEEDLE_BINDING
FRESH_NEAR_MISS_ANCHOR_SELECTION
FRESH_IRRELEVANT_POCKET_SUPPRESSION
FRESH_NEGATIVE_ROUTE_REJECTION
RETENTION_INSTRUCTION_FOLLOWING_CLOSED
RETENTION_MULTI_HOP_KEY_VALUE_BINDING
RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE
RETENTION_NON_ROUTE_TEXT_CONTROL
OPEN_ENDED_INTERFACE_LIMITATION
```

Human-readable samples include:

```text
task_family
input
expected_output
model_output
baseline_outputs
no_route_output
pass_fail
limitation_flag
```

`failure_case_samples.jsonl` exists even when empty.

## Verdict Semantics

Positive result requires the 070 best-arm checkpoint to verify, no training side
effects, unchanged checkpoint hash, no obvious 069/070 sample leakage, same-row
baseline/control evaluation, no collapse, family gates, retention gates, and
human-readable samples.

Failure is acceptable and must be preserved. If any hard gate fails, the runner
does not emit `REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_POSITIVE`.

## Observed Smoke Result

The eval-only 071 smoke completed and produced an honest failure:

```text
REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_FAILS
FRESH_HARD_DISTRACTOR_GENERALIZATION_FAILS
FRESH_COUNTERFACTUAL_GENERALIZATION_FAILS
CAPABILITY_FAMILY_GATE_FAILS
OPEN_ENDED_LIMITATION_RECORDED
NO_TRAINING_PERFORMED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Observed capability metrics:

```text
supported_accuracy = 0.822289156626506
family_min_accuracy = 0.06666666666666667
fresh_context_entity_extraction_accuracy = 0.7
fresh_counterfactual_binding_accuracy = 0.06666666666666667
fresh_distractor_resistance_accuracy = 0.9333333333333333
fresh_long_context_needle_accuracy = 0.9
fresh_near_miss_anchor_selection_accuracy = 0.9333333333333333
fresh_irrelevant_pocket_suppression_accuracy = 0.6666666666666666
fresh_negative_route_rejection_accuracy = 1.0
retention_instruction_following_closed_accuracy = 1.0
retention_multi_hop_key_value_accuracy = 0.8333333333333334
retention_symbolic_rule_closed_choice_accuracy = 1.0
retention_non_route_text_control_accuracy = 1.0
delta_vs_majority = 0.7228915662650602
delta_vs_copy_first_match = 0.5843373493975904
delta_vs_no_route_control = 0.027108433734939763
```

Integrity observations:

```text
train_step_count = 0
checkpoint_hash_unchanged = true
prediction_oracle_used = false
baseline_eval_mismatch = false
overlap_with_069_samples_count = 0
overlap_with_070_samples_count = 0
upstream_exact_overlap_audit_limited = true
```

Interpretation:

```text
passes:
  fresh distractor resistance
  fresh long-context needle
  fresh near-miss anchor selection
  fresh negative-route rejection
  retention families

fails:
  fresh counterfactual binding
  fresh context entity extraction
  fresh irrelevant pocket suppression
```

The dominant failure mode is fresh counterfactual wording. The repaired
checkpoint often selects the distractor scenario value instead of the active
scenario value. This supports the next milestone:

```text
071B_REPAIR_OVERFIT_FAILURE_ANALYSIS
```

071 therefore confirms that 070 repaired important capability families, but the
repair does not yet generalize across the fresh counterfactual/context template
shift used in this gate.

Positive verdicts include:

```text
REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_POSITIVE
UPSTREAM_070_CHECKPOINT_VERIFIED
NO_TRAINING_PERFORMED
CHECKPOINT_UNCHANGED
FRESH_HARD_DISTRACTOR_GENERALIZATION_PASSES
FRESH_COUNTERFACTUAL_GENERALIZATION_PASSES
FRESH_LONG_CONTEXT_GENERALIZATION_PASSES
RETENTION_CONFIRM_PASSES
NO_ROUTE_CONTROL_RECORDED
BASELINE_COMPARISON_RECORDED
HUMAN_READABLE_SAMPLES_WRITTEN
OPEN_ENDED_LIMITATION_RECORDED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Failure verdicts include:

```text
REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_FAILS
UPSTREAM_070_ARTIFACT_MISSING
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
ORACLE_SHORTCUT_DETECTED
BENCHMARK_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
NO_ROUTE_CONTROL_MISSING
FRESH_HARD_DISTRACTOR_GENERALIZATION_FAILS
FRESH_COUNTERFACTUAL_GENERALIZATION_FAILS
FRESH_LONG_CONTEXT_GENERALIZATION_FAILS
RETENTION_CONFIRM_FAILS
CAPABILITY_FAMILY_GATE_FAILS
STATIC_OUTPUT_COLLAPSE_DETECTED
HUMAN_SAMPLE_REPORT_MISSING
OPEN_ENDED_CLAIM_DETECTED
PERPLEXITY_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```
