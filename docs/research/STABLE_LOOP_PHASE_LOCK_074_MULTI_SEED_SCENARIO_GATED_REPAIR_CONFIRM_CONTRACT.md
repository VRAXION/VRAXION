# STABLE_LOOP_PHASE_LOCK_074_MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM Contract

Status: contract for multi-seed eval-only confirmation of the 072
`SCENARIO_GATED_SIDEPACKET_REPAIR` checkpoint.

074 orchestrates fresh 073 child eval runs for seeds `2027,2028,2029` to
verify that the 073 fresh pass was not a single-seed lucky pass.

This is multi-seed eval only.

no training
no checkpoint repair
no checkpoint mutation
no stale child artifacts
no mean-only pass
no open-ended assistant
no free-form generation
no perplexity
no full English LM
no language grounding
no production training
no release readiness by itself
no service API change
no deployment harness change
no release docs change
no public crate export change
no root LICENSE change

## Implementation Scope

074 adds only:

```text
scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm.py
scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_074_MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_074_MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_RESULT.md
```

It reuses the existing 073 eval runner:

```text
phase_lane_scenario_gated_repair_fresh_confirm
```

Generated outputs are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/
```

## Upstream Inputs

Default checkpoint:

```text
target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json
```

Required upstream roots:

```text
target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke
target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke
target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke
target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke
```

If any required upstream artifact is missing, emit:

```text
UPSTREAM_072_ARTIFACT_MISSING
```

Do not rerun 070/071/071B/072 and do not train or repair a replacement
checkpoint.

## Child Run Freshness

For every seed, 074 records and requires:

```text
child_run_started = true
child_run_completed = true
child_exit_code = 0
child_summary_newer_than_074_start = true
child_report_newer_than_074_start = true
child_command recorded exactly
```

Failure if stale artifacts are reused:

```text
STALE_CHILD_ARTIFACT_USED
```

The required seeds are exactly:

```text
2027,2028,2029
```

No mean-only pass is allowed. A best-seed pass or 2/3 seed pass fails with:

```text
MULTI_SEED_SCENARIO_INSTABILITY_DETECTED
```

## Child 073 Gate Recheck

Every child 073 run is independently parsed and rechecked for:

```text
SCENARIO_GATED_REPAIR_FRESH_CONFIRM_POSITIVE
train_step_count = 0
checkpoint_hash_unchanged = true
prediction_oracle_used = false
baseline_eval_mismatch = false
all overlap counts = 0
collapse_detected = false
```

Failure verdict:

```text
CHILD_073_GATE_RECHECK_FAILS
```

## Per-Seed Gates

Every seed must satisfy:

```text
fresh_active_scenario_binding_accuracy >= 0.90
fresh_counterfactual_scenario_switch_accuracy >= 0.85
fresh_distractor_scenario_rejection_accuracy >= 0.90
fresh_old_scenario_suppression_accuracy >= 0.90
fresh_inactive_pocket_suppression_accuracy >= 0.85
fresh_stale_pocket_suppression_accuracy >= 0.85
fresh_answer_only_scenario_binding_accuracy >= 0.85
family_min_accuracy >= 0.85
supported_accuracy >= 0.88
active_scenario_selection_accuracy >= 0.95
distractor_scenario_selection_rate <= 0.05
old_scenario_selection_rate <= 0.05
delta_vs_no_route_control > 0.10
delta_vs_ungated_sidepacket_control > 0.03
delta_vs_copy_first_match > 0.10
shuffled_scenario_label_control_accuracy < 0.70
collapse_detected = false
```

If any seed loses the gated advantage, emit:

```text
GATED_ADVANTAGE_REGRESSION_DETECTED
```

If any seed has a passing or near-passing shuffled scenario label control,
emit:

```text
SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS
```

## Scenario Source Attribution

The aggregate must record per-seed values plus min/max for:

```text
active_scenario_selection_accuracy
distractor_scenario_selection_rate
old_scenario_selection_rate
inactive_pocket_selection_rate
stale_pocket_selection_rate
first_ledger_bias_rate
side_note_leak_rate
```

## Required Artifacts

Aggregate artifacts:

```text
queue.json
progress.jsonl
multi_seed_config.json
upstream_072_manifest.json
child_run_manifest.json
seed_metrics.jsonl
aggregate_metrics.json
multi_seed_stability.json
baseline_knockout_aggregate.json
scenario_source_attribution_aggregate.json
retention_aggregate.json
failure_case_samples.jsonl
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from start and
refreshed after each seed.

`failure_case_samples.jsonl` must exist even if empty. If any child has failed
rows, include:

```text
seed
task_family
input
expected_output
model_output
baseline_outputs
no_route_output
ungated_control_output
shuffled_control_output
wrong_answer_source
short_diagnosis
```

Required honesty fields:

```text
multi_seed_eval_only = true
train_step_count = 0
open_ended_generation_supported = false
free_form_answering_supported = false
perplexity_supported = false
full_English_LM_supported = false
language_grounding_claimed = false
production_training_claimed = false
```

## Validation Commands

```powershell
cargo check -p instnct-core --example phase_lane_scenario_gated_repair_fresh_confirm
python -m py_compile scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm.py --out target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json --upstream-072-root target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke --upstream-071b-root target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke --upstream-071-root target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke --upstream-070-root target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --seeds 2027,2028,2029 --heartbeat-sec 20
python scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py --check-only
git diff --check
```

## Verdicts

Positive verdicts:

```text
MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE
FRESH_CHILD_RUNS_CONFIRMED
CHILD_073_GATES_RECHECKED
MULTI_SEED_MIN_GATE_PASSES
SCENARIO_GATED_REPAIR_STABLE_ACROSS_SEEDS
FRESH_GATED_ADVANTAGE_STABLE
SCENARIO_SOURCE_ATTRIBUTION_AGGREGATED
SHUFFLED_SCENARIO_CONTROL_FAILS_ALL_SEEDS
CHECKPOINT_UNCHANGED_ALL_SEEDS
NO_TRAINING_PERFORMED
FRESH_EVAL_LEAKAGE_REJECTED_ALL_SEEDS
BASELINE_COMPARISON_RECORDED
FAILURE_CASE_REPORT_WRITTEN
OPEN_ENDED_LIMITATION_RECORDED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Failure verdicts:

```text
MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_FAILS
UPSTREAM_072_ARTIFACT_MISSING
STALE_CHILD_ARTIFACT_USED
CHILD_073_GATE_RECHECK_FAILS
MULTI_SEED_SCENARIO_INSTABILITY_DETECTED
GATED_ADVANTAGE_REGRESSION_DETECTED
SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
ORACLE_SHORTCUT_DETECTED
BENCHMARK_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
STATIC_OUTPUT_COLLAPSE_DETECTED
FAILURE_CASE_REPORT_MISSING
OPEN_ENDED_CLAIM_DETECTED
PERPLEXITY_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

074 pass means multi-seed bounded scenario-state confirmation. It means no
general intelligence claim, no open-ended generation claim, no production
training claim, and no release readiness claim.
