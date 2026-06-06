# E7A13A Capture Radius Atlas Contract

## Purpose

E7A13A measures the mutation/rollback capture radius around a known good binary matrix-core seed. E7A12 showed that binary mutation from scratch was weak while local repair around a good seed was viable. This probe asks how far that local repair basin extends.

## Runner And Checker

- Runner: `scripts/probes/run_e7a13a_capture_radius_atlas.py`
- Checker: `scripts/probes/run_e7a13a_capture_radius_atlas_check.py`
- Default artifact root: `target/pilot_wave/e7a13a_capture_radius_atlas/`

## Required Shells

The center seed is a deterministically rebuilt block-scale binary QAT seed. The runner corrupts it at distances:

```text
0, 0.5%, 1%, 2%, 5%, 10%, 20%, 40%
```

Required corruption modes:

```text
random_bit_flip_shell
least_sensitive_bit_flip_shell
most_sensitive_bit_flip_shell
block_corruption_shell
scale_perturbation_shell
bits_plus_scale_corruption_shell
```

## Required Measurements

Each shell must report:

```text
raw_hamming_distance_to_center
normalized_hamming_distance_to_center
sensitivity_weighted_bit_distance_to_center
output_distance_to_center_seed
output_distance_to_teacher
seed_eval_before_repair
eval_gap_to_center_seed
eval_gap_to_qat_reference
```

Each repair run must report:

```text
seed_eval_before_repair
seed_eval_after_repair
repair_gain
accepted_mutations
rejected_mutations
rollback_count
mutation_attempts
budget_to_best_eval
recovered_to_within_epsilon
row_level_samples
```

Rejected mutations must equal rollback count.

## Budget Ladder

Default budget multipliers:

```text
1x, 4x
```

The 16x budget may be explicitly skipped with a reason to avoid a combinatorial run. It can be enabled by passing it in `--budget-multipliers`.

## Heartbeat Requirements

Long runs must write partial state continuously:

```text
progress.jsonl
partial_status/*.json
mutation_history_snapshots/*.json
current_best_candidate_summary/*.json
partial_aggregate_snapshot.json
```

## Required Final Artifacts

```text
backend_manifest.json
center_seed_report.json
shell_metrics.json
repair_metrics.json
capture_radius_report.json
falloff_model_report.json
deterministic_replay.json
checker_report.json
summary.json
final_summary.md
```

## Decisions

Allowed decision values:

```text
e7a13a_capture_radius_measured
e7a13a_invalid_artifact_detected
```

Allowed capture classifications:

```text
sharp_capture_boundary
smooth_falloff
ragged_island_basin
no_measurable_repair_basin
```

## Checker Gates

The checker fails on missing artifacts, missing shell modes, missing distances, missing budget multipliers or skipped-budget reasons, absent row-level samples, absent before/after repair fields, no accepted/rejected mutations, rollback mismatch, deterministic replay mismatch, forbidden repair backprop/optimizer calls, missing partial writeouts, or broad claims in the final report.

## Boundary

This is a controlled binary matrix-core capture-radius audit. It does not support broad claims outside this symbolic/numeric proxy.
