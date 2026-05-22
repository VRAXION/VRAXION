# STABLE_LOOP_PHASE_LOCK_010_COINCIDENCE_CRYSTALLIZE_PRUNE Contract

## Question

009 showed that an audited local coincidence mutation lane makes spatial phase
construction reachable, but the result was dense / motif-rich.

010 asks:

```text
Does the dense coincidence forest contain a sparse, stable, causal phase-transport core?
```

This is a dissection probe, not another growth-first probe.

## Runner

Add a runner-local `instnct-core` example:

```text
instnct-core/examples/phase_lane_coincidence_crystallize_prune.rs
```

It reuses the 009 spatial phase-lane layout:

```text
arrive_phase[4]
gate_token[4]
emit_phase[4]
coincidence[4 input phases x 4 gates x 4 output phases]
```

No public `instnct-core` API changes are allowed.

## Data Splits

Use deterministic case splits per seed:

```text
prune_train_cases
prune_guard_cases
final_holdout_cases
```

`--eval-examples N` is split approximately 50/25/25, with a minimum of 32 cases
per split.

Prune decisions may use train and guard only. Sparse-core claims must pass final
holdout.

## Pipeline

Stages:

```text
DENSE_RAW
SNAP_ONLY
SNAP_THEN_PRUNE
```

Dense source:

```text
regenerate a deterministic dense spatial coincidence forest
```

Snap/crystallize canonicalizes coincidence neurons:

```text
threshold = 1
channel = 1
polarity = +1
```

If snap changes behavior below gates, pruning stops.

Prune unit is a complete local motif:

```text
arrive_phase_i(cell) -> coincidence_i_g_o(cell)
gate_token_g(cell)   -> coincidence_i_g_o(cell)
coincidence_i_g_o(cell) -> emit_phase_o(cell)
```

Greedy prune removes least harmful motifs while guard gates hold:

```text
guard accuracy >= 0.95
guard correct probability >= 0.90
guard counterfactual accuracy >= 0.85
gate shuffle still collapses
forbidden/private/nonlocal leaks = 0
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
prune_curve.jsonl
motif_ablation.jsonl
cell_ablation.jsonl
retained_motifs.jsonl
pruned_motifs.jsonl
repair_metrics.jsonl
counterfactual_metrics.jsonl
locality_audit.jsonl
seed_core_overlap.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

No black-box runs:

```text
progress.jsonl every <=30 sec
metrics.jsonl after checkpoints
prune_curve.jsonl after every prune step
summary.json/report.md refreshed on heartbeat
```

## Metrics

```text
motif_count_initial
motif_count_final
motif_prune_fraction
train_accuracy
guard_accuracy
final_holdout_accuracy
guard_correct_probability
guard_counterfactual_accuracy
gate_shuffle_collapse
single_gate_ablation_drop_mean/max
cell_ablation_drop_mean/max
minimum_motifs_to_95_accuracy
minimum_motifs_to_90_probability
retained_motif_type_frequency
seed_overlap_jaccard
common_core_motifs
seed_specific_motifs
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

## Verdicts

```text
DENSE_SOLUTION_REPRODUCED
DENSE_SOLUTION_REPRODUCTION_FAIL
SNAP_REVEALS_CANONICAL_MOTIF
SNAP_BREAKS_DENSE_SOLUTION
CRYSTALLIZE_PRESERVES_BEHAVIOR
CRYSTALLIZE_DOES_NOT_PRESERVE_BEHAVIOR
PRUNE_FINDS_SPARSE_CAUSAL_CORE
DENSE_OVERBUILT_BUT_PRUNABLE
DENSE_REQUIRED_NOT_PRUNABLE
PRUNE_OVERFITS_TRAIN_CASES
PRUNE_BREAKS_COUNTERFACTUALS
PRUNE_THEN_REPAIR_RESCUES
COST_PENALIZED_REPAIR_WORKS
RANDOM_PRUNE_FAILS
CAUSAL_MOTIF_CORE_IDENTIFIED
SPARSE_CORE_STABLE_ACROSS_SEEDS
SPARSE_CORE_NOT_STABLE
DIRECT_SHORTCUT_CONTAMINATION
PRODUCTION_API_NOT_READY
EXPERIMENTAL_MUTATION_LANE_SUPPORTED
```

## Decision Rules

```text
If dense reproduction fails:
  DENSE_SOLUTION_REPRODUCTION_FAIL

If snap drops below gates:
  SNAP_BREAKS_DENSE_SOLUTION

If snap preserves behavior:
  SNAP_REVEALS_CANONICAL_MOTIF

If prune removes >=50% motifs and passes final holdout:
  PRUNE_FINDS_SPARSE_CAUSAL_CORE

If train/guard pass but final holdout fails:
  PRUNE_OVERFITS_TRAIN_CASES

If accuracy passes but counterfactual fails:
  PRUNE_BREAKS_COUNTERFACTUALS

If retained motif-type Jaccard across seeds >= 0.60:
  SPARSE_CORE_STABLE_ACROSS_SEEDS

If retained motif-type Jaccard < 0.60:
  SPARSE_CORE_NOT_STABLE
```

## Claim Boundary

010 can support sparse causal-core discovery in this toy phase-lane substrate.

010 cannot prove full VRAXION, consciousness, language grounding, production
architecture, or Prismion uniqueness.
