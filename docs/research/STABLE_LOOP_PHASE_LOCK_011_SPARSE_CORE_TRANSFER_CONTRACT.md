# STABLE_LOOP_PHASE_LOCK_011_SPARSE_CORE_TRANSFER Contract

## Summary

011 tests whether the sparse motif types found by 010 transfer as a reusable
local phase-transport rule, rather than a pruned residue specific to the 010
case distribution.

The runner is evaluation-only:

```text
no new growth
no new pruning
no public instnct-core API changes
no raw target output dependency
```

## Motif Templates

The official runner embeds the 010 common core from the sanitized result doc:

```text
COMMON_CORE_15 =
  0_0_0 0_1_1 0_2_2 0_3_3
  1_0_1 1_1_2 1_3_0
  2_0_2 2_1_3 2_2_0 2_3_1
  3_0_3 3_1_0 3_2_1 3_3_2

MISSING_MOTIF = 1_2_3
COMMON_CORE_15_PLUS_MISSING_1_2_3 = COMMON_CORE_15 + 1_2_3
FULL_16_RULE_TEMPLATE = all phase_i + gate_g -> phase_(i+g)
```

The missing `1_2_3` motif is a required adversarial diagnostic. Overall
accuracy must not hide a dead phase/gate pair.

## Arms

```text
FIXED_PHASE_LANE_REFERENCE
DENSE_009_REFERENCE
COMMON_CORE_15
COMMON_CORE_15_PLUS_MISSING_1_2_3
FULL_16_RULE_TEMPLATE
SEED_SPECIFIC_CORE_REINSERTED
RANDOM_MATCHED_15_MOTIF_CONTROL
RANDOM_MATCHED_16_MOTIF_CONTROL
RANDOM_MATCHED_25_MOTIF_CONTROL
CANONICAL_JACKPOT_007_BASELINE
```

Main placement is `template_on_all_free_cells`. Diagnostic placements include
frontier-reachable cells, random matched cell count, and private path cells for
analysis only.

## Evaluation Families

```text
width_8
width_10
width_12
width_16
short_path
medium_path
long_path
reverse_path
distractor_corridor
damaged_corridor
same_target_counterfactual
all_16_phase_gate_pair_coverage
gate_shuffle_control
```

Path length is capped to the recurrent horizon where `DENSE_009_REFERENCE`
remains valid. If the dense/full16 reference fails, the transfer task is invalid
for sparse-core claims.

## Metrics

```text
phase_final_accuracy
correct_target_lane_probability_mean
same_target_counterfactual_accuracy
gate_shuffle_collapse
width_transfer_accuracy
long_path_transfer_accuracy
new_layout_transfer_accuracy
reverse_path_consistency_accuracy
per_pair_accuracy[input_phase, gate]
per_pair_correct_probability[input_phase, gate]
min_per_pair_accuracy
motif_type_count
instantiated_motif_count
accuracy_per_motif
probability_per_motif
motif_ablation_drop
nonlocal_edge_count
forbidden_private_field_leak
direct_output_leak_rate
```

Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
family_metrics.jsonl
template_metrics.jsonl
per_pair_metrics.jsonl
random_control_metrics.jsonl
counterfactual_metrics.jsonl
locality_audit.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

No black-box rule:

```text
progress.jsonl every <=30 sec
metrics after each arm/family block
summary.json/report.md refreshed on heartbeat
```

## Verdicts

```text
SPARSE_CORE_TRANSFER_POSITIVE
SPARSE_CORE_TRANSFER_FAILS
COMMON_TEMPLATE_WORKS
COMMON_CORE_WAS_ONE_MOTIF_SHORT
SEED_SPECIFIC_ONLY
FULL_16_REQUIRED
DENSE_REFERENCE_STILL_REQUIRED
RANDOM_MOTIF_CONTROL_FAILS
RANDOM_MOTIF_CONTROL_TOO_STRONG
WIDTH_TRANSFER_PASSES
LONG_PATH_TRANSFER_PASSES
LAYOUT_TRANSFER_PASSES
TASK_OR_DENSE_REFERENCE_INVALID
PRODUCTION_API_NOT_READY
EXPERIMENTAL_MUTATION_LANE_SUPPORTED
```

Positive sparse-rule claim requires:

```text
DENSE_009_REFERENCE accuracy >= 0.95
main sparse arm overall accuracy >= 0.90
width/layout/long-path family accuracy >= 0.85
same_target_counterfactual_accuracy >= 0.85
all per-pair accuracies >= 0.80
gate_shuffle_collapse >= 0.50
random matched controls trail sparse template by >= 0.10
forbidden/private/nonlocal/direct leaks = 0
```

Decision rules:

```text
COMMON_CORE_15 passes:
  COMMON_TEMPLATE_WORKS
  SPARSE_CORE_TRANSFER_POSITIVE

COMMON_CORE_15 fails but +1_2_3 passes:
  COMMON_CORE_WAS_ONE_MOTIF_SHORT
  SPARSE_CORE_TRANSFER_POSITIVE

Only FULL_16_RULE_TEMPLATE passes:
  FULL_16_REQUIRED

Dense passes but sparse/full16 fail:
  DENSE_REFERENCE_STILL_REQUIRED

Random matched controls pass:
  RANDOM_MOTIF_CONTROL_TOO_STRONG
```

## Claim Boundary

011 can support reusable sparse phase-lane motif transfer in toy spatial tasks.
It cannot prove production architecture, full VRAXION, consciousness, language
grounding, or Prismion uniqueness.
