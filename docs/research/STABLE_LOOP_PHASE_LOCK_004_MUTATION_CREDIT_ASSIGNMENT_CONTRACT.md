# STABLE_LOOP_PHASE_LOCK_004_MUTATION_CREDIT_ASSIGNMENT Contract

## Purpose

Test whether mutation-selection can solve spatial phase-lock credit assignment without `gate_sum`, direct oracle fields, or named phase rules.

This is the direct follow-up to the 002/003 phase-lock blocker and the adversarial caveat from `HIGHWAY_POCKET_MUTATION_001`.

## Case Separation

`PrivateCase` may contain:

```text
label
true_path
path_phase_total
gate_sum
oracle routing info
family
```

`PublicCase` may contain only:

```text
wall/free mask
source location + source phase
target marker/location
per-cell local gate vectors
local frontier/reached state through propagation
```

Candidate prediction may receive only `PublicCase`. The evaluator may use only `PrivateCase.label`.

Forbidden prediction inputs:

```text
gate_sum
path_phase_total
true_path
shortest_path_phase_total
label
answer
oracle_phase_bucket
family id as model input
```

Any exposure triggers:

```text
DIRECT_SHORTCUT_CONTAMINATION
```

## Arms

```text
ORACLE_SPATIAL_PHASE_LOCK
HIGHWAY_ONLY_PHASE
HIGHWAY_WITH_RANDOM_POCKETS_NO_WRITEBACK_PHASE
HIGHWAY_WITH_GATED_POCKETS_PHASE
HIGHWAY_WITH_UNGATED_POCKETS_PHASE
UNRESTRICTED_GRAPH_MUTATION_PHASE
FIXED_COMPLEX_MULTIPLY_LOCAL_REFERENCE
ORACLE_ROUTING_MUTABLE_PHASE_POCKET
MUTABLE_ROUTING_ORACLE_PHASE
MUTABLE_ROUTING_MUTABLE_PHASE
```

`UNRESTRICTED_GRAPH_MUTATION_PHASE` removes protected highway/sidepocket restrictions but remains local: no global pooling, no flattening, no direct answer access, and no direct output write from non-target cells.

## Mutation Rules

Allowed local circuit mutations:

```text
add_local_edge
remove_local_edge
rewire_local_edge
mutate_gate_threshold
mutate_gate_channel
flip_gate_polarity
add_local_loop2
add_local_loop3
move_read_tap_local
move_writeback_local
```

Forbidden:

```text
phase_gate_compose rule
direct_phase_oracle
direct_oracle
pred = (source_phase + gate_sum) % 4
```

## Metrics

```text
phase_final_accuracy
heldout_path_length_accuracy
paired_counterfactual_accuracy
paired_counterfactual_margin
gate_shuffle_collapse
target_shuffle_collapse
wall_shuffle_degradation
highway_phase_retention
pocket_ablation_phase_drop
direct_output_leak_rate
wall_leak_rate
pre_wall_pressure
accepted_operator_rate
destructive_mutation_rate
```

Accepted-candidate locality audit:

```text
max_edge_distance
writes_to_target_readout_directly
reads_nonlocal_cell
writes_wall_cell
uses_forbidden_private_field
```

## Verdicts

```text
MUTATION_RESCUES_PHASE_CREDIT_ASSIGNMENT
PHASE_CREDIT_ASSIGNMENT_NOT_SOLVED
ROUTING_IS_BLOCKER
PHASE_TRANSPORT_IS_BLOCKER
ROUTING_PHASE_INTERACTION_IS_BLOCKER
GATED_POCKETS_UNIQUELY_HELPFUL
UNGATED_POCKETS_SUFFICIENT
UNRESTRICTED_GRAPH_SUFFICIENT
DIRECT_SHORTCUT_CONTAMINATION
TASK_TOO_EASY
TASK_TOO_HARD
```

Positive gate:

```text
gated or unrestricted local mutation beats highway-only by >= +0.05
beats random/no-writeback controls
gate shuffle collapses performance
paired counterfactual accuracy >= 0.85
pocket ablation drop >= 0.05 for pocket arms
highway retention >= 0.98
wall leak <= 0.02
direct_output_leak_rate near zero
no forbidden-field/locality audit failure
```

## Run Hygiene

Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
candidate_log.jsonl
operator_summary.json
pocket_ablation.jsonl
phase_credit_split_metrics.jsonl
counterfactual_metrics.jsonl
locality_audit.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

Raw `target/` outputs are not committed.

## Claim Boundary

This probe tests local spatial phase credit assignment only. It does not prove consciousness, full VRAXION, language grounding, or production sidepockets.
