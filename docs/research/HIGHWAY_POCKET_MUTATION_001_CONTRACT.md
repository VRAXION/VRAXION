# HIGHWAY_POCKET_MUTATION_001 Contract

## Purpose

Test whether a protected main highway plus gated sidepockets can evolve specialist correction behavior without damaging the protected signal path.

This is a runner-local mutation smoke. It does not promote sidepockets into `instnct-core` public APIs yet.

## Architecture

The probe uses fixed zones:

```text
input zone
protected main highway zone
sidepocket zones P0..Pk
writeback gate zone
readout zone
```

Rules:

```text
pockets may read input/highway
pockets may mutate internally
pockets write back only through explicit gates
pockets cannot directly write output/readout
highway is protected except optional repair edges
```

Mutation operators:

```text
pocket_add_internal_edge
pocket_rewire_internal_edge
pocket_add_loop2
pocket_add_loop3
pocket_mutate_gate_threshold
pocket_mutate_gate_channel
pocket_flip_gate_polarity
pocket_add_read_tap
pocket_move_writeback
highway_repair_edge
```

## Task Blocks

### Symbolic Correction Smoke

Abstract streams:

```text
A, B, C
anti_A, anti_B, anti_C
reset
mention_A, quote_anti_A
actually_B, instead_B
create_X, remove_X, restore_X, query_count
noise
```

Baselines:

```text
HIGHWAY_ONLY
HIGHWAY_WITH_RANDOM_POCKETS_NO_WRITEBACK
HIGHWAY_WITH_UNGATED_POCKETS
HIGHWAY_WITH_GATED_POCKETS
UNRESTRICTED_GRAPH_MUTATION
```

Metrics:

```text
final_answer_accuracy
heldout_composition_accuracy
length_generalization_accuracy
mention_noop_error_rate
refocus_accuracy
false_mutation_rate
false_cancellation_rate
highway_retention_accuracy
pocket_ablation_max_drop
pocket_ablation_mean_drop
pocket_writeback_sparsity
accepted_operator_rate
destructive_mutation_rate
```

### PHASE_LOCK_MICRO_BRIDGE

This bridge keeps the run connected to the phase-lock 002/003 blocker.

Setup:

```text
fixed protected highway carries phase state
sidepockets read highway/gate tokens
sidepockets write back only through gates
final phase answer only
no intermediate phase labels
```

Variants:

```text
HIGHWAY_ONLY_PHASE
HIGHWAY_WITH_RANDOM_POCKETS_NO_WRITEBACK_PHASE
HIGHWAY_WITH_GATED_POCKETS_PHASE
UNRESTRICTED_GRAPH_MUTATION_PHASE
```

Metrics:

```text
phase_final_accuracy
heldout_path_length_accuracy
phase_gate_shuffle_control
highway_phase_retention
pocket_ablation_phase_drop
```

## Verdicts

```text
HIGHWAY_POCKET_MUTATION_POSITIVE
POCKETS_DECORATIVE
UNGATED_POCKETS_SUFFICIENT
UNRESTRICTED_GRAPH_SUFFICIENT
HIGHWAY_DESTROYED_BY_POCKETS
MUTATION_SEARCH_TOO_WEAK
TASK_TOO_EASY
TASK_TOO_HARD
MUTATION_RESCUES_PHASE_CREDIT_ASSIGNMENT
PHASE_BRIDGE_NO_SIGNAL
POCKETS_HELP_SYMBOLIC_BUT_NOT_PHASE
TASK_TOO_EASY_FOR_PHASE_BRIDGE
```

Positive gate:

```text
HIGHWAY_WITH_GATED_POCKETS beats HIGHWAY_ONLY by >= +0.05
on correction/no-op/refocus,
highway_retention drops <= 0.02,
pocket ablation shows nontrivial contribution,
and destructive_mutation_rate stays controlled.
```

Phase bridge positive gate:

```text
HIGHWAY_WITH_GATED_POCKETS_PHASE beats HIGHWAY_ONLY_PHASE
and random/no-writeback phase controls,
phase_gate_shuffle_control degrades,
highway_phase_retention stays clean,
and pocket_ablation_phase_drop is nontrivial.
```

## Run Hygiene

No black-box runs. Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
candidate_log.jsonl
operator_summary.json
pocket_ablation.jsonl
phase_bridge_metrics.jsonl
summary.json
report.md
contract_snapshot.md
job_progress/*.jsonl
```

Raw `target/` outputs are not committed.

## Claim Boundary

This probe tests mutation-selected protected topology and a toy phase bridge only.

It does not prove consciousness, full VRAXION, language grounding, or a production sidepocket architecture.
