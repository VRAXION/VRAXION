# STABLE_LOOP_PHASE_LOCK_009_COINCIDENCE_OPERATOR_SPATIAL_REINSERTION Contract

## Question

Does the audited local coincidence operator from 008 make the spatial phase-lane
construction reachable when reinserted into the 007-style phase-lane substrate?

The isolated local motif is:

```text
phase_i + gate_g -> phase_(i+g)
```

009 tests whether the 007 failure was mainly caused by canonical mutation being
unable to assemble this local motif.

## Scope

This is a runner-local experiment. It uses `instnct-core::Network` directly, but
does not change public `instnct-core` APIs.

The coincidence, polarity, and channel operators in this runner are diagnostic
mutation lanes only.

## Spatial Cell Layout

Each spatial cell has:

```text
arrive_phase[4]
gate_token[4]
emit_phase[4]
coincidence[4 input phases x 4 gates x 4 output phases]
```

Spatial propagation is local:

```text
emit_phase_k(cell) -> arrive_phase_k(local neighbor)
```

The local operator:

```text
add_coincidence_gate(cell, input_phase, gate, output_phase)
```

creates only:

```text
arrive_phase_i(cell) -> coincidence_i_g_o(cell)
gate_token_g(cell)   -> coincidence_i_g_o(cell)
coincidence_i_g_o(cell) -> emit_phase_o(cell)
```

The operator does not read:

```text
gate_sum
label
true_path
phase target
oracle phase bucket
```

## Arms

```text
FIXED_PHASE_LANE_REFERENCE
HAND_BUILT_SPATIAL_COINCIDENCE_REFERENCE
CANONICAL_JACKPOT_007_BASELINE
ORACLE_ROUTING_PLUS_COINCIDENCE_OPERATOR
FULL_SPATIAL_PLUS_COINCIDENCE_OPERATOR
COINCIDENCE_OPERATOR_STRICT
COINCIDENCE_OPERATOR_TIES
COINCIDENCE_OPERATOR_ZEROP
POLARITY_ONLY
CHANNEL_ONLY
COINCIDENCE_PLUS_POLARITY
COINCIDENCE_PLUS_CHANNEL
COINCIDENCE_PLUS_POLARITY_CHANNEL
```

`ORACLE_ROUTING_PLUS_COINCIDENCE_OPERATOR` may sample motif locations from the
private path for diagnostic placement only.

The full spatial coincidence arms may sample only public free cells from the
wall/free mask.

## Hand-Built Gate

The hand-built spatial reference runs first. If it fails, mutation stages must
stop and the result is:

```text
HAND_BUILT_SPATIAL_MOTIF_FAILS
```

Pass gate:

```text
phase_final_accuracy >= 0.95
correct_target_lane_probability_mean >= 0.90
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse strong
wall_leak = 0
nonlocal_edge_count = 0
```

## Required Audits

For every accepted `add_coincidence_gate`, log:

```text
cell_id
input_phase
gate
output_phase
is_on_path_cell
is_target_cell
writes_to_target_directly
reads_private_path
reads_private_label
local_edge_only
```

Required motif generality metrics:

```text
unique_cells_with_motif
motif_density_per_cell
full_16_pair_coverage_per_cell
motif_reuse_across_examples
motif_ablation_drop_by_cell
total_coincidence_gates_added
useful_coincidence_gates
useless_coincidence_gates
motif_precision
motif_recall
```

If success requires dense all-cell/all-pair motifs, report:

```text
COINCIDENCE_OPERATOR_RESCUES_SPATIAL_PHASE_DENSE
```

not efficient learned motif.

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
candidate_log.jsonl
operator_summary.json
spatial_stage_metrics.jsonl
ablation_metrics.jsonl
counterfactual_metrics.jsonl
motif_placement_audit.jsonl
locality_audit.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

The runner must continuously write partial outputs; no black-box run is allowed.

## Verdicts

```text
HAND_BUILT_SPATIAL_MOTIF_WORKS
HAND_BUILT_SPATIAL_MOTIF_FAILS
COINCIDENCE_OPERATOR_RESCUES_SPATIAL_PHASE
COINCIDENCE_OPERATOR_RESCUES_SPATIAL_PHASE_DENSE
COINCIDENCE_OPERATOR_RESCUES_SHORT_CHAIN_ONLY
SPATIAL_REINSERTION_FAILS
POLARITY_OPERATOR_REQUIRED
CHANNEL_OPERATOR_REQUIRED
CANONICAL_JACKPOT_STILL_INSUFFICIENT
DIRECT_SHORTCUT_CONTAMINATION
```

## Claim Boundary

009 can support that an audited local coincidence mutation lane makes spatial
phase construction reachable in this toy phase-lane substrate.

009 cannot prove production efficiency, minimality, full VRAXION, consciousness,
language grounding, or Prismion uniqueness.
