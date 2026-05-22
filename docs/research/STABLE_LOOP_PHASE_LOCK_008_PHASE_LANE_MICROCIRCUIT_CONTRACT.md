# STABLE_LOOP_PHASE_LOCK_008_PHASE_LANE_MICROCIRCUIT Contract

## Question

Can `instnct-core` represent, repair, and grow the missing local phase-lane microcircuit?

The isolated motif is:

```text
phase_i + gate_g -> phase_(i+g)
```

This must be tested before any more spatial/wavefield/routing experiments.

## Stages

```text
HAND_BUILT_PHASE_LANE_MOTIF
DAMAGE_REPAIR_MOTIF
PARTIAL_SEED_COMPLETION
RANDOM_GROWTH_BASELINE
EXPLICIT_COINCIDENCE_OPERATOR
```

`HAND_BUILT_PHASE_LANE_MOTIF` manually wires the minimal integer `instnct-core` circuit:

```text
4 phase input lanes
4 gate input lanes
16 coincidence neurons
4 output phase lanes
```

Each pair uses:

```text
phase_i -> coincidence_(i,g)
gate_g  -> coincidence_(i,g)
coincidence_(i,g) -> output_((i+g) mod 4)
```

Coincidence neurons use threshold 1 so a single phase or gate input is insufficient, but the pair fires.

## Damage And Growth

Damage repair starts from the hand-built motif and damages 1, 2, or 3 components:

```text
edge removed
threshold shifted
channel shifted
polarity flipped
```

Partial seed completion starts with only 4, 8, or 12 of the 16 correct phase/gate pairs wired.

Random growth starts from a small random local graph.

Explicit coincidence is a runner-local diagnostic operator:

```text
add_coincidence_gate(input_phase_lane, gate_lane, output_phase_lane)
```

It samples a local output lane and does not see:

```text
gate_sum
label
true_path
phase target
```

If only this operator works, the verdict is `EXPLICIT_COINCIDENCE_OPERATOR_REQUIRED`, not canonical mutation success.

## Metrics

```text
all_16_phase_gate_pairs_accuracy
single_step_phase_rotation_accuracy
motif_repair_success_rate
partial_seed_completion_rate
random_growth_success_rate
positive_candidate_delta_fraction
accepted_operator_rate
edge_count
threshold_sensitivity
channel_sensitivity
polarity_sensitivity
damage_level_success_curve
```

## Verdicts

```text
PHASE_LANE_MOTIF_REPRESENTABLE
MOTIF_REPAIRABLE_BY_CANONICAL_MUTATION
MOTIF_NOT_REPAIRABLE_BY_CANONICAL_MUTATION
PARTIAL_SEED_REQUIRED
RANDOM_GROWTH_SUCCEEDS
MOTIF_NOT_GROWABLE_FROM_RANDOM
EXPLICIT_COINCIDENCE_OPERATOR_REQUIRED
POLARITY_MUTATION_REQUIRED
CHANNEL_MUTATION_REQUIRED
REPRESENTATION_INSUFFICIENT
```

## Decision Rules

```text
If hand-built motif fails:
  REPRESENTATION_INSUFFICIENT

If hand-built motif succeeds but damage repair fails:
  MOTIF_NOT_REPAIRABLE_BY_CANONICAL_MUTATION

If damage repair succeeds but partial seed fails:
  PARTIAL_SEED_REQUIRED or motif assembly is too brittle

If partial seed succeeds but random growth fails:
  PARTIAL_SEED_REQUIRED

If random growth succeeds:
  RANDOM_GROWTH_SUCCEEDS

If only add_coincidence_gate works:
  EXPLICIT_COINCIDENCE_OPERATOR_REQUIRED
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
candidate_log.jsonl
operator_summary.json
motif_sensitivity.jsonl
damage_level_success_curve.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

No black-box rule:

```text
progress.jsonl every <=30 sec
metrics.jsonl after each checkpoint
candidate_log.jsonl continuously
summary.json/report.md refreshed on heartbeat
```

## Claim Boundary

008 can prove:

```text
the local phase-lane motif is representable
canonical mutation can/cannot repair it
partial seed or coincidence operator may be required
```

008 cannot prove:

```text
full spatial phase-lock solved
full VRAXION solved
consciousness
language grounding
Prismion uniqueness
```

Only after 008 passes should the motif be reinserted into the spatial wavefield grid.
