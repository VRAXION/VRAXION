# E41 Logic Atom Genome Grow/Shrink And Commit Probe Contract

Milestone:

```text
E41_LOGIC_ATOM_GENOME_GROW_SHRINK_AND_COMMIT_PROBE
```

## Purpose

E40 showed that fixed-slot mutable Logic Atom rules can learn local conditional
router behavior. E41 tests the next claim:

```text
LogicAtom should emit proposals, not directly overwrite Flow.
An Arbiter should commit, reject, or defer.
The LogicAtom genome should be able to grow/shrink small proposal rules.
```

This is a controlled spatial Flow-grid proxy. It is not a raw language,
deployed-model, AGI, consciousness, or model-scale claim.

## Task

Each row contains visible local condition cells:

```text
a bit
b bit
c bit
blocker bit
missing bit
guard cell
target patch
```

The E40 scale/op local logic remains available. E41 adds action state:

```text
missing == 1 -> DEFER
blocker == 1 -> REJECT
otherwise    -> WRITE
```

The target Flow changes only for committed WRITE rows. REJECT and DEFER should
produce no direct Flow write.

## Systems

```text
oracle_proposal_commit_reference
direct_write_logic_atom_baseline
proposal_without_arbiter
fixed_slot_proposal_arbiter
grow_shrink_logic_atom_genome
full_flow_painter_control
random_genome_control
```

The oracle reference is ineligible for learned comparison.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
proposal_commit_report.json
logic_atom_genome_report.json
system_results.json
footprint_report.json
mutation_report.json
row_level_results.jsonl
footprint_frames.jsonl
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

Sample pack:

```text
docs/research/artifact_samples/e41_logic_atom_genome_grow_shrink_and_commit_probe/
```

## Metrics

```text
exact_rate
action_accuracy
false_commit_rate
missed_commit_rate
proposal_count_mean
read_spread_ratio
write_spread_ratio
accepted/rejected/rollback mutation counts
parameter diff/hash
deterministic replay hash match
```

## Decision Labels

```text
e41_logic_atom_grow_shrink_commit_positive
e41_fixed_slots_sufficient_growth_not_needed
e41_direct_write_sufficient
e41_arbiter_required_but_growth_failed
e41_full_flow_required
e41_invalid_artifact_detected
```

Positive requires:

```text
grow_shrink exact >= 0.95
grow_shrink action_accuracy >= 0.95
grow_shrink false_commit_rate <= 0.03
grow_shrink missed_commit_rate <= 0.03
direct_write and no-arbiter controls fail
fixed slot arbiter reference passes
random control action accuracy stays low; grid exact alone is not a valid random-control gate because no-write rows can inflate exact matches
full-flow diagnostic is diffuse
checker failure_count = 0
sample-only checker passes
deterministic replay passes
```

## Hard Requirements

No gradient descent, optimizer, backprop, direct solver, or hidden oracle access
inside valid primary systems. Long-ish runs must write progress/heartbeat and
mutation history during execution.
