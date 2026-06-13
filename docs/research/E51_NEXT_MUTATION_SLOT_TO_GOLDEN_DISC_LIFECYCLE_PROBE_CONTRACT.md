# E51 Next Mutation Slot To Golden Disc Lifecycle Probe Contract

## Purpose

E51 tests the first pocket-generation lifecycle after E49/E50.

Core question:

```text
Can the system keep exactly one active Next Mutation candidate, test it in a
sandbox, refine it by mutation/rollback, prove unique S-rank value, and save it
as a Golden Disc without overpromoting shortcuts, duplicates, or unsafe pockets?
```

## Boundary

This is a controlled symbolic/numeric lifecycle probe. It does not test raw
language reasoning, deployed assistant behavior, AGI, consciousness, or
model-scale behavior.

## Systems

```text
no_candidate_baseline
parallel_candidate_spam_control
light_probe_only_control
refinement_without_uniqueness_control
next_mutation_slot_to_golden_disc
oracle_lifecycle_reference
```

## Lifecycle

```text
NEXT_MUTATION
LIGHT_PROBE_PASS
ACTIVE_REFINEMENT
STABLE
S_RANK
GOLDEN_DISC
DISCARD
```

## Primary Rules

The primary system must:

```text
keep exactly one active next-mutation lane
run candidates in sandbox only
allow candidate writes only to Proposal Field
pass light probe before active refinement
use mutation/rollback during refinement
prune/crystallize before S-rank
require unique value by counterfactual ablation
require challenger sweep not to find a better mutation
require trace/replay/wrong-commit gates
save only S-rank as frozen Golden Disc
```

## Metrics

```text
exact_stage_accuracy
single_slot_integrity
slot_violation_rate
light_probe_precision
active_refinement_quality
s_rank_precision
golden_disc_count
golden_disc_quality
unique_value_score
challenger_defense_rate
prune_stability_rate
bad_promotion_rate
missed_golden_rate
wrong_commit_rate
direct_flow_write_violation_rate
cost_adjusted_value
```

## Decisions

Allowed decisions:

```text
e51_next_mutation_to_golden_disc_positive
e51_light_probe_insufficient
e51_parallel_candidate_spam_unsafe
e51_refinement_without_uniqueness_overpromotes
e51_no_unique_golden_value_detected
e51_invalid_artifact_detected
```

Positive requires:

```text
primary exact_stage_accuracy >= 0.99
single_slot_integrity = 1.0
golden_disc_count = 1
s_rank_precision = 1.0
golden_disc_quality >= 0.999
unique_value_score >= 0.05
challenger_defense_rate = 1.0
prune_stability_rate = 1.0
bad_promotion_rate = 0.0
missed_golden_rate = 0.0
wrong_commit_rate = 0.0
direct_flow_write_violation_rate = 0.0
light_probe_only_control overpromotes
refinement_without_uniqueness_control overpromotes
deterministic replay passes
target checker failure_count = 0
sample-only checker passes
```

## Required Artifacts

```text
backend_manifest.json
candidate_pool.json
lifecycle_rows.jsonl
light_probe_report.json
active_refinement_report.json
s_rank_report.json
golden_disc_registry.json
challenger_sweep_report.json
prune_crystallization_report.json
mutation_history.jsonl
system_results.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
results_table.md
report.md
```

## Sample Pack

The sample pack must live under:

```text
docs/research/artifact_samples/e51_next_mutation_slot_to_golden_disc_lifecycle_probe/
```

## Hard Requirements

```text
no gradient descent
no optimizer/backprop
row-level lifecycle events
real mutation/rollback for primary refinement
accepted/rejected mutation evidence
rollback count equals rejected count
Golden Disc has frozen digest and PocketToken metadata
target checker passes with failure_count = 0
sample-only checker passes
```
