# E117 Alpha-Weave Targeted Pressure Gauntlet Contract

## Purpose

E116 generated synthetic alpha-Weave pressure cells for Operators that were
still sparse after the FineWeb projection. E117 runs those generated cells as a
deterministic targeted pressure gauntlet.

Core question:

```text
Do the 77 rare Operators reach the next 300k activation limit under targeted
alpha-Weave pressure without hard negatives, public leakage, unsupported
answers, wrong-scope calls, or false commits?
```

## Boundary

This is a targeted activation/no-harm gauntlet over synthetic pressure data.
It is not final training, not PermaCore, not TrueGolden, and not automatic Core
promotion.

## Source

```text
target/pilot_wave/e116_alpha_weave_synthetic_pressure_generation/generated_cells.jsonl
```

Every cell must keep synthetic origin in metadata and must not leak target
operator or oracle metadata into public input.

## Evaluation

For every generated cell and adversarial variant:

```text
schema valid
metadata valid
public input leak-free
expected action valid
ANSWER has answer + decisive evidence + citation
non-answer carries no answer
NO_CALL stays negative-scope
route budget is not exceeded by the mechanical target
```

The repeat count is deterministic. E117 validates each unique variant and then
accounts for the scheduled repeat count as qualified activation only if the
variant has no hard-negative failure.

## Hard Negative

Any of these is a hard negative:

```text
public oracle/target leakage
schema failure
missing synthetic metadata
unsupported answer
wrong-scope call
route over-budget
hidden/public inconsistency
```

One hard negative blocks the gauntlet.

## Required Artifacts

```text
run_manifest.json
gauntlet_manifest.json
operator_gauntlet_results.json
row_level_samples.json
hard_negative_samples.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
partial_aggregate_snapshot.json
progress.jsonl
report.md
checker_summary.json
```

## Decisions

```text
e117_targeted_pressure_gauntlet_next_limit_reached
e117_targeted_pressure_gauntlet_partial
e117_hard_negative_detected
```

## Pass

```text
checker failure_count = 0
target_reach_count = target_operator_count
targeted_needed_remaining_count = 0
hard_negative_total = 0
deterministic replay passes
no Core/PermaCore/TrueGolden promotion claim
```
