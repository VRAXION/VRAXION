# E8H4 Region Operator Composition Scale Probe Contract

## Purpose

`E8H4_REGION_OPERATOR_COMPOSITION_SCALE_PROBE` tests whether the E8H3 local
proxy abstraction scales compositionally:

```text
Pocket = feature detector + direct Flow-grid region transform operator
```

There is no internal pocket language, producer/consumer RAM-code translation,
proposal memory dependency, semantic lane, language task, image task, or dense
graph success criterion.

## Model

A homogeneous region-transform pocket has:

```text
read_region
detector / condition
transform_op
write_region
optional trace_check
cost / footprint
```

Allowed transform primitives:

```text
copy
move
delete / clear
invert
shift
fill gap
bind marker
threshold
delete isolated features
```

Primary Flow is a binary grid.

## Systems

```text
identity_noop_baseline
direct_overwrite_matrix_baseline
handcoded_oracle_region_operator_reference
random_region_rule_control
mutation_discovered_single_operator
mutation_discovered_composed_3_step
mutation_discovered_composed_6_step
mutation_discovered_composed_12_step
mutation_discovered_composed_24_step
mutation_discovered_plus_trace_check
mutation_discovered_plus_prune
reusable_operator_library_router
dense_transform_danger_control
answer_shortcut_control
```

Oracle/reference and shortcut controls are never valid primary learned systems.

## Task Families

Rows are controlled binary Flow-grid tasks with known oracle traces:

```text
local cleanup
motion
completion
binding
routing/composition
conflict
delayed dependency
OOD shifted/larger/noisier grids
counterfactual route order
adversarial distractors
```

Scale axes:

```text
grid size: 6x6, 8x8, OOD larger grid
route length: 1, 3, 6, 12, 24
operator count: discovered library over fixed primitive skill set
noise/distractor density: split dependent
```

## Mutation Contract

Mutation search mutates:

```text
read region
transform type
write region
shift dx/dy
threshold
trace-check objective
footprint/cost pressure
```

The runner must report accepted/rejected/rollback counts through
`mutation_history.jsonl`.

## Required Metrics

```text
usefulness
answer accuracy
trace validity
frame MAE to oracle trace
delta MAE to oracle transition
drift per step
drift slope by route length
first divergence step
operator reuse rate
operator footprint / read-write cell count
mutation acceptance rate
discovered operator count
route length survival curve
OOD usefulness
counterfactual usefulness
adversarial usefulness
dense shortcut trace gap
random control gap
deterministic replay hash match
checker failure_count
```

## Positive Gate

A learned region-operator system is positive only if:

```text
usefulness > identity_noop_baseline + 0.03
trace_validity >= identity_noop_baseline
route length 6 and 12 stay above baseline
route length 24 does not collapse for full positive
OOD/counterfactual/adversarial are not collapsed
dense/answer shortcut controls do not win with valid trace
deterministic_replay_passed = true
checker_failure_count = 0
```

## Decision Labels

```text
e8h4_region_operator_composition_scale_positive
e8h4_region_operator_partial_scale
e8h4_single_operator_only_no_composition
e8h4_trace_drift_accumulation_failure
e8h4_operator_reuse_positive
e8h4_mutation_search_insufficient
e8h4_dense_shortcut_trace_invalid
e8h4_region_operator_not_sufficient
```

## Required Artifacts

```text
aggregate_metrics.json
split_metrics.json
depth_scaling_report.json
operator_discovery_report.json
operator_reuse_report.json
mutation_history.jsonl
row_level_samples.jsonl
dense_shortcut_control_report.json
deterministic_replay.json
decision.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

## Boundary

E8H4 is a controlled Flow-grid region-operator proxy only. It does not prove raw
language reasoning, AGI, consciousness, deployed-model behavior, or model-scale
behavior.
