# STABLE_LOOP_PHASE_LOCK_034_INSTNCT_ROUTE_GRAMMAR_API_INTEGRATION Result

Status: complete.

## Verdicts

```text
INSTNCT_ROUTE_GRAMMAR_API_INTEGRATION_POSITIVE
INSTNCT_EXPERIMENTAL_ROUTE_GRAMMAR_API_WORKS
GENERALIZED_SINGLE_PATH_WORKS
VARIABLE_WIDTH_PATH_WORKS
LONG_ROUTE_STRESS_WORKS
MULTI_TARGET_ROUTE_SET_WORKS
BRANCHING_ROUTE_TREE_WORKS
VARIABLE_GATE_RULE_FAMILY_WORKS
NO_GRAMMAR_API_CONTROL_FAILS
RANDOM_ROUTE_TASK_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Smoke

```text
seeds = 2026,2027,2028
eval_examples = 1024
widths = 8,12,16
path_lengths = 4,8,16,24,32
ticks = 8,16,24,32,48
completed rows = 18900
```

## Main Result

034 moves the 033 runner-local route-grammar subsystem into an
`instnct-core` owned experimental module:

```text
instnct_core::experimental_route_grammar
```

Passing API-backed/generalized arms:

| Arm | Suff final | Long path | Family min | Wrong-if-delivered | Retained successor | Route order | Missing successor |
|---|---:|---:|---:|---:|---:|---:|---:|
| GENERALIZED_SINGLE_PATH | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| VARIABLE_WIDTH_PATH | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| LONG_ROUTE_STRESS | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| MULTI_TARGET_ROUTE_SET | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| BRANCHING_ROUTE_TREE | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| VARIABLE_GATE_RULE_FAMILY | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| INSTNCT_EXPERIMENTAL_ROUTE_GRAMMAR_API | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |

Controls fail as required:

| Arm | Suff final | Long path | Family min | Wrong-if-delivered | Retained successor | Route order | Missing successor |
|---|---:|---:|---:|---:|---:|---:|---:|
| NO_GRAMMAR_API_CONTROL | 0.985 | 0.968 | 0.000 | 0.015 | 0.651 | 0.643 | 9.8 |
| RANDOM_ROUTE_TASK_CONTROL | 0.641 | 0.356 | 0.000 | 0.389 | 0.480 | 0.409 | 12.1 |
| RANDOM_PHASE_RULE_CONTROL | 0.495 | 0.490 | 0.000 | 0.383 | 1.000 | 1.000 | 0.0 |

## Implementation Note

The first quick run exposed an integration bug: the API accepted any reachable
seed path before applying graph-diagnostic successor labels. That reproduced
the old aggregate-good / route-order-bad failure. The API was tightened to
prefer diagnostic successor labels before reachability-only seed acceptance.

## Interpretation

The route-grammar mechanism now has an `instnct-core` experimental API surface
and the 033 toy generalization still passes through it:

```text
dense candidate edges
weak seed successors
graph-diagnostic successor labels
  -> construct_route_grammar
  -> ordered successor route
  -> receive-commit delivery scoring
```

The causal control remains intact: without the grammar API/labels, aggregate
delivery can look strong, but family-min and route-order collapse return.

## Current Boundary

034 supports experimental `instnct-core` API integration for the toy
route-grammar subsystem. It does not prove production API readiness, non-toy
production tasks, full VRAXION, language grounding, consciousness, Prismion
uniqueness, biological equivalence, FlyWire wiring, or physical quantum
behavior.

Next blocker:

```text
non-toy task suite / production readiness hardening
```
