# STABLE_LOOP_PHASE_LOCK_035_NON_TOY_ROUTE_GRAMMAR_TASK_SUITE Result

Status: complete.

## Verdicts

```text
NON_TOY_ROUTE_GRAMMAR_SUITE_POSITIVE
NOISY_GRAPH_ROUTE_TASK_WORKS
MULTI_REQUEST_BATCH_WORKS
DYNAMIC_EDGE_DELETION_REPAIR_WORKS
DISTRACTOR_SAME_SOURCE_TARGET_WORKS
COMPOSITIONAL_ROUTE_LABELS_WORKS
VARIABLE_PHASE_GATE_POLICY_WORKS
REACHABLE_SEED_BUG_REGRESSION_WORKS
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

The experimental `instnct-core` route-grammar API holds under the 035
non-toy hardening suite.

Passing hardening arms:

| Arm | Suff final | Long path | Family min | Wrong-if-delivered | Retained successor | Route order | Missing successor |
|---|---:|---:|---:|---:|---:|---:|---:|
| NOISY_GRAPH_ROUTE_TASK | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| MULTI_REQUEST_BATCH | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| DYNAMIC_EDGE_DELETION_REPAIR | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| DISTRACTOR_SAME_SOURCE_TARGET | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| COMPOSITIONAL_ROUTE_LABELS | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| VARIABLE_PHASE_GATE_POLICY | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| REACHABLE_SEED_BUG_REGRESSION | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |

Controls fail as required:

| Arm | Suff final | Long path | Family min | Wrong-if-delivered | Retained successor | Route order | Missing successor |
|---|---:|---:|---:|---:|---:|---:|---:|
| NO_GRAMMAR_API_CONTROL | 0.985 | 0.968 | 0.000 | 0.015 | 0.651 | 0.643 | 9.8 |
| RANDOM_ROUTE_TASK_CONTROL | 0.641 | 0.356 | 0.000 | 0.389 | 0.480 | 0.409 | 12.1 |
| RANDOM_PHASE_RULE_CONTROL | 0.495 | 0.490 | 0.000 | 0.383 | 1.000 | 1.000 | 0.0 |

Bounded API error handling passed:

```text
source_out_of_bounds_rejected = true
edge_out_of_bounds_rejected = true
bounded_api_error_handling_pass = true
```

## Interpretation

035 supports that the route-grammar API is no longer only a toy single-path
runner integration. It survives harder suite members that exercise noisy
candidate graphs, multiple request batches, repair after edge deletion,
distractors, compositional labels, variable phase/gate policies, and the
reachable-seed bug regression.

The old failure still appears when the grammar API/labels are removed:
aggregate delivery remains high, but family-min and route-order collapse.

## Current Boundary

035 is still a research hardening suite. It does not prove production API
readiness, full VRAXION, language grounding, consciousness, Prismion uniqueness,
biological equivalence, FlyWire wiring, or physical quantum behavior.

Next blocker:

```text
production API stabilization / external non-runner consumers
```
