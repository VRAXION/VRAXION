# STABLE_LOOP_PHASE_LOCK_024_DENSE_ROUTE_GROW_PRUNE_POLICY Contract

## Summary

023 showed that a dense candidate route field can be crystallized/pruned into the ordered successor route-token representation.

024 tests whether that branch can be turned into a controlled route-growth policy:

```text
dense candidate edge field
  -> phase/delivery-guided crystallize
  -> prune
  -> ordered successor route
```

This is not production routing and not random-from-scratch route growth.

## Fixed Substrate

Keep the completed local phase rule:

```text
phase_i + gate_g -> phase_(i+g)
```

Keep directed route transport and receive-commit target ledger semantics.

No public `instnct-core` API changes.

## Required Arms

```text
TRUE_PATH_UPPER_BOUND
HAND_DENSE_THEN_PRUNE_REFERENCE
DENSE_FIELD_EDGE_BUDGET_1X
DENSE_FIELD_EDGE_BUDGET_2X
DENSE_FIELD_EDGE_BUDGET_4X
DENSE_FIELD_EDGE_BUDGET_8X
COST_PENALIZED_DENSE_GROW
EARLY_PRUNE_SCHEDULE
LATE_PRUNE_SCHEDULE
GROW_THEN_PRUNE_ONCE
GROW_PRUNE_ALTERNATING
SOURCE_TARGET_ANCHOR_PLUS_DENSE_GROW
RANDOM_DENSE_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

`TRUE_PATH_UPPER_BOUND` is diagnostic-only. `HAND_DENSE_THEN_PRUNE_REFERENCE` is a supplied-reference arm and cannot prove operational public route growth by itself.

## Metrics

```text
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
wrong_if_delivered_rate
initial_edge_count
final_edge_count
prune_fraction
retained_successor_accuracy
route_order_accuracy
branch_count
cycle_count
reciprocal_edge_fraction
backflow_edge_fraction
edge_cost_adjusted_score
grow_accept_rate
prune_accept_rate
positive_delta_fraction
seed_stability
random_control_accuracy
gate_shuffle_collapse
same_target_counterfactual_accuracy
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

## Verdicts

```text
DENSE_GROW_PRUNE_POLICY_POSITIVE
EDGE_BUDGET_2X_SUFFICIENT
EDGE_BUDGET_4X_REQUIRED
COST_PENALIZED_GROW_WORKS
EARLY_PRUNE_FAILS
LATE_PRUNE_WORKS
ALTERNATING_GROW_PRUNE_WORKS
SOURCE_TARGET_ANCHOR_IMPROVES_GROW
RANDOM_DENSE_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

Report `DENSE_GROW_PRUNE_POLICY_POSITIVE` only if a non-private dense-grow-prune policy reaches:

```text
sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
random controls fail
wall/private/nonlocal/direct leaks = 0
```

If only the hand-dense reference works, report that the strategy is not yet operationalized.

If high edge budget is required, report the budget requirement without claiming production readiness.

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
grow_prune_metrics.jsonl
budget_metrics.jsonl
delivery_metrics.jsonl
routing_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
control_metrics.jsonl
locality_audit.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

No black-box runs: append heartbeat progress and refresh `summary.json` / `report.md` during long runs.

## Claim Boundary

024 can support dense-route grow/prune policy behavior in toy phase-lane tasks only. It cannot claim production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.
