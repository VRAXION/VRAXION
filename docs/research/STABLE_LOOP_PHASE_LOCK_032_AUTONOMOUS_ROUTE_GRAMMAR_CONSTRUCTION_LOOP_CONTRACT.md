# STABLE_LOOP_PHASE_LOCK_032_AUTONOMOUS_ROUTE_GRAMMAR_CONSTRUCTION_LOOP Contract

## Summary

031 showed that missing-successor / order-completion labels can be acquired from graph diagnostics:

```text
frontier expansion traces
prune residual missing-link analysis
graph invariant successor/continuity checks
```

032 integrates the successful pieces into one loop:

```text
dense candidate grow
graph diagnostic label acquisition
route grammar repair/update
order-aware prune
receive-commit delivery eval
missing-successor repair
repeat
```

No public `instnct-core` API changes.

## Required Arms

```text
HAND_PIPELINE_REFERENCE
ONE_PASS_GROW_DIAGNOSE_PRUNE
ITERATIVE_GROW_DIAGNOSE_PRUNE_2
ITERATIVE_GROW_DIAGNOSE_PRUNE_4
ITERATIVE_GROW_DIAGNOSE_PRUNE_8
FRONTIER_LABEL_LOOP
PRUNE_RESIDUAL_LABEL_LOOP
GRAPH_INVARIANT_LABEL_LOOP
MIXED_LABEL_LOOP
NO_LABEL_LOOP_CONTROL
RANDOM_LABEL_LOOP_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Metrics

```text
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
wrong_if_delivered_rate
route_order_accuracy
retained_successor_accuracy
missing_successor_count
duplicate_successor_count
branch_count
cycle_count
source_to_target_reachability
loop_iterations_to_pass
label_precision
label_recall
repair_success_rate
prune_success_rate
edge_count_initial
edge_count_final
prune_fraction
candidate_delta_nonzero_fraction
positive_delta_fraction
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
```

## Verdicts

```text
AUTONOMOUS_ROUTE_GRAMMAR_LOOP_POSITIVE
ONE_PASS_LOOP_WORKS
ITERATIVE_LOOP_REQUIRED
FRONTIER_LABEL_LOOP_WORKS
PRUNE_RESIDUAL_LABEL_LOOP_WORKS
GRAPH_INVARIANT_LABEL_LOOP_WORKS
MIXED_LABEL_LOOP_WORKS
NO_LABEL_LOOP_FAILS
RANDOM_LABEL_LOOP_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

```text
AUTONOMOUS_ROUTE_GRAMMAR_LOOP_POSITIVE if any non-hand loop reaches:

sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
route_order_accuracy >= 0.90
retained_successor_accuracy >= 0.90
missing_successor_count <= 0.05
branch/cycle near zero
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
random controls fail
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
loop_metrics.jsonl
repair_metrics.jsonl
prune_metrics.jsonl
label_loop_metrics.jsonl
grammar_metrics.jsonl
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

## Claim Boundary

032 can support an integrated autonomous route-grammar construction loop in the toy phase-lane substrate. It does not prove production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.
