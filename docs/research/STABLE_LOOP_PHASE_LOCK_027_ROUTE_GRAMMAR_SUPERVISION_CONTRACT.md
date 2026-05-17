# STABLE_LOOP_PHASE_LOCK_027_ROUTE_GRAMMAR_SUPERVISION Contract

## Summary

026 showed that self-supervised delivery/local critics do not recover the ordered successor route objective:

```text
best non-hand critic:
  sufficient_tick_final_accuracy ~= 0.929
  long_path_accuracy ~= 0.871
  family_min_accuracy = 0.000
  route_order_accuracy ~= 0.489
  retained_successor_accuracy ~= 0.501
```

027 tests whether explicit staged route-grammar supervision can construct the order objective that 026 could not infer from delivery alone.

## Fixed Substrate

Keep:

```text
phase_i + gate_g -> phase_(i+g)
directed route transport
receive-commit target ledger
dense candidate route fields
025 ordered-route gate
026 self-supervised delivery baseline
```

No public `instnct-core` API changes.

## Required Arms

```text
HAND_ORDER_REGULARIZER_REFERENCE
SUCCESSOR_VALIDITY_ONLY
BRANCH_CYCLE_ONLY
ROUTE_CONTINUITY_ONLY
FAMILY_MIN_ONLY
LEARNED_SUCCESSOR_GRAMMAR
LEARNED_BRANCH_CYCLE_GRAMMAR
LEARNED_CONTINUITY_GRAMMAR
COMPOSED_ROUTE_GRAMMAR_CRITIC
COMPOSED_GRAMMAR_PLUS_DELIVERY
SELF_SUPERVISED_DELIVERY_GRAMMAR
RANDOM_GRAMMAR_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

The composed grammar arms model explicit supervised grammar labels. They may support route-grammar supervision, but not self-supervised or production route construction.

## Metrics

```text
route_order_accuracy
retained_successor_accuracy
branch_count
cycle_count
duplicate_successor_count
missing_successor_count
source_to_target_reachability
family_min_accuracy
sufficient_tick_final_accuracy
long_path_accuracy
wrong_if_delivered_rate
grammar_precision
grammar_recall
grammar_false_positive_rate
grammar_false_negative_rate
prune_success_rate
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
```

## Verdicts

```text
ROUTE_GRAMMAR_SUPERVISION_POSITIVE
SUCCESSOR_VALIDITY_LEARNABLE
BRANCH_CYCLE_GRAMMAR_LEARNABLE
CONTINUITY_GRAMMAR_LEARNABLE
FAMILY_MIN_GRAMMAR_REQUIRED
COMPOSED_GRAMMAR_CRITIC_WORKS
DELIVERY_PLUS_GRAMMAR_WORKS
RANDOM_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
ORDER_OBJECTIVE_NO_LONGER_HAND_SPECIFIED
ORDER_OBJECTIVE_STILL_HAND_SPECIFIED
PRODUCTION_API_NOT_READY
```

## Decision Gate

Report `ROUTE_GRAMMAR_SUPERVISION_POSITIVE` only if a non-control composed grammar arm reaches:

```text
sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
route_order_accuracy >= 0.90
retained_successor_accuracy >= 0.90
branch_count <= 0.05
cycle_count <= 0.05
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
random controls fail
wall/private/nonlocal/direct leaks = 0
```

If only self-supervised delivery grammar has signal, the 026 blocker remains. If composed supervised grammar passes, the order objective is constructible under explicit route-grammar supervision but not yet autonomously discovered.

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
grammar_metrics.jsonl
regularizer_metrics.jsonl
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

027 can support route-grammar supervision in toy phase-lane tasks. It cannot claim self-supervised grammar discovery, production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.
