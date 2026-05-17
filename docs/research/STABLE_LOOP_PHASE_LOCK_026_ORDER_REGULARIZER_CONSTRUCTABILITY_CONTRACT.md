# STABLE_LOOP_PHASE_LOCK_026_ORDER_REGULARIZER_CONSTRUCTABILITY Contract

## Summary

025 showed that dense candidate route fields can be pruned into a clean ordered successor route when the prune objective is handed route-order regularizers:

```text
delivery + order regularizers:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  route_order_accuracy = 1.000
```

026 asks whether those order constraints can be approximated by constructible critics rather than treated only as a hand-specified objective.

## Fixed Substrate

Keep:

```text
phase_i + gate_g -> phase_(i+g)
directed route transport
receive-commit target ledger
dense candidate route fields
025 positive gate
```

No public `instnct-core` API changes.

## Required Arms

```text
HAND_ORDER_REGULARIZER_REFERENCE
SUCCESSOR_CONSISTENCY_ONLY
BRANCH_PENALTY_ONLY
CYCLE_PENALTY_ONLY
ROUTE_CONTINUITY_ONLY
FAMILY_MIN_ADVERSARIAL_ONLY
PAIRWISE_ORDER_CRITIC
LOCAL_BRANCH_CYCLE_CRITIC
SOURCE_TARGET_REACHABILITY_CRITIC
LEARNED_ORDER_CRITIC_FROM_DENSE_EXAMPLES
SELF_SUPERVISED_ORDER_CRITIC_FROM_DELIVERY
RANDOM_CRITIC_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

`HAND_ORDER_REGULARIZER_REFERENCE` is diagnostic only. It may reproduce 025 but cannot support constructability by itself.

## Metrics

```text
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
wrong_if_delivered_rate
route_order_accuracy
retained_successor_accuracy
branch_count
cycle_count
duplicate_successor_count
missing_successor_count
route_continuity_score
source_to_target_reachability
critic_precision
critic_recall
critic_false_positive_rate
critic_false_negative_rate
prune_success_rate
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
```

## Verdicts

```text
ORDER_REGULARIZER_CONSTRUCTABILITY_POSITIVE
SUCCESSOR_CONSISTENCY_SUFFICIENT
BRANCH_CYCLE_CRITIC_HAS_PARTIAL_SIGNAL
ROUTE_CONTINUITY_REQUIRED
FAMILY_MIN_ADVERSARIAL_REQUIRED
LEARNED_ORDER_CRITIC_HAS_SIGNAL
SELF_SUPERVISED_ORDER_CRITIC_INSUFFICIENT
HAND_ORDER_REGULARIZER_REFERENCE_REPRODUCED
RANDOM_CRITIC_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
ORDER_OBJECTIVE_STILL_HAND_SPECIFIED
PRODUCTION_API_NOT_READY
```

## Decision Gate

Report `ORDER_REGULARIZER_CONSTRUCTABILITY_POSITIVE` only if a non-control, non-hand critic reaches:

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

If only the hand reference passes, the order objective remains hand-specified.

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
critic_metrics.jsonl
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

026 can support order-critic constructability signals in toy phase-lane tasks only. It cannot claim production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.
