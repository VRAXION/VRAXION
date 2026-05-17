# STABLE_LOOP_PHASE_LOCK_027_ROUTE_GRAMMAR_SUPERVISION Result

Status: complete.

## Verdicts

```text
ROUTE_GRAMMAR_SUPERVISION_POSITIVE
COMPOSED_GRAMMAR_CRITIC_WORKS
DELIVERY_PLUS_GRAMMAR_WORKS
SUCCESSOR_VALIDITY_LEARNABLE
BRANCH_CYCLE_GRAMMAR_LEARNABLE
CONTINUITY_GRAMMAR_LEARNABLE
FAMILY_MIN_GRAMMAR_REQUIRED
SELF_SUPERVISED_GRAMMAR_INSUFFICIENT
RANDOM_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
ORDER_OBJECTIVE_NO_LONGER_HAND_SPECIFIED
PRODUCTION_API_NOT_READY
```

## Smoke Summary

```text
seeds: 2026,2027,2028
widths: 8,12,16
path_lengths: 4,8,16,24,32
ticks: 8,16,24,32,48
rows: 20475
```

## Positive Branch

The composed supervised grammar arms reproduce the 025 ordered-route gate:

```text
COMPOSED_GRAMMAR_PLUS_DELIVERY:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  retained_successor_accuracy = 1.000
  route_order_accuracy = 1.000
  branch_count = 0.0
  cycle_count = 0.0
  source_to_target_reachability = 1.000
  gate_shuffle_collapse = 0.719
  same_target_counterfactual_accuracy = 1.000
```

`COMPOSED_ROUTE_GRAMMAR_CRITIC` matches the same gate. This means explicit staged grammar labels can supply the route objective that 026 failed to infer from delivery alone.

## Baseline Failure Reproduced

Self-supervised delivery grammar remains insufficient:

```text
SELF_SUPERVISED_DELIVERY_GRAMMAR:
  sufficient_tick_final_accuracy = 0.931
  long_path_accuracy = 0.898
  family_min_accuracy = 0.000
  wrong_if_delivered_rate = 0.046
  retained_successor_accuracy = 0.483
  route_order_accuracy = 0.474
```

Single grammar components also do not solve the whole objective. They provide partial signals, but without composition the route order remains around half-correct and family-min collapses.

## Controls

Random controls fail:

```text
RANDOM_GRAMMAR_CONTROL:
  sufficient_tick_final_accuracy = 0.641
  long_path_accuracy = 0.356
  wrong_if_delivered_rate = 0.389

RANDOM_PHASE_RULE_CONTROL:
  sufficient_tick_final_accuracy = 0.495
  long_path_accuracy = 0.490
  wrong_if_delivered_rate = 0.383
```

## Interpretation

027 resolves the 026 blocker under explicit supervision:

```text
delivery-only/self-supervised critics do not learn route grammar,
but composed route-grammar supervision recovers the ordered successor objective.
```

The order objective is no longer only a hand-written prune regularizer in this toy runner. It is constructible when supplied as staged grammar supervision:

```text
successor validity
branch/cycle rejection
source-target continuity
family-min adversarial pressure
```

The remaining blocker is not the grammar itself. It is how to acquire or distill those grammar labels without direct supervision.

## Claim Boundary

027 supports explicit route-grammar supervision in toy phase-lane tasks. It does not prove self-supervised grammar discovery, production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.
