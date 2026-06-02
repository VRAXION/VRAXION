# D68R Concrete Counter-Action Routing Repair Contract

## Goal

Repair the D68 counter-support triage failure identified by D68A.

D68A showed that the problem is not simply "too much counter-support". D67 has
some real causal extra support cost, but D68 lost accuracy because its trained
threshold gate sometimes selected the cheaper `REQUEST_COUNTER_TOP1_TOP2` action
when the row required `REQUEST_JOINT_COUNTER`.

## Question

Can we reduce support cost below D67 while preserving D67-level exact accuracy by
routing the selected concrete counter action correctly?

```text
cheap path:
  REQUEST_COUNTER_TOP1_TOP2

required stronger path:
  REQUEST_JOINT_COUNTER
```

## Required Files

```text
scripts/probes/run_d68r_concrete_counter_action_routing_repair.py
scripts/probes/run_d68r_concrete_counter_action_routing_repair_check.py
docs/research/D68R_CONCRETE_COUNTER_ACTION_ROUTING_REPAIR_CONTRACT.md
docs/research/D68R_CONCRETE_COUNTER_ACTION_ROUTING_REPAIR_RESULT.md
```

Generated artifacts live only under:

```text
target/pilot_wave/d68r_concrete_counter_action_routing_repair/
```

## Upstream

Use D68A as the immediate upstream audit:

```text
decision = counter_support_metric_pipeline_not_confirmed
verdict  = D68A_COUNTER_SUPPORT_METRIC_PIPELINE_NOT_CONFIRMED
next     = D68R_COUNTER_METRIC_REPAIR
```

Key D68A finding:

```text
d68_loss_rows_vs_d67 = 52
classification       = selected_top1_top2_failed_but_joint_counter_would_fix
classification_rate  = 1.0
```

## Arms

```text
D67_BEST_REPLAY
D68_TRAINED_THRESHOLD_REPLAY
D68R_CONCRETE_ROUTER
D68R_CONCRETE_ROUTER_COST_WEIGHTED
D68R_CONSERVATIVE_JOINT_REPAIR
TOP1_ONLY_CONTROL
JOINT_ONLY_CONTROL
NEVER_COUNTER_CONTROL
RANDOM_COUNTER_CONTROL
SHUFFLED_ROUTER_CONTROL
CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY
CHEAPEST_CORRECT_ORACLE_REFERENCE_ONLY
```

Oracle arms are reference-only and must not be counted as fair learned arms.

## Metrics

```text
exact_joint_accuracy
effective_accuracy
average_total_support_used
selected_counter_support_used_mean
causal_unnecessary_counter_support_rate
concrete_selected_counter_missed_rate
wrong_concrete_counter_rate
weak_top1_top2_path_failure_rate
selected_concrete_counter_fixes_rate
support_over_cheapest_effective_mean
false_confidence_rate
abstain_rate
fallback_rows
failed_jobs
```

## Positive Gate

`D68R_CONCRETE_ROUTER` is positive if:

```text
exact_joint_accuracy >= 0.9990
wrong_concrete_counter_rate <= 0.001
weak_top1_top2_path_failure_rate <= 0.001
average_total_support_used < D67_BEST_REPLAY.average_total_support_used
controls worse or more expensive
fallback_rows = 0
failed_jobs = []
```

## Decisions

```text
concrete_counter_action_routing_repair_positive
concrete_counter_action_routing_repair_positive_high_cost
concrete_counter_action_routing_repair_partial
concrete_counter_action_routing_repair_not_confirmed
```

## Hard Gates

```text
no broad claims
no label echo as fair oracle
truth hidden from fair arms
oracle arms reference-only
no Python hash
no fake accuracies
no random.random hit sampling
selected concrete counter correctness measured
top1-vs-joint routing reported
failed jobs visible
no black-box run: queue/progress/partial reports written regularly
```

## Boundary

D68R only tests concrete counter-action routing repair for controlled symbolic
joint formula discovery. It does not prove full VRAXION brain, raw visual Raven
reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture
superiority, or production readiness.
