# D68R Concrete Counter-Action Routing Repair Result

Status: completed local smoke run.

Artifact root:

```text
target/pilot_wave/d68r_concrete_counter_action_routing_repair/smoke/
```

Decision:

```text
decision = concrete_counter_action_routing_repair_positive_high_cost
verdict  = D68R_CONCRETE_COUNTER_ACTION_ROUTING_REPAIR_HIGH_COST
next     = D68C_SUPPORT_COST_OPTIMIZATION
```

## Summary

D68R repaired the concrete action-routing failure found by D68A. The D68 trained
threshold replay chose `REQUEST_COUNTER_TOP1_TOP2` on rows where
`REQUEST_JOINT_COUNTER` was required. D68R routed those rows back to the stronger
joint-counter path and recovered D67-level exact accuracy.

The run is high-cost rather than fully positive because support use did not drop
below D67. D68R fixed the harmful cheap path but did not yet find a cheaper
policy that preserves the same accuracy.

## Core Metrics

```text
D67_BEST_REPLAY:
  exact_joint_accuracy = 0.999333
  average_total_support_used = 7.6795
  wrong_concrete_counter_rate = 0.000333
  weak_top1_top2_path_failure_rate = 0.000000
  causal_unnecessary_counter_support_rate = 0.058833

D68_TRAINED_THRESHOLD_REPLAY:
  exact_joint_accuracy = 0.993833
  average_total_support_used = 6.4795
  wrong_concrete_counter_rate = 0.005833
  weak_top1_top2_path_failure_rate = 0.005833
  causal_unnecessary_counter_support_rate = 0.058833

D68R_CONCRETE_ROUTER:
  exact_joint_accuracy = 0.999333
  average_total_support_used = 7.6795
  wrong_concrete_counter_rate = 0.000333
  weak_top1_top2_path_failure_rate = 0.000000
  causal_unnecessary_counter_support_rate = 0.058833

CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY:
  exact_joint_accuracy = 0.999667
  average_total_support_used = 6.3195
  wrong_concrete_counter_rate = 0.000000
  weak_top1_top2_path_failure_rate = 0.000000
  causal_unnecessary_counter_support_rate = 0.000000
```

## Harm Repair

```text
d68_loss_rows_vs_d67 = 52
d68_loss_rows_repaired_by_d68r = 52
d68_loss_repair_rate = 1.0
```

Action distribution changed from D68's cheap but unsafe route:

```text
D68_TRAINED_THRESHOLD_REPLAY:
  REQUEST_COUNTER_TOP1_TOP2 = 4159
  REQUEST_JOINT_COUNTER = 0

D68R_CONCRETE_ROUTER:
  REQUEST_COUNTER_TOP1_TOP2 = 559
  REQUEST_JOINT_COUNTER = 3600
```

This confirms the D68A diagnosis: the failure was not simply "counter-support is
bad" or "too much support is bad"; the concrete counter action had to distinguish
the weak top1-vs-top2 path from the required joint-counter path.

## Controls

```text
TOP1_ONLY_CONTROL:
  exact_joint_accuracy = 0.993833
  average_total_support_used = 6.4795
  weak_top1_top2_path_failure_rate = 0.005833

JOINT_ONLY_CONTROL:
  exact_joint_accuracy = 0.999333
  average_total_support_used = 7.9590

RANDOM_COUNTER_CONTROL:
  exact_joint_accuracy = 0.783667
  average_total_support_used = 6.8530

SHUFFLED_ROUTER_CONTROL:
  exact_joint_accuracy = 0.993833
  average_total_support_used = 6.7590
```

## Runtime Integrity

```text
rust_path_invoked = true
rust_aggregation_rows = 345600
rust_controller_rows = 230400
fallback_rows = 0
failed_jobs = []
```

The run wrote `queue.json`, `progress.jsonl`, and partial reports throughout the
long Rust aggregation stages.

## Boundary

D68R only tests concrete counter-action routing repair for controlled symbolic
joint formula discovery. It does not prove full VRAXION brain, raw visual Raven
reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture
superiority, or production readiness.
