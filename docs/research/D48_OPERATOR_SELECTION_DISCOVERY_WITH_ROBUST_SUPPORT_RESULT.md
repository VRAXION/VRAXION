# D48 Operator Selection Discovery With Robust Support Result

Status:

```text
completed
```

Artifact root:

```text
target/pilot_wave/d48_operator_selection_discovery_with_robust_support/smoke/
```

Expected route:

```text
decision = operator_selection_discovery_with_robust_support_positive
verdict = D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT_POSITIVE
next = D49_JOINT_CELL_OPERATOR_DISCOVERY_WITH_ROBUST_SUPPORT
```

Observed route:

```text
decision = operator_selection_discovery_with_robust_support_positive
verdict = D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT_POSITIVE
next = D49_JOINT_CELL_OPERATOR_DISCOVERY_WITH_ROBUST_SUPPORT
```

Primary test metrics:

```text
ALL_OPERATOR_ENUMERATION_SOFT_BASELINE:
  clean = 1.0000
  correlated = 0.9583
  adversarial = 0.2333
  mixed_clean_correlated = 0.9998
  mixed_clean_adversarial = 0.9958

COUNTERFACTUAL_TOP1_TOP2_REPAIR:
  clean = 1.0000
  correlated = 1.0000
  adversarial = 0.9880
  mixed_clean_correlated = 1.0000
  mixed_clean_adversarial = 1.0000

FULL_ROBUST_ECF_CONTROLLER:
  clean = 1.0000
  correlated = 1.0000
  adversarial = 0.9880
  mixed_clean_correlated = 1.0000
  mixed_clean_adversarial = 1.0000
  exact_operator_accuracy = 0.9976
  operator_equivalence_accuracy = 0.9886

RANDOM_EXTRA_SUPPORT_CONTROL:
  correlated = 0.1547
  adversarial = 0.0000

SHUFFLED_COUNTER_SUPPORT_CONTROL:
  correlated = 0.4235
  adversarial = 0.2357

NO_COUNTERFACTUAL_CONTROL:
  correlated = 0.9583
  adversarial = 0.2333
```

Gains:

```text
robust_gain_vs_baseline = 0.1602
robust_gain_vs_random_extra = 0.49575
robust_gain_vs_no_counter = 0.1602
controls_worse = true
failed_jobs = []
```

Boundary:

```text
D48 only tests controlled symbolic operator-selection discovery with robust ECF support.
It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or that truth is recoverable from indistinguishable evidence.
```

The machine-readable run result is written under the artifact root in `decision.json`, `aggregate_metrics.json`, `policy_comparison_report.json`, `operator_diagnostic_report.json`, and `support_cost_frontier_report.json`.
