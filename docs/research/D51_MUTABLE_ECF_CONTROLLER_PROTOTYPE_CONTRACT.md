# D51 Mutable ECF Controller Prototype Contract

## Goal

D51 tests whether a mutable controller can learn the ECF control policy for the already established controlled symbolic joint formula discovery task.

The formula task remains fixed:

```text
formula = operator(cell_a, cell_b) mod 9
```

D51 does not learn the formula solver. It learns only the controller decision over bounded actions:

```text
DECIDE
REQUEST_SUPPORT
REQUEST_COUNTER_TOP1_TOP2
REQUEST_JOINT_COUNTER
REQUEST_EXTERNAL_TEST
ABSTAIN
```

## Upstream

```text
D50 decision = joint_formula_discovery_scale_confirmed
D50 verdict = D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRMED
D50 next = D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE
```

## Inputs

The mutable controller receives diagnostics derived from score vectors and support-channel state:

```text
scalar confidence
top1/top2 margin
entropy
collision count
support cluster count
dominant cluster fraction
top1/factorised disagreement
cell confidence
operator confidence
joint confidence
internal unresolvable support-channel indicator
external test-channel availability
```

The controller must not receive the true joint candidate, true cell pair, true operator, false joint candidate, or expected answer label as inference input.

## Arms

```text
HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE
HANDCODED_CAP_7_REFERENCE
HANDCODED_CAP_9_REFERENCE
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
ALWAYS_COUNTER_SUPPORT_CONTROL
MUTABLE_LINEAR_CONTROLLER
MUTABLE_RULE_TABLE_CONTROLLER
MUTABLE_SMALL_TREE_CONTROLLER
MUTABLE_HYBRID_CONTROLLER
```

## Required Reports

```text
d50_upstream_manifest.json
controller_input_feature_report.json
mutation_acceptance_report.json
policy_action_distribution_report.json
support_cost_frontier_report.json
false_confidence_report.json
regime_breakdown_report.json
controller_comparison_report.json
best_policy_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

## Positive Route

```text
decision = mutable_ecf_controller_prototype_positive
verdict = D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE_POSITIVE
next = D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM
```

The positive gate requires the best mutable controller to meet:

```text
exact_joint_accuracy >= 0.995
correlated_echo_accuracy >= 0.995
adversarial_distractor_accuracy >= 0.995
indistinguishable_false_confidence_rate <= 0.01
support_used <= D50 FULL support
accuracy-cost >= D50 CAP_9 reference
failed_jobs = []
```

## Boundary

D51 only tests mutable control policy for controlled symbolic joint formula discovery. It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
