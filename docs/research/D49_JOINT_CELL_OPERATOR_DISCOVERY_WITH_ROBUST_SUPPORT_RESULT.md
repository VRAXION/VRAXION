# D49 Joint Cell Operator Discovery With Robust Support Result

Status:

```text
completed
```

Artifact root:

```text
target/pilot_wave/d49_joint_cell_operator_discovery_with_robust_support/smoke/
```

Expected route:

```text
decision = joint_cell_operator_discovery_with_robust_support_positive
verdict = D49_JOINT_CELL_OPERATOR_DISCOVERY_WITH_ROBUST_SUPPORT_POSITIVE
next = D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM
```

Observed route:

```text
decision = joint_binding_bottleneck
verdict = D49_JOINT_BINDING_BOTTLENECK
next = D49B_JOINT_BINDING_REPAIR
```

Primary `ALL28_UNORDERED x operators`, core regimes only:

```text
JOINT_ENUMERATION_SOFT_BASELINE:
  accuracy = 0.5637
  cell_pair_equivalence_accuracy = 0.5818
  operator_exact_accuracy = 0.6263
  support = 5.000

COUNTERFACTUAL_TOP1_TOP2_REPAIR:
  accuracy = 0.9581
  cell_pair_equivalence_accuracy = 0.9748
  operator_exact_accuracy = 0.9633
  support = 6.250

FULL_ROBUST_ECF_CONTROLLER:
  accuracy = 0.9581
  exact_joint_accuracy = 0.9581
  cell_pair_equivalence_accuracy = 0.9748
  cell_hit_top2_accuracy = 0.9748
  operator_exact_accuracy = 0.9633
  operator_equivalence_accuracy = 0.9633
  support = 6.250

RANDOM_EXTRA_SUPPORT_CONTROL:
  accuracy = 0.3119

SHUFFLED_COUNTER_SUPPORT_CONTROL:
  accuracy = 0.4459

NO_COUNTERFACTUAL_CONTROL:
  accuracy = 0.5637
```

Regime breakdown for `FULL_ROBUST_ECF_CONTROLLER`:

```text
clean = 1.0000
correlated_echo = 0.8952
adversarial_distractor = 0.8950
mixed_clean_correlated = 1.0000
mixed_clean_adversarial = 1.0000
```

D47B-style abstain behavior:

```text
ABSTAIN_ON_INDISTINGUISHABLE_CASES:
  indistinguishable_false_abstain_rate = 1.0000
  indistinguishable_false_false_confidence_rate = 0.0000
  external_test_required_accuracy = 0.3890
```

Interpretation:

```text
D49 does not pass as a full positive.
Counterfactual top1-vs-top2 repair is again the main useful repair and gives a large gain over baseline.
However, correlated/adversarial joint formula cases remain below the 0.95 gate.
Cell and operator signals are mostly recoverable, but exact joint binding is still the bottleneck.
The next route is D49B_JOINT_BINDING_REPAIR.
```

Boundary:

```text
D49 only tests controlled symbolic joint cell+operator discovery with robust ECF support.
It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
```

The machine-readable run result is written under the artifact root in `decision.json`, `aggregate_metrics.json`, `policy_comparison_report.json`, `exact_equivalence_audit_report.json`, and `indistinguishable_abstain_report.json`.
