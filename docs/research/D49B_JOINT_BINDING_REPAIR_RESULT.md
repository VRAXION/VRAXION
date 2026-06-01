# D49B Joint Binding Repair Result

Status:

```text
completed
```

Artifact root:

```text
target/pilot_wave/d49b_joint_binding_repair/smoke/
```

Expected route if the repair passes:

```text
decision = joint_binding_repair_positive
verdict = D49B_JOINT_BINDING_REPAIR_POSITIVE
next = D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM
```

D49B is a repair/diagnostic milestone after D49:

```text
D49 result:
  decision = joint_binding_bottleneck
  next = D49B_JOINT_BINDING_REPAIR

D49B question:
  can targeted cell/operator/joint counter-support repair pure correlated/adversarial joint binding?
```

Observed route:

```text
decision = joint_binding_repair_positive
verdict = D49B_JOINT_BINDING_REPAIR_POSITIVE
next = D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM
```

Primary `ALL28_UNORDERED x operators`, core regimes only:

```text
D49_BASELINE_REPLAY:
  exact_joint_accuracy = 0.5665
  cell_pair_equivalence_accuracy = 0.5821
  operator_exact_accuracy = 0.6312
  support = 5.000

JOINT_INTERACTION_COUNTERFACTUAL:
  exact_joint_accuracy = 0.9964
  cell_pair_equivalence_accuracy = 0.9995
  operator_exact_accuracy = 0.9966
  support = 6.880

MULTI_STAGE_COUNTERFACTUAL_REPAIR:
  exact_joint_accuracy = 0.9990
  cell_pair_equivalence_accuracy = 1.0000
  operator_exact_accuracy = 0.9990
  support = 8.134

FULL_REPAIRED_ECF_CONTROLLER:
  exact_joint_accuracy = 0.9994
  cell_pair_equivalence_accuracy = 1.0000
  cell_hit_top2_accuracy = 1.0000
  operator_exact_accuracy = 0.9994
  operator_equivalence_accuracy = 0.9994
  support = 8.761
```

Regime breakdown for `FULL_REPAIRED_ECF_CONTROLLER`:

```text
clean = 1.0000
correlated_echo = 0.9980
adversarial_distractor = 0.9992
mixed_clean_correlated = 1.0000
mixed_clean_adversarial = 1.0000
distinguishable_correlated_false = 0.9988
indistinguishable_false_abstain_rate = 1.0000
indistinguishable_false_false_confidence_rate = 0.0000
external_test_required_accuracy = 0.9970
```

Controls:

```text
RANDOM_EXTRA_SUPPORT_CONTROL:
  exact_joint_accuracy = 0.3132

SHUFFLED_COUNTER_SUPPORT_CONTROL:
  exact_joint_accuracy = 0.4457

NO_COUNTERFACTUAL_CONTROL:
  exact_joint_accuracy = 0.5665
```

Interpretation:

```text
D49B repaired the D49 pure correlated/adversarial joint binding gap.
The useful repair is not random extra support.
The useful repair is targeted joint interaction counter-support, improved further by multi-stage cell/operator/joint counterfactual support.
Indistinguishable cases abstain instead of producing false confidence.
External-test-required cases are solved only when the external/interventional support channel is used.
```

The machine-readable run writes `decision.json`, `aggregate_metrics.json`, `joint_error_taxonomy_report.json`, `counterfactual_stage_report.json`, `external_test_required_report.json`, and `support_cost_frontier_report.json` under the artifact root.

Boundary:

```text
D49B only tests controlled symbolic joint cell+operator binding repair with robust ECF support.
It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
```
