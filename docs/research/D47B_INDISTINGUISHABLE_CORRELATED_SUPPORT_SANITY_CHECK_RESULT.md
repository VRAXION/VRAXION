# D47B Indistinguishable Correlated Support Sanity Check Result

Status:

```text
completed
```

Run artifact root:

```text
target/pilot_wave/d47b_indistinguishable_correlated_support_sanity_check/smoke/
```

Expected positive route:

```text
decision = indistinguishability_boundary_confirmed
verdict = D47B_IDENTIFIABILITY_BOUNDARY_CONFIRMED
next = D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT
```

Observed result:

```text
decision = indistinguishability_boundary_confirmed
verdict = D47B_IDENTIFIABILITY_BOUNDARY_CONFIRMED
next = D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT
```

Primary test metrics:

```text
ROBUST_ECF_COUNTER_SUPPORT:
  distinguishable_correlated_false_accuracy = 1.0000
  indistinguishable_correlated_false_accuracy = 0.0000
  external_test_required_accuracy = 0.0000

ROBUST_ECF_WITH_ABSTAIN:
  independent_true_accuracy = 1.0000
  correlated_true_accuracy = 1.0000
  distinguishable_correlated_false_accuracy = 1.0000
  indistinguishable_correlated_false_abstain_rate = 1.0000
  indistinguishable_correlated_false_false_confidence_rate = 0.0000
  external_test_required_abstain_rate = 1.0000

ROBUST_ECF_WITH_EXTERNAL_TEST:
  external_test_required_accuracy = 1.0000
  external_test_used = 1.0000

NAIVE_IPF:
  distinguishable_correlated_false_accuracy = 0.0000
  indistinguishable_correlated_false_accuracy = 0.0000
  external_test_required_accuracy = 0.0000
```

Certificate summary:

```text
DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT:
  support_true_false_delta = 0.0000
  internal_counter_true_false_delta = 4.0000

INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT:
  support_true_false_delta = 0.0000
  internal_counter_true_false_delta = 0.0000

EXTERNAL_TEST_REQUIRED_SUPPORT:
  support_true_false_delta = 0.0000
  internal_counter_true_false_delta = 0.0000
  external_counter_true_false_delta = 4.0000
```

Boundary:

```text
D47B only tests identifiability limits of robust ECF under correlated support in controlled symbolic tasks.
It does not prove raw visual Raven, AGI, consciousness, architecture superiority, or that truth is recoverable from indistinguishable evidence.
```

The machine-readable run result is written to `decision.json`, `aggregate_metrics.json`, `identifiability_report.json`, and `indistinguishability_certificate_report.json` under the artifact root.
