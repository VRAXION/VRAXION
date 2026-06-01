# D47 Cell Reference Discovery With Robust Support Result

Status:

```text
completed
```

Branch:

```text
codex/d47-cell-reference-discovery-with-robust-support
```

Artifact root:

```text
target/pilot_wave/d47_cell_reference_discovery_with_robust_support/smoke/
```

Decision:

```text
decision = cell_reference_discovery_with_robust_support_positive
verdict = D47_CELL_REFERENCE_DISCOVERY_WITH_ROBUST_SUPPORT_POSITIVE
next = D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT
rows_evaluated = 3172000
```

Primary `ALL28_UNORDERED`, support budget 5:

```text
ALL28_PAIR_ENUMERATION_SOFT_BASELINE:
  clean = 0.9996
  correlated = 0.2412
  adversarial = 0.8890
  mixed = 0.9926
  unordered = 0.8230
  cell_hit_top2 = 0.8482
  support = 5.000

CELL_REFERENCE_FACTORISED_FIELD:
  clean = 0.9998
  correlated = 0.2184
  adversarial = 0.8439
  mixed = 0.9906
  unordered = 0.8086
  cell_hit_top2 = 0.8416
  support = 5.000

COUNTERFACTUAL_TOP1_TOP2_REPAIR:
  clean = 1.0000
  correlated = 0.9230
  adversarial = 0.9932
  mixed = 0.9995
  unordered = 0.9830
  cell_hit_top2 = 0.9907
  support = 5.610

FULL_ROBUST_ECF_CONTROLLER:
  clean = 1.0000
  correlated = 0.9839
  adversarial = 0.9991
  mixed = 0.9999
  unordered = 0.9966
  cell_hit_top2 = 0.9980
  support = 6.220

RANDOM_EXTRA_SUPPORT_CONTROL:
  correlated = 0.8256
  adversarial = 0.9860
  support = 5.610

SHUFFLED_COUNTER_SUPPORT_CONTROL:
  correlated = 0.2190
  adversarial = 0.4818
  support = 6.220

SHUFFLED_CELL_REFERENCE_CONTROL:
  correlated = 0.0361
  adversarial = 0.0387
```

Support-cost frontier for `FULL_ROBUST_ECF_CONTROLLER`:

```text
budget1: corr = 0.8696, adv = 0.8721, support = 2.000
budget2: corr = 0.8662, adv = 0.8670, support = 2.989
budget3: corr = 0.9824, adv = 0.9835, support = 4.184
budget4: corr = 0.9864, adv = 0.9835, support = 4.937
budget5: corr = 0.9839, adv = 0.9991, support = 6.220
```

Interpretation:

```text
Robust ECF support transfers from primitive/family choice to cell-reference discovery.
Counterfactual top1-vs-top2 remains the main single repair path.
Full robust controller is strongest and passes equivalence/cell-hit gates.
Factorised cell-reference did not beat raw pair enumeration in this run.
ORDERED56 exact candidate is weaker than unordered/equivalence, as expected under aliasing.
```

Boundary: D47 only tests controlled symbolic cell-reference discovery with robust ECF support. It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
