# D45 Robust Support Policy Prototype Contract

D45 tests whether an IPF/ECF-style evidence field can survive correlated and adversarial support by asking for independent, diverse, counterfactual support instead of simply asking for more support.

## Scope

Controlled symbolic primitive discovery only:

```text
3x3 board
hidden true primitive family
2-cell addition mod 9
candidate/family/equivalence metrics separated
```

## Policy Arms

```text
NAIVE_IPF_BASELINE
STAGED_SUPPORT_BASELINE
DUPLICATE_SUPPORT_DOWNWEIGHTING
SOURCE_DIVERSITY_WEIGHTING
LEAVE_ONE_SUPPORT_OUT_STABILITY
ROBUST_MEDIAN_FIELD_AGGREGATION
COUNTER_SUPPORT_QUERY_POLICY
ADVERSARIAL_DISTRACTOR_DETECTOR
ROBUST_COMBINED_POLICY
RANDOM_EXTRA_SUPPORT_CONTROL
BAD_ROBUSTNESS_SIGNAL_CONTROL
```

## Success Target

The primary target is `ALL28_UNORDERED`.

Positive D45 requires `ROBUST_COMBINED_POLICY` to preserve clean performance, repair correlated support, improve adversarial support, and beat random/bad robustness controls:

```text
clean_test_accuracy >= 0.995
correlated_support_test_accuracy >= 0.90
adversarial_support_test_accuracy >= 0.95
correlated gain over naive >= 0.50
adversarial gain over naive >= 0.05
random extra support control worse
bad robustness signal control worse
clean regression <= 0.005
```

## Hard Gates

```text
no label echo as fair oracle
no Python hash()
no fake sampling
no fixed synthetic accuracy dict
true family hidden from fair arms
candidate/family/equivalence metrics separated
clean/correlated/adversarial regimes separated
row outputs include support regime and policy diagnostics
no broad claims
```

## Boundary

D45 does not prove raw visual Raven reasoning, Raven solved, DNA/genome success, AGI, consciousness, architecture superiority, or that intelligence is literally a force.
