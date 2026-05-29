# D48 Operator Selection Discovery With Robust Support Contract

## Goal

D48 tests whether the robust ECF support policy transfers from cell-reference discovery to operator-selection discovery.

The controlled question:

```text
given a fixed/controlled pair of cells,
can the system discover which bounded operator generated the center target,
and can robust support repair correlated/adversarial operator evidence?
```

## Scope

This is a controlled symbolic 3x3 board task. It is not raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or proof that truth can be recovered from indistinguishable evidence.

## Operators

```text
add_mod9
sub_ab_mod9
sub_ba_mod9
mul_mod9
absdiff_mod9
a_plus_2b_mod9
2a_plus_b_mod9
a_minus_2b_mod9
```

## Support Regimes

```text
CLEAN_INDEPENDENT_SUPPORT
CORRELATED_ECHO_SUPPORT
ADVERSARIAL_DISTRACTOR_SUPPORT
MIXED_CLEAN_AND_CORRELATED
MIXED_CLEAN_AND_ADVERSARIAL
```

## Arms

```text
CURRENT_OP_ORACLE_REFERENCE_ONLY
ALL_OPERATOR_ENUMERATION_SOFT_BASELINE
OPERATOR_FAMILY_FACTORISED_FIELD
OPERATOR_EQUIVALENCE_GROUPING
COUNTERFACTUAL_TOP1_TOP2_REPAIR
FULL_ROBUST_ECF_CONTROLLER
FULL_ROBUST_ECF_CONTROLLER_CAP_5
FULL_ROBUST_ECF_CONTROLLER_CAP_7
RANDOM_EXTRA_SUPPORT_CONTROL
BAD_SIGNAL_CONTROL
SHUFFLED_OPERATOR_CONTROL
SHUFFLED_COUNTER_SUPPORT_CONTROL
NO_COUNTERFACTUAL_CONTROL
```

`CURRENT_OP_ORACLE_REFERENCE_ONLY` is a reference arm, not a fair arm.

## Positive Route

```text
decision = operator_selection_discovery_with_robust_support_positive
verdict = D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT_POSITIVE
next = D49_JOINT_CELL_OPERATOR_DISCOVERY_WITH_ROBUST_SUPPORT
```

Positive gates:

```text
clean >= 0.995
correlated >= 0.95
adversarial >= 0.95
mixed >= 0.95
exact_operator_accuracy >= 0.95
operator_equivalence_accuracy >= 0.95
controls worse
failed_jobs = []
```
