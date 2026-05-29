# D47B Indistinguishable Correlated Support Sanity Check Contract

## Goal

D47B tests the identifiability boundary of robust ECF/IPF support policy.

The question is simple:

```text
If wrong correlated support can be separated by a counter-test, does ECF repair it?
If wrong correlated support is indistinguishable under all allowed internal support, does ECF abstain instead of pretending certainty?
If an external/interventional test exists, does only the external-test arm solve it?
```

## Scope

This is a controlled symbolic primitive/cell-reference setup using 3x3 boards and two-cell add mod 9. It is not raw visual Raven reasoning, AGI, consciousness, architecture superiority, or proof that truth can be recovered from indistinguishable evidence.

## Regimes

```text
INDEPENDENT_TRUE_SUPPORT
CORRELATED_TRUE_SUPPORT
DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT
INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT
EXTERNAL_TEST_REQUIRED_SUPPORT
```

The indistinguishable regimes must include a certificate showing that the true and false hypotheses receive identical internal evidence.

## Arms

```text
NAIVE_IPF
ROBUST_ECF_COUNTER_SUPPORT
ROBUST_ECF_WITH_ABSTAIN
ROBUST_ECF_WITH_EXTERNAL_TEST
RANDOM_EXTRA_SUPPORT_CONTROL
LABEL_ECHO_REFERENCE_ONLY
```

`LABEL_ECHO_REFERENCE_ONLY` is not a fair arm.

## Required Behavior

```text
independent true -> solve
correlated true -> solve or safely retain
distinguishable false -> repair with counter-support
indistinguishable false -> abstain / unresolved
external-test required -> only external-test arm solves
```

## Hard Gates

```text
no label echo as fair oracle
no Python hash()
no fake random threshold sampling
no fixed synthetic accuracies
identifiability upper bound reported
indistinguishable certificate reported
false confidence measured
abstain allowed and reported
no broad claims
```
