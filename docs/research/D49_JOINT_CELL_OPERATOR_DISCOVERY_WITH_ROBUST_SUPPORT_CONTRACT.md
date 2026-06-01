# D49 Joint Cell Operator Discovery With Robust Support Contract

## Goal

D49 tests whether robust ECF support transfers to joint formula discovery:

```text
target = operator(cell_a, cell_b) mod 9
```

Both the relevant cell pair and the operator are hidden from fair arms.

## Candidate Space

```text
cell pairs:
  CURRENT5
  ALL28 unordered non-center pairs
  ORDERED56 control

operators:
  add
  sub_ab
  sub_ba
  mul
  absdiff
  a_plus_2b
  2a_plus_b
  a_minus_2b

joint candidate:
  cell_pair x operator
```

## Required Metrics

```text
exact joint candidate accuracy
cell-pair equivalence accuracy
cell-hit top2 accuracy
operator exact accuracy
operator equivalence accuracy
family/group accuracy
support and counter-support used
support-cost frontier
false-confidence and abstain behavior
```

## Required D47B-Style Cases

```text
distinguishable correlated false support
indistinguishable correlated false support
external-test-required support
```

Indistinguishable cases must abstain instead of producing high-confidence false answers.

## Positive Route

```text
decision = joint_cell_operator_discovery_with_robust_support_positive
verdict = D49_JOINT_CELL_OPERATOR_DISCOVERY_WITH_ROBUST_SUPPORT_POSITIVE
next = D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM
```

## Boundary

D49 only tests controlled symbolic joint cell+operator discovery with robust ECF support. It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
