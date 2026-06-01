# D49B Joint Binding Repair Contract

## Goal

D49B tests whether the D49 joint-binding bottleneck can be repaired by separating the error source:

```text
cell-pair error
operator error
joint interaction binding error
```

The task remains the D49 controlled symbolic setup:

```text
target = operator(cell_a, cell_b) mod 9
candidate = cell_pair x operator
```

Both the relevant cells and the operator are hidden from fair arms.

## Repair Arms

```text
D49_BASELINE_REPLAY
FACTORISED_CELL_OPERATOR_SCORE
CELL_FIRST_OPERATOR_SECOND_PIPELINE
OPERATOR_FIRST_CELL_SECOND_PIPELINE
JOINT_BINDING_MATRIX
CELL_ONLY_COUNTERFACTUAL
OPERATOR_ONLY_COUNTERFACTUAL
JOINT_INTERACTION_COUNTERFACTUAL
MULTI_STAGE_COUNTERFACTUAL_REPAIR
FULL_REPAIRED_ECF_CONTROLLER
FULL_REPAIRED_ECF_CAP_7
FULL_REPAIRED_ECF_CAP_9
RANDOM_EXTRA_SUPPORT_CONTROL
SHUFFLED_COUNTER_SUPPORT_CONTROL
NO_COUNTERFACTUAL_CONTROL
ABSTAIN_ON_INDISTINGUISHABLE
```

The multi-stage repair uses:

```text
stage A: distinguish cell ambiguity
stage B: distinguish operator ambiguity
stage C: distinguish joint interaction ambiguity
```

## Required Diagnostics

```text
joint_error_taxonomy_report.json
cell_vs_operator_error_report.json
binding_consistency_report.json
counterfactual_stage_report.json
external_test_required_report.json
support_cost_frontier_report.json
regime_breakdown_report.json
```

The required error categories are:

```text
ok
cell_only_error
operator_only_error
both_cell_and_operator_wrong
joint_interaction_binding_error
indistinguishable_abstain
false_confidence_on_unidentifiable
external_test_required_unresolved
```

## Positive Route

```text
decision = joint_binding_repair_positive
verdict = D49B_JOINT_BINDING_REPAIR_POSITIVE
next = D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM
```

The positive gate requires the full repaired controller to meet:

```text
clean >= 0.995
correlated_echo >= 0.95
adversarial_distractor >= 0.95
mixed >= 0.95
exact_joint >= 0.97
cell_pair_equivalence >= 0.97
operator_exact >= 0.97
false_confidence_indistinguishable <= 0.01
controls worse
failed_jobs = []
```

## Boundary

D49B only tests controlled symbolic joint cell+operator binding repair with robust ECF support. It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
