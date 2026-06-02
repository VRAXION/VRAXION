# D50 Joint Formula Discovery Scale Confirm Contract

## Goal

D50 scale-confirms controlled symbolic joint formula discovery after D49B repaired the joint binding bottleneck.

```text
formula = operator(cell_a, cell_b) mod 9
```

Both the relevant cell pair and operator are hidden from fair arms.

## Upstream

```text
D49B decision = joint_binding_repair_positive
D49B verdict = D49B_JOINT_BINDING_REPAIR_POSITIVE
D49B next = D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM
```

## Arms

```text
D49B_BASELINE_REPLAY
JOINT_INTERACTION_COUNTERFACTUAL
MULTI_STAGE_COUNTERFACTUAL_REPAIR
FULL_REPAIRED_ECF_CONTROLLER
FULL_REPAIRED_ECF_CAP_7
FULL_REPAIRED_ECF_CAP_9
NO_CELL_COUNTERFACTUAL
NO_OPERATOR_COUNTERFACTUAL
NO_JOINT_INTERACTION_COUNTERFACTUAL
RANDOM_EXTRA_SUPPORT_CONTROL
SHUFFLED_COUNTER_SUPPORT_CONTROL
NO_COUNTERFACTUAL_CONTROL
ABSTAIN_ON_INDISTINGUISHABLE
```

## Required Metrics

```text
exact_joint
cell_pair_equivalence
cell_hit_top2
operator_exact
operator_equivalence
clean/correlated/adversarial/mixed accuracy
distinguishable false accuracy
indistinguishable abstain rate
indistinguishable false confidence rate
external_test_required accuracy
support used
counter-support used
support-cost frontier
min-seed exact/correlated/adversarial
failed jobs
```

## Positive Route

```text
decision = joint_formula_discovery_scale_confirmed
verdict = D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRMED
next = D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE
```

The positive gate requires:

```text
FULL_REPAIRED_ECF exact_joint >= 0.995
correlated_echo >= 0.995
adversarial_distractor >= 0.995
external_test_required >= 0.99
false_confidence <= 0.01
min_seed_exact_joint >= 0.99
controls worse
failed_jobs = []
```

## Boundary

D50 only scale-confirms controlled symbolic joint formula discovery with robust ECF support. It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
