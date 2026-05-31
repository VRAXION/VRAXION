# D62 Policy Ensemble ECF Controller With Learned Gate Contract

## Purpose

D62 tests whether the D61 two-policy gate can be generalized into a learned policy-ensemble gate.

The narrow question:

```text
Can a fair learned gate choose among saturated, hard-budget, counterfactual,
external-test, abstain, and adversarial-repair Rust sparse ECF action-controller
modules without using truth labels or support-regime labels?
```

## Boundary

This is controlled symbolic joint formula discovery only. The Rust sparse controller chooses ECF actions such as `DECIDE`, `REQUEST_COUNTER_TOP1_TOP2`, `REQUEST_JOINT_COUNTER`, `REQUEST_EXTERNAL_TEST`, and `ABSTAIN`.

The formula solver remains the fixed symbolic stack. D62 does not claim full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

## Upstream

Expected upstream D61:

```text
decision = gated_rust_sparse_mutation_scale_confirmed
verdict = D61_GATED_RUST_SPARSE_MUTATION_SCALE_CONFIRMED
next = D62_POLICY_ENSEMBLE_ECF_CONTROLLER_WITH_LEARNED_GATE
```

D61 showed that:

```text
SATURATED_STABILITY needs the D59-like saturated policy.
HARD_CAP8_LEARNING needs the D60 hard-budget policy.
MIXED/OOD contexts can be repaired by a fair runtime gate.
```

## Policy Modules

```text
SATURATED_POLICY
HARD_BUDGET_POLICY
COUNTERFACTUAL_POLICY
EXTERNAL_TEST_POLICY
ABSTAIN_POLICY
ADVERSARIAL_REPAIR_POLICY
```

These are controller/action modules. They are not formula solvers.

## Tracks

```text
SATURATED_STABILITY
HARD_CAP8_LEARNING
MIXED_EVAL
OOD_CONTEXT_SHIFT
ADVERSARIAL_GATE_CONFUSION
EXTERNAL_TEST_REQUIRED
INDISTINGUISHABLE_SUPPORT
NOISY_CONTEXT
HIDDEN_BUDGET_CONTEXT
```

Fair gates may use observable runtime diagnostics only. They may not use truth labels, row ids, support regime labels, track labels, mixed-source labels, true cells/operators, or expected answers.

## Required Reports

Artifacts are written only under:

```text
target/pilot_wave/d62_policy_ensemble_ecf_controller_with_learned_gate/
```

Required reports include:

```text
d61_upstream_manifest.json
gate_feature_audit_report.json
truth_leak_audit_report.json
observable_feature_origin_report.json
gate_training_report.json
gate_routing_accuracy_report.json
policy_ensemble_report.json
explicit_vs_inferred_gate_report.json
noisy_hidden_context_report.json
external_test_policy_report.json
abstain_policy_report.json
support_cost_frontier_report.json
false_confidence_report.json
rust_invocation_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
row_outputs_test.jsonl
row_outputs_ood.jsonl
trained_policy_manifest.json
```

Long runs must write `queue.json`, `progress.jsonl`, partial row-generation, pack-build, track-clone, per-track metric snapshots, per-track final metrics, and final reports. There is no black-box run.

## Positive Gate

The best fair learned ensemble must satisfy:

```text
saturated_exact >= 0.9974
hard_exact >= 0.99
mixed_exact >= 0.995
OOD_exact >= 0.99
external_test_required >= 0.99
indistinguishable_abstain >= 0.99
false_confidence <= 0.01
hard_gain_vs_D58 >= +0.30
saturated_regression_vs_D59 >= -0.002
noisy_context_exact >= 0.99
hidden_budget_context_exact >= 0.99
Rust path invoked = true
fallback rows = 0
controls worse
failed jobs = []
```

## Decision Logic

```text
If learned ensemble passes inferred/noisy/hidden context:
  decision = policy_ensemble_learned_gate_confirmed
  next = D63_RUST_SPARSE_ECF_CONTROLLER_COMPONENT_MIGRATION

If only explicit budget gate works:
  decision = explicit_context_gate_required
  next = D62B_DIAGNOSTIC_GATE_REPAIR

If external/abstain policy fails:
  decision = policy_ensemble_external_abstain_gap
  next = D62E_EXTERNAL_ABSTAIN_GATE_REPAIR

If a fair gate leaks truth/forbidden labels:
  decision = invalid_gate_truth_leak_detected
  next = D62R_GATE_FEATURE_REPAIR

Otherwise:
  decision = policy_ensemble_learned_gate_not_confirmed
  next = D62_REPAIR
```
