# D61 Gated Rust Sparse Mutation Scale Confirm Contract

## Purpose

D61 scale-confirms the D60S finding that a gated Rust sparse ECF action controller can preserve saturated stability while using the hard-cap policy where the support budget is constrained.

The question is narrow:

```text
Can a fair runtime gate choose between the D59 saturated Rust controller and the D60/D60S hard-cap Rust controller across saturated, hard, mixed, OOD, and gate-confusion contexts?
```

## Boundary

This is controlled symbolic joint formula discovery only. The Rust sparse network chooses ECF actions such as `DECIDE`, `REQUEST_JOINT_COUNTER`, `REQUEST_EXTERNAL_TEST`, and `ABSTAIN`. The symbolic formula solver remains outside the sparse controller.

D61 does not claim full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

## Upstream

Expected upstream D60S:

```text
decision = gated_policy_required_for_no_forgetting
verdict = D60S_GATED_POLICY_REQUIRED_FOR_NO_FORGETTING
next = D61_GATED_RUST_SPARSE_MUTATION_SCALE_CONFIRM
```

The D60S result established that:

```text
SATURATED_STABILITY needs the D59-like saturated policy.
HARD_CAP8_LEARNING needs the D60 hard-cap policy.
MIXED_EVAL is repaired by a gate that selects the right policy for the runtime context.
```

## Tracks

```text
SATURATED_STABILITY
HARD_CAP8_LEARNING
MIXED_EVAL
OOD_CONTEXT_SHIFT
ADVERSARIAL_GATE_CONFUSION
```

The fair gate may use runtime budget and diagnostic features. It may not use truth labels, row ids, support regime labels, track labels, or mixed-source labels.

## Arms

```text
D59_REFERENCE
D60_HARD_POLICY_REPLAY
SINGLE_POLICY_MULTI_ENV_CONTROL
THRESHOLD_ABLATION
REWIRE_ABLATION
D60S_DUAL_POLICY_GATED_REPLAY
DUAL_POLICY_GATED_CONTROLLER
CONTEXT_GATED_POLICY_ENSEMBLE
LEARNED_GATE_MUTATION_CONTROLLER
ORACLE_TRACK_GATE_REFERENCE_ONLY
TRUTH_LEAK_SENTINEL_CONTROL
RANDOM_GATE_CONTROL
WRONG_GATE_CONTROL
GATE_ABLATION
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
ALWAYS_COUNTER_CONTROL
SPIKE_SHUFFLE_CONTROL
```

Reference-only arms are not fair evidence.

## Required Artifacts

Artifacts are written only under:

```text
target/pilot_wave/d61_gated_rust_sparse_mutation_scale_confirm/
```

Required reports:

```text
d60s_upstream_manifest.json
gate_training_report.json
gate_feature_audit_report.json
gate_routing_accuracy_report.json
truth_leak_audit_report.json
multi_track_scale_report.json
ood_context_shift_report.json
adversarial_gate_confusion_report.json
gate_ablation_report.json
policy_comparison_report.json
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

Long runs must write `queue.json`, `progress.jsonl`, partial snapshots, and final reports. There is no black-box run.

## Positive Gate

The best fair gated controller must satisfy:

```text
saturated exact >= D59 reference - 0.002
hard exact >= 0.99
mixed exact >= 0.995
OOD exact >= 0.99
adversarial gate-confusion exact >= 0.99
hard gain vs D58 >= 0.30
saturated regression vs D59 >= -0.002
false confidence <= 0.01
indistinguishable abstain >= 0.99
Rust path invoked = true
fallback rows = 0
controls worse
failed jobs = []
```

## Decision Logic

```text
If learned fair gate passes:
  decision = gated_rust_sparse_mutation_scale_confirmed
  next = D62_POLICY_ENSEMBLE_ECF_CONTROLLER_WITH_LEARNED_GATE

If only handcoded fair gate passes:
  decision = handcoded_gate_required
  next = D61L_LEARNED_GATE_REPAIR

If a fair gate leaks truth/forbidden labels:
  decision = invalid_gate_truth_leak_detected
  next = D61R_GATE_FEATURE_REPAIR

Otherwise:
  decision = gated_rust_sparse_mutation_scale_not_confirmed
  next = D61_REPAIR
```
