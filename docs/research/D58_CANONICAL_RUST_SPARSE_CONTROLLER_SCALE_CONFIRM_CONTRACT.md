# D58 Canonical Rust Sparse Controller Scale Confirm Contract

## Purpose

D58 scale-confirms the D57 canonical Rust sparse-path ECF action controller under larger row/seed coverage and stricter bridge audits.

The task remains controlled symbolic joint formula discovery:

```text
formula = operator(cell_a, cell_b) mod 9
```

The Rust sparse controller chooses only the ECF action:

```text
DECIDE
REQUEST_SUPPORT
REQUEST_COUNTER_TOP1_TOP2
REQUEST_JOINT_COUNTER
REQUEST_EXTERNAL_TEST
ABSTAIN
```

It does not learn or solve the formula directly.

## Upstream

D57 must be positive:

```text
decision = canonical_rust_sparse_path_bridge_positive
verdict = D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE_POSITIVE
next = D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRM
```

## Boundary

D58 only scale-confirms a canonical Rust sparse ECF action controller on controlled symbolic joint formula discovery.

It does not prove:

```text
full VRAXION brain
raw visual Raven reasoning
Raven solved
AGI
consciousness
DNA/genome success
architecture superiority
production readiness
```

## Required Bridge Properties

```text
rust_network_path_invoked = true
rust_propagate_sparse_called = true
fallback_rows = 0 for CANONICAL_RUST_SPARSE_CONTROLLER
truth_hidden_from_controller_inputs = true
controller_only_not_formula_solver = true
```

## Arms

```text
D57_CANONICAL_RUST_REPLAY
CANONICAL_RUST_SPARSE_CONTROLLER
PYTHON_SPARSE_REFERENCE
RUST_SPIKE_SHUFFLE_CONTROL
RUST_THRESHOLD_ABLATION
RUST_REWIRE_ABLATION
RUST_PATH_DISABLED_CONTROL
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
ALWAYS_COUNTER_CONTROL
```

## Required Reports

Artifacts are written only under:

```text
target/pilot_wave/d58_canonical_rust_sparse_controller_scale_confirm/
```

Required reports:

```text
queue.json
progress.jsonl
compute_probe.json
dataset_manifest.json
d57_upstream_manifest.json
rust_bridge_invocation_report.json
rust_path_usage_report.json
python_fallback_audit.json
python_vs_rust_action_parity_report.json
firing_dynamics_report.json
network_topology_report.json
support_cost_frontier_report.json
false_confidence_report.json
regime_breakdown_report.json
ablation_report.json
controller_comparison_report.json
min_seed_gate_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
row_outputs_test.jsonl
row_outputs_ood.jsonl
trained_policy_manifest.json
```

## Positive Gate

For `CANONICAL_RUST_SPARSE_CONTROLLER`:

```text
exact_joint_accuracy >= 0.995
correlated_echo_accuracy >= 0.995
adversarial_distractor_accuracy >= 0.995
external_test_required_accuracy >= 0.99
indistinguishable_abstain_rate >= 0.99
indistinguishable_false_confidence_rate <= 0.01
rust_network_path_invoked = true
rust_propagate_sparse_called = true
fallback_rows = 0
action_parity_with_python_reference >= 0.995
beats RANDOM_POLICY_CONTROL
beats GREEDY_DECIDE_CONTROL
beats RUST_SPIKE_SHUFFLE_CONTROL
beats RUST_THRESHOLD_ABLATION
beats RUST_REWIRE_ABLATION
support <= ALWAYS_COUNTER_CONTROL support
min_seed_exact_joint >= 0.99
min_seed_correlated_echo >= 0.99
min_seed_adversarial_distractor >= 0.99
failed_jobs = []
```

## Decisions

Full pass:

```text
decision = canonical_rust_sparse_controller_scale_confirmed
verdict = D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRMED
next = D59_RUST_SPARSE_ECF_CONTROLLER_WITH_MUTATION
```

Non-full run passes:

```text
decision = canonical_rust_sparse_controller_scale_lite_confirmed
verdict = D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_LITE_CONFIRMED
next = D58F_FULL_SCALE_RERUN_OR_D59
```

Rust path unavailable or fallback used:

```text
decision = rust_sparse_path_not_cleanly_exercised
next = D58R_RUST_PATH_REPAIR
```

Failure:

```text
decision = canonical_rust_sparse_controller_scale_not_confirmed
next = D58_REPAIR
```

## Hard Guardrails

```text
no broad claims
no label echo as fair oracle
no Python hash
no fake fixed accuracies
no random threshold hit sampling
truth hidden from controller inputs
Rust invocation explicit
Python fallback zero for canonical arm
ablation controls required
failed jobs visible
```
