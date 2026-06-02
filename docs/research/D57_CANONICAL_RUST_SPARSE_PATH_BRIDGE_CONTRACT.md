# D57 Canonical Rust Sparse Path Bridge Contract

## Purpose

D57 bridges the D56 controller-local sparse-firing ECF action policy to the canonical Rust sparse network path.

The task remains controlled symbolic joint formula discovery:

```text
formula = operator(cell_a, cell_b) mod 9
```

The sparse controller still chooses only the ECF action:

```text
DECIDE
REQUEST_SUPPORT
REQUEST_COUNTER_TOP1_TOP2
REQUEST_JOINT_COUNTER
REQUEST_EXTERNAL_TEST
ABSTAIN
```

D57 does not train a full sparse-firing brain and does not make the Rust network solve the formula directly. It tests whether the sparse controller gate firing and action readout can be driven through `instnct-core`'s canonical Rust `Network::propagate_sparse` path instead of the D55/D56 controller-local Python tick.

## Upstream

D56 must be positive:

```text
decision = sparse_firing_controller_scale_confirmed_python_path_only
verdict = D56_SPARSE_FIRING_CONTROLLER_SCALE_CONFIRMED_PYTHON_PATH_ONLY
next = D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE
```

## Boundary

D57 only tests a canonical Rust sparse-path bridge for the ECF action controller on controlled symbolic joint formula discovery.

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
canonical_rust_network_path_invoked = true
rust_propagate_sparse_called = true
python_fallback_used_for_canonical_arm = false
controller_only_not_formula_solver = true
truth_hidden_from_controller_inputs = true
```

The Rust bridge must be generated under the D57 target artifact root and must use `instnct-core` as a path dependency. It must not be represented by source-code audit alone.

## Arms

```text
D56_PYTHON_SPARSE_REPLAY
PYTHON_FALLBACK_REFERENCE
CANONICAL_RUST_SPARSE_NETWORK_BRIDGE
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
target/pilot_wave/d57_canonical_rust_sparse_path_bridge/
```

Required reports:

```text
queue.json
progress.jsonl
compute_probe.json
dataset_manifest.json
d56_upstream_manifest.json
rust_bridge_build_report.json
rust_bridge_invocation_report.json
rust_path_usage_report.json
python_fallback_audit.json
python_vs_rust_action_parity_report.json
canonical_sparse_path_report.json
action_readout_report.json
firing_dynamics_report.json
support_cost_frontier_report.json
false_confidence_report.json
regime_breakdown_report.json
ablation_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
row_outputs_test.jsonl
row_outputs_ood.jsonl
```

## Positive Gate

For `CANONICAL_RUST_SPARSE_NETWORK_BRIDGE`:

```text
exact_joint_accuracy >= 0.995
correlated_echo_accuracy >= 0.995
adversarial_distractor_accuracy >= 0.995
external_test_required_accuracy >= 0.99
indistinguishable_abstain_rate >= 0.99
indistinguishable_false_confidence_rate <= 0.01
support <= ALWAYS_COUNTER_CONTROL support
beats RANDOM_POLICY_CONTROL
beats GREEDY_DECIDE_CONTROL
beats RUST_SPIKE_SHUFFLE_CONTROL
beats RUST_THRESHOLD_ABLATION
beats RUST_REWIRE_ABLATION
rust_network_path_invoked = true
rust_propagate_sparse_called = true
python_fallback_used = false
action_parity_with_d56_python_replay >= 0.995
failed_jobs = []
```

## Decisions

Pass:

```text
decision = canonical_rust_sparse_path_bridge_positive
verdict = D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE_POSITIVE
next = D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRM
```

Rust path invoked but behavior does not match D56:

```text
decision = rust_path_invoked_but_behavior_mismatch
verdict = D57_RUST_PYTHON_SEMANTICS_MISMATCH
next = D57B_RUST_PYTHON_SEMANTICS_ALIGNMENT
```

Rust path not actually invoked:

```text
decision = rust_path_not_actually_invoked
verdict = D57_BRIDGE_INSTRUMENTATION_FAILURE
next = D57R_BRIDGE_INSTRUMENTATION_REPAIR
```

General failure:

```text
decision = canonical_rust_sparse_path_bridge_not_confirmed
next = D57_REPAIR
```

## Hard Guardrails

```text
no broad claims
no label echo as fair oracle
no Python hash
no fake fixed accuracies
no random threshold hit sampling
truth hidden from controller inputs
canonical Rust path must be invoked for canonical arm
Python fallback must be explicit and cannot be the canonical arm
ablation controls required
failed jobs visible
```
