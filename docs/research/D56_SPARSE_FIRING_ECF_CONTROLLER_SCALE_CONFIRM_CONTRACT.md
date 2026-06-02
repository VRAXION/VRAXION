# D56 Sparse Firing ECF Controller Scale Confirm Contract

## Purpose

D56 scale-confirms the D55 sparse-firing ECF action controller on the controlled symbolic joint formula task.

The task remains:

```text
formula = operator(cell_a, cell_b) mod 9
```

The sparse-firing controller chooses only the ECF action:

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

D55 must be positive:

```text
decision = sparse_firing_ecf_controller_prototype_strong_positive
verdict = D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE_STRONG_POSITIVE
next = D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRM
```

## Boundary

D56 only scale-confirms a sparse-firing ECF controller for controlled symbolic joint formula discovery.

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

## Arms

```text
D55_BEST_SPARSE_REPLAY
D54_BEST_HYBRID_REPLAY
D50_HANDCODED_FULL_REFERENCE
REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION
REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION
SMALL_SPARSE_CONTROLLER
MEDIUM_SPARSE_CONTROLLER
SPIKE_SHUFFLE_CONTROL
THRESHOLD_ABLATION
CONNECTION_REWIRE_ABLATION
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
ALWAYS_COUNTER_CONTROL
```

`CANONICAL_RUST_NETWORK_PATH_PROBE` is reported separately. If the canonical Rust runtime path is not invoked, D56 must say so explicitly and route accordingly.

## Required Reports

Artifacts are written only under:

```text
target/pilot_wave/d56_sparse_firing_ecf_controller_scale_confirm/
```

Required reports:

```text
queue.json
progress.jsonl
compute_probe.json
dataset_manifest.json
d55_upstream_manifest.json
sparse_firing_usage_report.json
canonical_sparse_path_report.json
python_local_vs_rust_path_report.json
mutation_causality_report.json
network_topology_report.json
firing_dynamics_report.json
mutation_acceptance_report.json
action_readout_report.json
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

For the best D56 sparse controller:

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
beats SPIKE_SHUFFLE_CONTROL
beats THRESHOLD_ABLATION
beats CONNECTION_REWIRE_ABLATION
sparse_firing_used = true
min_seed_exact_joint >= 0.99
failed_jobs = []
```

## Decisions

Rust path invoked and scale passes:

```text
decision = sparse_firing_ecf_controller_scale_confirmed
verdict = D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRMED
next = D57_CANONICAL_VRAXION_SPARSE_NETWORK_INTEGRATION
```

Python-local sparse path passes, but Rust network path is not invoked:

```text
decision = sparse_firing_controller_scale_confirmed_python_path_only
verdict = D56_SPARSE_FIRING_CONTROLLER_SCALE_CONFIRMED_PYTHON_PATH_ONLY
next = D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE
```

Failure:

```text
decision = sparse_firing_ecf_controller_scale_not_confirmed
next = D56_REPAIR
```

## Hard Guardrails

```text
no broad claims
no label echo as fair oracle
no Python hash
no fake fixed accuracies
no random threshold hit sampling
truth hidden from controller inputs
sparse_firing_used true/false explicit
canonical Rust path available/unavailable explicit
ablation controls required
failed jobs visible
```
