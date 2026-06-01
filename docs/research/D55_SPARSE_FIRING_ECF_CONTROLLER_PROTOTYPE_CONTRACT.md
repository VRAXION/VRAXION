# D55 Sparse Firing ECF Controller Prototype Contract

## Purpose

D55 tests whether the ECF controller action policy from D54 can be executed by a small sparse-firing controller on the controlled symbolic joint formula task.

The formula task remains fixed:

```text
formula = operator(cell_a, cell_b) mod 9
```

The sparse-firing controller does not solve the formula directly. It only chooses the ECF action:

```text
DECIDE
REQUEST_SUPPORT
REQUEST_COUNTER_TOP1_TOP2
REQUEST_JOINT_COUNTER
REQUEST_EXTERNAL_TEST
ABSTAIN
```

## Boundary

D55 only tests a controller-local sparse firing action policy for controlled symbolic joint formula discovery.

It does not prove:

```text
full VRAXION sparse firing brain learning
raw visual Raven reasoning
Raven solved
AGI
consciousness
DNA/genome success
architecture superiority
production readiness
```

## Upstream

D55 starts from accepted D54:

```text
decision = vraxion_sparse_gate_controller_path_confirmed
verdict = D54_VRAXION_SPARSE_GATE_CONTROLLER_PATH_CONFIRMED
next = D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE
```

D54 proved a sparse-gate-style controller path at the mutable controller layer, but explicitly did not run a real sparse firing update.

## Required Arms

```text
D54_BEST_HYBRID_REPLAY
D54_SPARSE_GATE_REPLAY
HANDCODED_D50_FULL_REFERENCE
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
ALWAYS_COUNTER_CONTROL
MUTABLE_RULE_TABLE_REFERENCE
REAL_SPARSE_FIRING_CONTROLLER_SMALL
REAL_SPARSE_FIRING_CONTROLLER_MEDIUM
REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION
REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION
SPIKE_SHUFFLE_CONTROL
FIRING_THRESHOLD_ABLATION
CONNECTION_REWIRE_ABLATION
```

## Sparse Firing Requirement

`sparse_firing_used=true` is allowed only when the runner executes a real spike update:

```text
charge accumulation
threshold decision in the 0..15 stored threshold / 1..16 effective range
phase/channel multiplier
sparse active gate readout
action selected from spike-derived output state
```

The D55 runner may implement this controller-local path in Python, but must report whether the Rust network binary path was invoked.

## Reports

Required generated outputs under:

```text
target/pilot_wave/d55_sparse_firing_ecf_controller_prototype/
```

Required reports:

```text
queue.json
progress.jsonl
compute_probe.json
dataset_manifest.json
d54_upstream_manifest.json
canonical_sparse_firing_audit_report.json
sparse_firing_usage_report.json
network_topology_report.json
firing_dynamics_report.json
mutation_acceptance_report.json
action_readout_report.json
threshold_ablation_report.json
spike_shuffle_control_report.json
support_cost_frontier_report.json
false_confidence_report.json
regime_breakdown_report.json
controller_comparison_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
row_outputs_test.jsonl
row_outputs_ood.jsonl
```

## Gates

For `REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION`:

```text
exact_joint_accuracy >= 0.98
correlated_echo_accuracy >= 0.95
adversarial_distractor_accuracy >= 0.95
external_test_required_accuracy >= 0.95
indistinguishable_abstain_rate >= 0.99
indistinguishable_false_confidence_rate <= 0.01
support <= ALWAYS_COUNTER_CONTROL support
beats RANDOM_POLICY_CONTROL
beats GREEDY_DECIDE_CONTROL
beats SPIKE_SHUFFLE_CONTROL
sparse_firing_used = true
failed_jobs = []
```

## Decisions

Strong positive:

```text
decision = sparse_firing_ecf_controller_prototype_strong_positive
verdict = D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE_STRONG_POSITIVE
next = D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRM
```

Positive but below D54-level:

```text
decision = sparse_firing_ecf_controller_prototype_positive
verdict = D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE_POSITIVE
next = D56_SPARSE_FIRING_CONTROLLER_HARDENING
```

Sparse path not exercised:

```text
decision = sparse_firing_path_not_exercised
next = D55R_SPARSE_INTEGRATION_REPAIR
```

Failure:

```text
decision = sparse_firing_ecf_controller_not_confirmed
next = D55R_SPARSE_CONTROLLER_REPAIR
```

## Hard Guardrails

```text
no label echo as fair oracle
no Python hash
no fake fixed accuracies
no random threshold hit sampling
truth hidden from controller inputs
false confidence measured
ablation controls included
spike shuffle control included
failed jobs visible
no broad claims
```
