# D59 Rust Sparse ECF Controller With Mutation Contract

## Purpose

D59 tests whether the canonical Rust sparse ECF action controller can be mutated and selected using a Rust-serializable sparse controller representation.

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

The formula solver remains the symbolic stack. D59 does not train a full sparse-firing formula solver.

## Upstream

D58 must be positive:

```text
decision = canonical_rust_sparse_controller_scale_confirmed
verdict = D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRMED
next = D59_RUST_SPARSE_ECF_CONTROLLER_WITH_MUTATION
```

## Boundary

D59 only tests mutation and selection of a canonical Rust sparse ECF action controller for controlled symbolic joint formula discovery.

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

## Required Mutation Properties

```text
mutation_representation = Rust-serializable sparse controller config
candidate_eval_path = canonical Rust Network::propagate_sparse
fallback_rows = 0 for Rust arms
accepted/rejected mutation counts reported by type
fitness before/after reported
train/validation/test/ood separation reported
support cost and false confidence included in fitness
```

The Python runner may orchestrate mutation and selection, but candidate actions must be produced by the generated Rust sparse harness.

## Arms

```text
D58_RUST_REPLAY_REFERENCE
RUST_SPARSE_MUTATION_CONTROLLER
RUST_SPARSE_MUTATION_CONTROLLER_COST_WEIGHTED
RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED
MUTATION_DISABLED_CONTROL
RANDOM_MUTATION_CONTROL
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
ALWAYS_COUNTER_CONTROL
RUST_SPIKE_SHUFFLE_CONTROL
RUST_THRESHOLD_ABLATION
RUST_REWIRE_ABLATION
```

## Required Reports

Artifacts are written only under:

```text
target/pilot_wave/d59_rust_sparse_ecf_controller_with_mutation/
```

Required reports:

```text
queue.json
progress.jsonl
compute_probe.json
dataset_manifest.json
d58_upstream_manifest.json
rust_mutation_representation_report.json
rust_invocation_report.json
rust_path_usage_report.json
python_fallback_audit.json
mutation_acceptance_report.json
fitness_landscape_report.json
before_after_mutation_report.json
support_cost_frontier_report.json
false_confidence_report.json
regime_breakdown_report.json
ablation_report.json
controller_comparison_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
row_outputs_test.jsonl
row_outputs_ood.jsonl
trained_policy_manifest.json
```

## Positive Gate

For the best Rust-mutated controller:

```text
exact_joint_accuracy >= 0.995
correlated_echo_accuracy >= 0.995
adversarial_distractor_accuracy >= 0.995
external_test_required_accuracy >= 0.99
indistinguishable_abstain_rate >= 0.99
indistinguishable_false_confidence_rate <= 0.01
rust_path_invoked = true
fallback_rows = 0
beats RANDOM_POLICY_CONTROL
beats GREEDY_DECIDE_CONTROL
beats RUST_SPIKE_SHUFFLE_CONTROL
beats RUST_THRESHOLD_ABLATION
beats RUST_REWIRE_ABLATION
ALWAYS_COUNTER_CONTROL has higher support cost
mutation path exercised with accepted mutations
failed_jobs = []
```

## Decisions

Pass and accuracy-cost improves:

```text
decision = rust_sparse_mutation_controller_positive
verdict = D59_RUST_SPARSE_MUTATION_CONTROLLER_POSITIVE
next = D60_RUST_SPARSE_CONTROLLER_COST_OPTIMIZATION
```

Pass but no gain over D58 replay:

```text
decision = rust_sparse_mutation_path_confirmed_no_gain
verdict = D59_RUST_SPARSE_MUTATION_PATH_CONFIRMED_NO_GAIN
next = D60_RUST_SPARSE_MUTATION_FITNESS_REPAIR
```

Mutation not actually exercised:

```text
decision = rust_sparse_mutation_path_not_exercised
next = D59R_RUST_MUTATION_BRIDGE_REPAIR
```

Failure:

```text
decision = rust_sparse_mutation_controller_not_confirmed
next = D59_REPAIR
```

## Hard Guardrails

```text
no broad claims
no label echo as fair oracle
no Python hash
no fake fixed accuracies
no random threshold hit sampling
truth hidden from controller inputs
Rust path invoked for candidate policies
fallback rows must be 0 for Rust arms
mutation representation explicit
ablations required
failed jobs visible
```
