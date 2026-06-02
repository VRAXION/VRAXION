# D59 Rust Sparse ECF Controller With Mutation Result

Status: full run completed.

Artifact root:

```text
target/pilot_wave/d59_rust_sparse_ecf_controller_with_mutation/smoke_full
```

## Decision

```text
decision = rust_sparse_mutation_path_confirmed_no_gain
verdict = D59_RUST_SPARSE_MUTATION_PATH_CONFIRMED_NO_GAIN
next = D60_RUST_SPARSE_MUTATION_FITNESS_REPAIR
```

D59 confirmed that the canonical Rust sparse controller representation can be
mutated, selected, serialized, and executed through the generated Rust harness.
It did not show an accuracy-cost gain over the D58 replay reference.

## Run Mode

```text
scale_mode = full
seeds = 11301,11302,11303,11304,11305
train_rows_per_seed = 800
test_rows_per_seed = 800
ood_rows_per_seed = 800
generations = 160
population = 64
```

The requested full mutation run was executed. Pack construction was the slowest
part of the job; the Rust mutation loop itself evaluated 64 candidate
controllers per generation through the generated Rust harness.

## Best Arm

```text
best_arm = RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED
rust_network_path_invoked = true
fallback_rows = 0
mutation_path_exercised = true
```

Key metrics:

```text
exact_joint_accuracy = 0.9994
correlated_echo = 0.99825
adversarial_distractor = 0.99875
external_test_required = 0.9960
indistinguishable_abstain = 1.0000
false_confidence = 0.0000
support = 7.68335
counter_support = 2.68335
cost_adjusted = 0.9840333
```

## Comparison

```text
D58_RUST_REPLAY_REFERENCE:
  exact = 0.9994
  correlated = 0.99825
  adversarial = 0.99875
  external = 0.9960
  abstain = 1.0000
  support = 7.68335
  cost_adjusted = 0.9840333

RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED:
  exact = 0.9994
  correlated = 0.99825
  adversarial = 0.99875
  external = 0.9960
  abstain = 1.0000
  support = 7.68335
  cost_adjusted = 0.9840333

RUST_SPARSE_MUTATION_CONTROLLER_COST_WEIGHTED:
  exact = 0.9936
  correlated = 0.9865
  adversarial = 0.9815
  external = 0.9960
  abstain = 1.0000
  support = 6.48335
  cost_adjusted = 0.9806333
```

Interpretation:

```text
mutation path works
mutation selection did not beat D58 replay
cost-weighted mutation reduced support but lost too much accuracy
false-confidence weighted mutation preserved D58 behavior but did not improve it
```

## Mutation Acceptance

Accepted mutations:

```text
RUST_SPARSE_MUTATION_CONTROLLER:
  accepted_total = 130

RUST_SPARSE_MUTATION_CONTROLLER_COST_WEIGHTED:
  accepted_total = 136

RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED:
  accepted_total = 141
```

Accepted mutation types included:

```text
gate_threshold
gate_weight
gate_priority
gate_channel
gate_action_readout
edge_action_rewire
```

So the negative/no-gain outcome is not because mutation was not exercised.

## Controls

```text
RANDOM_POLICY_CONTROL:
  exact = 0.5663
  correlated = 0.33475
  adversarial = 0.3340

GREEDY_DECIDE_CONTROL:
  exact = 0.5632
  correlated = 0.01475
  adversarial = 0.01125

RUST_SPIKE_SHUFFLE_CONTROL:
  exact = 0.51075
  correlated = 0.01475
  adversarial = 0.01125

RUST_THRESHOLD_ABLATION:
  exact = 0.5632
  correlated = 0.01475
  adversarial = 0.01125

RUST_REWIRE_ABLATION:
  exact = 0.5580
  correlated = 0.0000
  adversarial = 0.0000

ALWAYS_COUNTER_CONTROL:
  exact = 0.9994
  support = 11.0
```

The always-counter control matched high exact accuracy but used much higher
support. The Rust shuffle/threshold/rewire ablations collapsed, so the sparse
action-controller path remained behaviorally meaningful.

## Issue Fixed During Run

The first scale-lite run exposed a fitness semantics bug:

```text
false_confidence = 0.0 did not imply indistinguishable_abstain = 1.0
```

Mutated controllers could preserve low false confidence while losing the
required abstain behavior on indistinguishable cases. The runner was patched so
mutation fitness explicitly includes:

```text
indistinguishable_abstain_rate
external_test_required_accuracy
```

The best-arm selection was also patched to prefer arms that preserve the
indistinguishable abstain boundary before comparing exact accuracy and support.

After the patch, the full run preserved the abstain boundary and passed the D59
checker.

## Validation

```text
python -m py_compile scripts/probes/run_d59_rust_sparse_ecf_controller_with_mutation.py
python -m py_compile scripts/probes/run_d59_rust_sparse_ecf_controller_with_mutation_check.py
python scripts/probes/run_d59_rust_sparse_ecf_controller_with_mutation_check.py --check-only --out target/pilot_wave/d59_rust_sparse_ecf_controller_with_mutation/smoke_full
python scripts/probes/run_d58_canonical_rust_sparse_controller_scale_confirm_check.py --check-only --out target/pilot_wave/d58_canonical_rust_sparse_controller_scale_confirm/smoke
git diff --check
```

All validation commands passed.

## Boundary

```text
D59 only tests mutation and selection of a canonical Rust sparse ECF action
controller for controlled symbolic joint formula discovery.
It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved,
AGI, consciousness, DNA/genome success, architecture superiority, or production
readiness.
```
