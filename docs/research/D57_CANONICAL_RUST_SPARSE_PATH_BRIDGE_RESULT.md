# D57 Canonical Rust Sparse Path Bridge Result

## Status

```text
status = POSITIVE
scale_mode = smoke
decision = canonical_rust_sparse_path_bridge_positive
verdict = D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE_POSITIVE
next = D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRM
```

Artifact root:

```text
target/pilot_wave/d57_canonical_rust_sparse_path_bridge/smoke
```

D57 bridges the D56 controller-local sparse-firing ECF policy to a generated Rust harness that uses the canonical `instnct-core` path dependency and calls `Network::propagate_sparse`.

## Run

```text
seeds = 11101,11102,11103,11104,11105
train_rows_per_seed = 800
test_rows_per_seed = 800
ood_rows_per_seed = 800
scale_mode = smoke
```

The Rust bridge harness was generated under:

```text
target/pilot_wave/d57_canonical_rust_sparse_path_bridge/smoke/rust_bridge_harness
```

The harness used:

```text
Network::new
Network::graph_mut().add_edge
Network::propagate_sparse
Network::spike_data
```

## Bridge Audit

```text
rust_network_path_invoked = true
rust_propagate_sparse_called = true
canonical_arm_python_fallback_rows = 0
python_vs_rust_action_parity = 1.00000
compared_actions = 64,000
matched_actions = 64,000
```

Canonical Rust invocation report:

```text
test rows_processed = 32,000
test propagate_sparse_calls = 32,000
ood rows_processed = 32,000
ood propagate_sparse_calls = 32,000
gate_count = 6
```

The canonical bridge arm did not use the Python sparse controller as a fallback. The Python replay arm remains in the run only as a reference/parity control.

## Canonical Rust Bridge Metrics

```text
arm = CANONICAL_RUST_SPARSE_NETWORK_BRIDGE
exact_joint_accuracy = 0.99955
correlated_echo = 0.99875
adversarial_distractor = 0.99900
external_test_required = 0.99225
indistinguishable_abstain = 1.00000
false_confidence = 0.00000
support = 7.66520
counter_support = 2.66520
```

## Arm Table

```text
arm                                exact    corr     adv      external support counter rust
ALWAYS_COUNTER_CONTROL             0.99955  0.99875  0.99900  0.35450  11.000  6.000  false
CANONICAL_RUST_SPARSE_NETWORK_BRIDGE 0.99955 0.99875 0.99900 0.99225  7.665  2.665  true
D56_PYTHON_SPARSE_REPLAY           0.99955  0.99875  0.99900  0.99225   7.665  2.665  false
GREEDY_DECIDE_CONTROL              0.56795  0.01450  0.01025  0.01300   5.000  0.000  false
PYTHON_FALLBACK_REFERENCE          0.99955  0.99875  0.99900  0.99225   7.665  2.665  false
RANDOM_POLICY_CONTROL              0.56325  0.33400  0.33250  0.29175   6.843  1.843  false
RUST_PATH_DISABLED_CONTROL         0.56795  0.01450  0.01025  0.01300   5.000  0.000  false
RUST_REWIRE_ABLATION               0.56300  0.00000  0.00000  0.01300   6.535  1.535  true
RUST_SPIKE_SHUFFLE_CONTROL         0.51655  0.01450  0.01025  0.00000   8.070  3.070  true
RUST_THRESHOLD_ABLATION            0.56795  0.01450  0.01025  0.99225   5.000  0.000  true
```

The Rust ablations and disabled-path controls collapse while the canonical Rust bridge matches the D56 Python sparse replay. This supports the narrow bridge claim: the canonical Rust sparse propagation path can carry the D56 sparse controller gate firing and action readout for this controlled ECF action task.

## Interpretation

D57 confirms:

```text
controlled symbolic support features
-> canonical Rust Network::propagate_sparse gate firing
-> Rust marker/readout action
-> robust ECF support/counter-support/external-test/abstain behavior
```

D57 does not show that a full Rust sparse network learned the formula. It does not show raw visual Raven reasoning or general intelligence. It bridges the controller action path only.

## Boundary

D57 only tests a canonical Rust sparse-path bridge for the ECF action controller on controlled symbolic joint formula discovery. It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
