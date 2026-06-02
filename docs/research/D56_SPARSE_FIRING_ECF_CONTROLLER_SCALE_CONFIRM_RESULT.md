# D56 Sparse Firing ECF Controller Scale Confirm Result

## Status

```text
status = POSITIVE
scale_mode = scale_lite
decision = sparse_firing_controller_scale_confirmed_python_path_only
verdict = D56_SPARSE_FIRING_CONTROLLER_SCALE_CONFIRMED_PYTHON_PATH_ONLY
next = D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE
```

Artifact root:

```text
target/pilot_wave/d56_sparse_firing_ecf_controller_scale_confirm/smoke
```

D56 scale-confirms the D55 controller-local sparse-firing ECF action path on the controlled symbolic joint formula task. It does not confirm the canonical Rust sparse network runtime path; that path was audited and found available, but was not invoked by this run.

## Run

```text
seeds = 11001,11002,11003,11004,11005
train_rows_per_seed = 800
test_rows_per_seed = 800
ood_rows_per_seed = 800
generations = 160
population = 64
scale_mode = scale_lite
```

The full requested scale was not used; the accepted run is explicitly reported as scale-lite.

## Best Sparse Controller

```text
arm = REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION
exact_joint_accuracy = 0.99925
correlated_echo = 0.99825
adversarial_distractor = 0.99800
external_test_required = 0.99400
indistinguishable_abstain = 1.00000
false_confidence = 0.00000
support = 7.67345
counter_support = 2.67345
```

Sparse firing audit:

```text
sparse_firing_used = true
actual_spike_update_executed = true
spike_update_executed_count = 3,072,000
controller_local_sparse_firing_path_used = true
controller_only_not_formula_solver = true
full_sparse_firing_brain_trained = false
```

## Arm Table

```text
arm                                      exact    corr     adv      external support counter
ALWAYS_COUNTER_CONTROL                   0.99925  0.99825  0.99800  0.35475  11.000  6.000
CONNECTION_REWIRE_ABLATION               0.55975  0.00000  0.00000  0.01375   6.527  1.527
D50_HANDCODED_FULL_REFERENCE             0.99925  0.99825  0.99800  0.99400   8.750  3.750
D54_BEST_HYBRID_REPLAY                   0.99925  0.99825  0.99800  0.99400   7.670  2.670
D55_BEST_SPARSE_REPLAY                   0.99925  0.99825  0.99800  0.99400   7.673  2.673
GREEDY_DECIDE_CONTROL                    0.56490  0.01500  0.01075  0.01375   5.000  0.000
MEDIUM_SPARSE_CONTROLLER                 0.99925  0.99825  0.99800  0.99400   7.674  2.674
RANDOM_POLICY_CONTROL                    0.55725  0.33900  0.33300  0.28400   6.838  1.838
REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION 0.99925 0.99825  0.99800  0.99400   7.673  2.673
REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION 0.99925 0.99825 0.99800 0.99400  7.673  2.673
SMALL_SPARSE_CONTROLLER                  0.99925  0.99825  0.99800  0.99400   7.674  2.674
SPIKE_SHUFFLE_CONTROL                    0.51400  0.01500  0.01075  0.00000   8.053  3.053
THRESHOLD_ABLATION                       0.56490  0.01500  0.01075  0.99400   5.000  0.000
```

The best sparse controller beat the random, greedy, spike-shuffle, threshold, and connection-rewire controls. `ALWAYS_COUNTER_CONTROL` reached high clean/correlated/adversarial scores but failed the external-test behavior and used higher support cost, so it is a cost and boundary control, not the accepted policy.

## Mutation Causality

```text
mutation_changed_accuracy = 0.00000
with_mutation.exact_joint_accuracy = 0.99925
no_mutation.exact_joint_accuracy = 0.99925
d55_replay.exact_joint_accuracy = 0.99925
```

D56 exercised the mutation path, but the final evaluation does not show mutation-caused accuracy improvement over no-mutation or D55 replay. The positive claim is scale-lite stability of the sparse-firing controller-local path, not mutation causality.

## Canonical Rust Path

```text
canonical_rust_network_audited = true
canonical_rust_network_path_available = true
canonical_rust_network_path_invoked = false
rust_network_binary_invoked = false
next_required_bridge = D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE
```

The canonical Rust sparse network surface was audited in:

```text
instnct-core/src/network.rs
instnct-core/src/evolution.rs
```

The audit found sparse propagation and mutation surface markers, but D56 did not run the canonical Rust path as an evaluation arm. That is why the route is `sparse_firing_controller_scale_confirmed_python_path_only`.

## Interpretation

D56 confirms that the controller-local sparse-firing ECF action policy remains strong at scale-lite on controlled symbolic joint formula discovery:

```text
canonical symbolic support
-> sparse-firing controller chooses ECF action
-> robust counter-support / external-test / abstain behavior
-> controlled formula decision stack reads out the final answer
```

It does not show that a full sparse-firing brain learned the formula. It also does not show that the canonical Rust sparse network path is integrated into the controller loop. D57 should bridge and exercise that canonical path directly.

## Boundary

D56 only scale-confirms a sparse-firing ECF controller for controlled symbolic joint formula discovery. It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
