# D58 Canonical Rust Sparse Controller Scale Confirm Result

## Status

```text
status = POSITIVE
scale_mode = full
decision = canonical_rust_sparse_controller_scale_confirmed
verdict = D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRMED
next = D59_RUST_SPARSE_ECF_CONTROLLER_WITH_MUTATION
```

Artifact root:

```text
target/pilot_wave/d58_canonical_rust_sparse_controller_scale_confirm/smoke
```

D58 scale-confirms the D57 canonical Rust sparse controller path under larger coverage.

## Run

```text
seeds = 11201,11202,11203,11204,11205,11206,11207,11208
train_rows_per_seed = 1200
test_rows_per_seed = 1200
ood_rows_per_seed = 1200
scale_mode = full
```

This run used the canonical Rust sparse bridge harness under the D58 artifact root, with `instnct-core` as a path dependency.

## Bridge Audit

```text
rust_network_path_invoked = true
rust_propagate_sparse_called = true
canonical_arm_python_fallback_rows = 0
python_vs_rust_action_parity = 1.00000
compared_actions = 153,600
matched_actions = 153,600
```

The canonical arm used Rust rows:

```text
CANONICAL_RUST_SPARSE_CONTROLLER.rust_rows = 76,800
D57_CANONICAL_RUST_REPLAY.rust_rows = 76,800
RUST_THRESHOLD_ABLATION.rust_rows = 76,800
RUST_REWIRE_ABLATION.rust_rows = 76,800
RUST_SPIKE_SHUFFLE_CONTROL.rust_rows = 76,800
```

## Canonical Rust Controller Metrics

```text
arm = CANONICAL_RUST_SPARSE_CONTROLLER
exact_joint_accuracy = 0.999729
correlated_echo = 0.998854
adversarial_distractor = 0.999792
external_test_required = 0.993021
indistinguishable_abstain = 1.000000
false_confidence = 0.000000
support = 7.670688
counter_support = 2.670688
```

Seed floor:

```text
min_seed_exact_joint = 0.999333
min_seed_correlated_echo = 0.996667
min_seed_adversarial_distractor = 0.998333
```

## Arm Table

```text
arm                          exact     corr      adv       external  support  counter  rust
ALWAYS_COUNTER_CONTROL       0.999729  0.998854  0.999792  0.357604  11.000   6.000    false
CANONICAL_RUST_SPARSE_CONTROLLER 0.999729 0.998854 0.999792 0.993021 7.671 2.671 true
D57_CANONICAL_RUST_REPLAY    0.999729  0.998854  0.999792  0.993021   7.671   2.671    true
GREEDY_DECIDE_CONTROL        0.566625  0.014063  0.013438  0.012813   5.000   0.000    false
PYTHON_SPARSE_REFERENCE      0.999729  0.998854  0.999792  0.993021   7.671   2.671    false
RANDOM_POLICY_CONTROL        0.571396  0.336667  0.340104  0.288333   6.830   1.830    false
RUST_PATH_DISABLED_CONTROL   0.566625  0.014063  0.013438  0.012813   5.000   0.000    false
RUST_REWIRE_ABLATION         0.561125  0.000000  0.000000  0.012813   6.529   1.529    true
RUST_SPIKE_SHUFFLE_CONTROL   0.515271  0.014063  0.013438  0.000000   8.059   3.059    true
RUST_THRESHOLD_ABLATION      0.566625  0.014063  0.013438  0.993021   5.000   0.000    true
```

The canonical Rust sparse controller matches the D57 replay and Python sparse reference while keeping `fallback_rows = 0`. The random, greedy, path-disabled, spike-shuffle, threshold, and rewire controls are all much worse.

## Runtime Note

An initial full run attempt exposed a bad partial-progress implementation: heartbeat snapshots summarized the full accumulated split each time, which became too slow near the end of the test split. The runner was patched so heartbeat snapshots summarize only a bounded recent window while final reports still summarize the complete split. The final full run completed cleanly after that fix.

## Interpretation

D58 confirms:

```text
controlled symbolic support features
-> canonical Rust Network::propagate_sparse gate firing
-> Rust sparse controller action readout
-> robust ECF support/counter-support/external-test/abstain behavior
```

It confirms the canonical Rust sparse ECF action controller at full D58 scale. It still does not show that a full Rust sparse network learned the formula itself.

## Boundary

D58 only scale-confirms a canonical Rust sparse ECF action controller on controlled symbolic joint formula discovery. It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
