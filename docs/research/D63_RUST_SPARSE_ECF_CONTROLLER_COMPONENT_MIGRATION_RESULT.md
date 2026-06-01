# D63 Rust Sparse ECF Controller Component Migration Result

Status: accepted smoke / scale-lite run.

## Run

Artifact root:

```text
target/pilot_wave/d63_rust_sparse_ecf_controller_component_migration/smoke
```

Config:

```text
seeds = 11801,11802,11803,11804,11805
train_rows_per_seed = 800
test_rows_per_seed = 800
ood_rows_per_seed = 800
workers = auto
cpu_target = 50-75
scale_mode = scale-lite
heartbeat_sec = 20
```

The run wrote `queue.json`, `progress.jsonl`, partial row/pack/track reports,
per-track metrics, Rust bridge inputs/actions, aggregate reports, and
`decision.json`. No black-box section was used.

## Decision

```text
decision = rust_sparse_ecf_diagnostic_component_migration_confirmed
verdict = D63_RUST_SPARSE_ECF_DIAGNOSTIC_COMPONENT_MIGRATION_CONFIRMED
best_arm = RUST_SPARSE_ALL_DIAGNOSTICS_GATE
next = D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_PROTOTYPE
```

## Result Summary

The D63 migration confirmed that selected ECF diagnostic components can be
estimated through the canonical Rust sparse path and used by the learned D62
gate without Python action fallback.

Best-arm metrics:

| Track | exact | corr | adv | external | abstain | false_confidence | support | counter_support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SATURATED_STABILITY | 0.99930 | 0.99850 | 0.99800 | 0.99525 | 1.00000 | 0.00000 | 8.82785 | 3.82785 |
| HARD_CAP8_LEARNING | 0.99505 | 0.98725 | 0.98800 | 0.01150 | 1.00000 | 0.00000 | 6.47330 | 1.47330 |
| MIXED_EVAL | 0.99690 | 0.99200 | 0.99250 | 0.50525 | 1.00000 | 0.00000 | 7.65230 | 2.65230 |
| OOD_CONTEXT_SHIFT | 0.99505 | 0.98725 | 0.98800 | 0.99525 | 1.00000 | 0.00000 | 6.47330 | 1.47330 |
| ADVERSARIAL_GATE_CONFUSION | 0.99690 | 0.99200 | 0.99250 | 0.50525 | 1.00000 | 0.00000 | 8.73980 | 3.73980 |
| EXTERNAL_TEST_REQUIRED | 0.99930 | 0.99850 | 0.99800 | 0.99525 | 1.00000 | 0.00000 | 8.82785 | 3.82785 |
| INDISTINGUISHABLE_SUPPORT | 0.99930 | 0.99850 | 0.99800 | 0.99525 | 1.00000 | 0.00000 | 8.82785 | 3.82785 |
| NOISY_CONTEXT | 0.99690 | 0.99200 | 0.99250 | 0.50525 | 1.00000 | 0.00000 | 7.93190 | 2.93190 |
| HIDDEN_BUDGET_CONTEXT | 0.99505 | 0.98725 | 0.98800 | 0.01150 | 1.00000 | 0.00000 | 6.47330 | 1.47330 |

Aggregate notes:

```text
min_seed_exact = 0.99375
mixed_regression_vs_D62 = -0.00055
saturated_regression_vs_D62 = -0.00050
fallback_rows = 0
failed_jobs = []
```

## Rust Diagnostic Estimators

Mixed-track estimator accuracy from the checker:

| Diagnostic | accuracy | target_positive_rate | pred_positive_rate | rows |
|---|---:|---:|---:|---:|
| support_budget_pressure | 1.00000 | 0.50000 | 0.50000 | 32000 |
| counterfactual_pressure | 1.00000 | 0.77184 | 0.77184 | 32000 |
| adversarial_pressure | 1.00000 | 0.75000 | 0.75000 | 32000 |
| internal_unresolvable | 1.00000 | 0.25000 | 0.25000 | 32000 |
| external_channel | 1.00000 | 0.12500 | 0.12500 | 32000 |

The Rust invocation report records both policy and diagnostic bridge calls for
each evaluated track. Example diagnostic bridge invocation processed 5
controllers over 32,000 rows with 160,000 `propagate_sparse` calls and
return code 0.

## Controls

Representative mixed-track exact accuracy:

| Arm | mixed exact | mixed support | min_seed_exact |
|---|---:|---:|---:|
| RUST_SPARSE_ALL_DIAGNOSTICS_GATE | 0.99690 | 7.65230 | 0.99375 |
| HYBRID_SYMBOLIC_RUST_DIAGNOSTICS_GATE | 0.99690 | 7.65230 | 0.99375 |
| DIAGNOSTIC_ABLATION_CONTROL | 0.80280 | 7.07330 | 0.60375 |
| RANDOM_DIAGNOSTIC_CONTROL | 0.75305 | 6.23210 | 0.70750 |
| SHUFFLED_DIAGNOSTIC_CONTROL | 0.48165 | 5.03710 | 0.28000 |
| TRUTH_LEAK_SENTINEL_REFERENCE_ONLY | 0.79880 | 7.07330 | 0.00000 |

The truth-leak audit found no forbidden feature use in fair arms:

```text
fair_arms_using_forbidden_features = []
fair_arms_with_truth_leak = []
```

## Validation

Commands passed:

```text
python -m py_compile scripts/probes/run_d63_rust_sparse_ecf_controller_component_migration.py
python -m py_compile scripts/probes/run_d63_rust_sparse_ecf_controller_component_migration_check.py
python scripts/probes/run_d63_rust_sparse_ecf_controller_component_migration_check.py --check-only --out target/pilot_wave/d63_rust_sparse_ecf_controller_component_migration/smoke
git diff --check
```

## Boundary

D63 only tests migration of selected ECF diagnostic components to the Rust
sparse path in controlled symbolic joint formula discovery. It does not prove
full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI,
consciousness, DNA/genome success, architecture superiority, or production
readiness.
