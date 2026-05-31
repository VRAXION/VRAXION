# D64 Rust Sparse IPF Diagnostic Layer Prototype Result

## Decision

```text
decision = rust_sparse_ipf_diagnostic_layer_confirmed
verdict = D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_CONFIRMED
next = D65_RUST_SPARSE_IPF_SCORE_AGGREGATION_PROTOTYPE
best_arm = RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER
```

Artifacts:

```text
target/pilot_wave/d64_rust_sparse_ipf_diagnostic_layer_prototype/smoke/
```

## What Ran

Scale-lite smoke:

```text
seeds = 11901,11902,11903,11904,11905
train_rows_per_seed = 800
test_rows_per_seed = 800
ood_rows_per_seed = 800
cpu_target = 50-75
heartbeat_sec = 20
```

The run wrote `queue.json`, `progress.jsonl`, partial row/pack progress,
per-track metric snapshots, per-track final metrics, and final reports.

## Core Result

`RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER`:

| track | exact | corr | adv | external | abstain | false_conf | support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SATURATED_STABILITY | 0.994900 | 0.989000 | 0.985500 | 0.995000 | 1.000000 | 0.000000 | 6.4736 |
| HARD_CAP8_LEARNING | 0.994900 | 0.989000 | 0.985500 | 0.012250 | 1.000000 | 0.000000 | 6.4736 |
| MIXED_EVAL | 0.994900 | 0.989000 | 0.985500 | 0.503000 | 1.000000 | 0.000000 | 6.4736 |
| OOD_CONTEXT_SHIFT | 0.994900 | 0.989000 | 0.985500 | 0.995000 | 1.000000 | 0.000000 | 6.4736 |
| ADVERSARIAL_GATE_CONFUSION | 0.994900 | 0.989000 | 0.985500 | 0.503000 | 1.000000 | 0.000000 | 6.4736 |
| EXTERNAL_TEST_REQUIRED | 0.994900 | 0.989000 | 0.985500 | 0.995000 | 1.000000 | 0.000000 | 6.4736 |
| INDISTINGUISHABLE_SUPPORT | 0.994900 | 0.989000 | 0.985500 | 0.995000 | 1.000000 | 0.000000 | 6.4736 |
| NOISY_CONTEXT | 0.994900 | 0.989000 | 0.985500 | 0.503000 | 1.000000 | 0.000000 | 6.4736 |
| HIDDEN_BUDGET_CONTEXT | 0.994900 | 0.989000 | 0.985500 | 0.012250 | 1.000000 | 0.000000 | 6.4736 |

Additional gate metrics:

```text
min_seed_exact = 0.992500
mixed_regression_vs_D62 = -0.002550
saturated_regression_vs_D62 = -0.004900
rust_path_invoked = true
fallback_rows = 0
failed_jobs = []
```

## Controls

MIXED_EVAL exact:

```text
RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER = 0.994900
SHUFFLED_SCORE_VECTOR_CONTROL         = 0.984800
RANDOM_DIAGNOSTIC_CONTROL             = 0.738250
DIAGNOSTIC_ABLATION_CONTROL           = 0.802250
```

The shuffled score-vector control is close but still worse. Random and ablated
diagnostics drop much harder.

## Proxy Audit

The Rust estimator input did not include the clean D63 proxy flags:

```text
uses_clean_d63_proxy_flags_as_rust_inputs = false
violating_input_features = []
```

Forbidden direct inputs included:

```text
support_budget_pressure_norm
counterfactual_pressure_norm
adversarial_pressure_norm
internal_unresolvable_indicator
external_channel_available
support_regime
track
mixed_source_track
row_id
seed
```

## Calibration Note

The controller-level result is positive, but not every binary diagnostic label
is calibrated equally well. Strong diagnostic matches:

```text
margin_low = 1.000000
support_independence_low = 1.000000
collision_pressure = 1.000000
counterfactual_pressure = 0.993875
adversarial_pressure = 0.993719
```

Weak audit-label matches:

```text
entropy_high = 0.482656
external_test_need = 0.243063
internal_unresolvable = 0.455563
support_effort_pressure = 0.500000
```

This means D64 should be read as a positive controller/usefulness result for the
Rust sparse IPF diagnostic layer, not as proof that every internal diagnostic
bit is individually well calibrated.

## Validation

```text
python -m py_compile scripts/probes/run_d64_rust_sparse_ipf_diagnostic_layer_prototype.py
python -m py_compile scripts/probes/run_d64_rust_sparse_ipf_diagnostic_layer_prototype_check.py
python scripts/probes/run_d64_rust_sparse_ipf_diagnostic_layer_prototype_check.py --check-only --out target/pilot_wave/d64_rust_sparse_ipf_diagnostic_layer_prototype/smoke
git diff --check
```

All passed.

## Boundary

D64 only tests a Rust sparse IPF diagnostic layer for controlled symbolic joint
formula discovery. It does not prove full VRAXION brain, raw visual Raven
reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture
superiority, or production readiness.
