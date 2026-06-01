# D61 Gated Rust Sparse Mutation Scale Confirm Result

## Decision

```text
decision = gated_rust_sparse_mutation_scale_confirmed
verdict = D61_GATED_RUST_SPARSE_MUTATION_SCALE_CONFIRMED
next = D62_POLICY_ENSEMBLE_ECF_CONTROLLER_WITH_LEARNED_GATE
best_arm = LEARNED_GATE_MUTATION_CONTROLLER
```

Artifact root:

```text
target/pilot_wave/d61_gated_rust_sparse_mutation_scale_confirm/smoke
```

## Run

```text
seeds = 11601,11602,11603,11604,11605,11606,11607,11608
train_rows_per_seed = 1000
test_rows_per_seed = 1000
ood_rows_per_seed = 1000
scale_mode = full
workers = auto
cpu_target = 50-75
```

The final runner writes row-generation progress, pack-build progress, track-clone progress, per-track full metrics, and final reports. Earlier full attempts exposed missing progress in track cloning and final aggregation; the runner was patched to add those writeouts before accepting the result.

## Best Fair Gate

Learned gate:

```json
{
  "feature": "support_budget_cap_norm",
  "threshold": 0.01,
  "policy_if_ge": "D60_HARD_POLICY_REPLAY",
  "policy_if_lt": "D59_REFERENCE"
}
```

This is an allowed runtime feature, not a truth label, track label, row id, support-regime label, or mixed-source label.

## Key Metrics

`LEARNED_GATE_MUTATION_CONTROLLER`:

| track | exact | corr | adv | external | abstain | false_conf | support | counter_support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SATURATED_STABILITY | 0.999175 | 0.997750 | 0.998125 | 0.995250 | 1.000000 | 0.000000 | 7.681025 | 2.681025 |
| HARD_CAP8_LEARNING | 0.993450 | 0.985750 | 0.981500 | 0.012250 | 1.000000 | 0.000000 | 6.481025 | 1.481025 |
| MIXED_EVAL | 0.996275 | 0.992125 | 0.989250 | 0.503750 | 1.000000 | 0.000000 | 7.081025 | 2.081025 |
| OOD_CONTEXT_SHIFT | 0.993450 | 0.985750 | 0.981500 | 0.995250 | 1.000000 | 0.000000 | 6.481025 | 1.481025 |
| ADVERSARIAL_GATE_CONFUSION | 0.996275 | 0.992125 | 0.989250 | 0.503750 | 1.000000 | 0.000000 | 7.081025 | 2.081025 |

Additional gate checks:

```text
hard_gain_vs_D58 = 0.388400
saturated_regression_vs_D59 = -0.000225
min_seed_exact = 0.991800
gate_accuracy = 1.000000
rust_path_invoked = true
fallback_rows = 0
failed_jobs = []
```

## Controls

The reference-only truth leak sentinel scored slightly higher in saturated exact, as expected, and is not fair evidence. Fair learned/handcoded gates matched the D60S replay without using forbidden labels.

The checker verified:

```text
random gate control worse
wrong gate control worse
gate ablation worse
random policy control worse
spike shuffle control worse
truth leak sentinel reference-only
fair arms with truth leak = []
fair arms using forbidden features = []
```

## Validation

```text
python -m py_compile scripts/probes/run_d61_gated_rust_sparse_mutation_scale_confirm.py
python -m py_compile scripts/probes/run_d61_gated_rust_sparse_mutation_scale_confirm_check.py
python scripts/probes/run_d61_gated_rust_sparse_mutation_scale_confirm_check.py --check-only --out target/pilot_wave/d61_gated_rust_sparse_mutation_scale_confirm/smoke
git diff --check
```

All validation commands passed.

## Boundary

D61 only scale-confirms gated Rust sparse ECF action-controller mutation in controlled symbolic joint formula discovery. It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
