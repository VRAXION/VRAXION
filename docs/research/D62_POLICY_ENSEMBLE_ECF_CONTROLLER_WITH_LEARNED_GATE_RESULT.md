# D62 Policy Ensemble ECF Controller With Learned Gate Result

## Decision

```text
decision = policy_ensemble_learned_gate_confirmed
verdict = D62_POLICY_ENSEMBLE_LEARNED_GATE_CONFIRMED
next = D63_RUST_SPARSE_ECF_CONTROLLER_COMPONENT_MIGRATION
best_arm = LEARNED_MULTI_POLICY_GATE
```

Artifact root:

```text
target/pilot_wave/d62_policy_ensemble_ecf_controller_with_learned_gate/smoke
```

## Run

```text
scale_mode = scale-lite
seeds = 11701,11702,11703,11704,11705
train_rows_per_seed = 800
test_rows_per_seed = 800
ood_rows_per_seed = 800
workers = auto
cpu_target = 50-75
heartbeat_sec = 20
```

The runner wrote row-generation, pack-build, track-clone, gate-training, per-track evaluation snapshots, per-track final metrics, and final reports. No black-box run was used.

## Learned Gate

```json
{
  "default_policy": "SATURATED_POLICY",
  "rules": [
    {
      "feature": "support_budget_pressure_norm",
      "policy": "HARD_BUDGET_POLICY",
      "threshold": 0.25
    },
    {
      "feature": "external_channel_available",
      "policy": "EXTERNAL_TEST_POLICY",
      "threshold": 0.25
    },
    {
      "feature": "internal_unresolvable_indicator",
      "policy": "ABSTAIN_POLICY",
      "threshold": 0.25
    },
    {
      "feature": "adversarial_pressure_norm",
      "policy": "ADVERSARIAL_REPAIR_POLICY",
      "threshold": 0.5
    },
    {
      "feature": "counterfactual_pressure_norm",
      "policy": "COUNTERFACTUAL_POLICY",
      "threshold": 0.5
    }
  ]
}
```

The gate uses observable runtime diagnostics only. It does not use truth labels, support-regime labels, track labels, row ids, true cells/operators, or expected answers.

## Key Metrics

`LEARNED_MULTI_POLICY_GATE`:

| track | exact | corr | adv | external | abstain | false_conf | support | counter_support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SATURATED_STABILITY | 0.999800 | 0.999000 | 1.000000 | 0.996000 | 1.000000 | 0.000000 | 8.828300 | 3.828300 |
| HARD_CAP8_LEARNING | 0.995600 | 0.988500 | 0.989500 | 0.012500 | 1.000000 | 0.000000 | 6.471500 | 1.471500 |
| MIXED_EVAL | 0.997450 | 0.993500 | 0.993750 | 0.504750 | 1.000000 | 0.000000 | 7.651700 | 2.651700 |
| OOD_CONTEXT_SHIFT | 0.995600 | 0.988500 | 0.989500 | 0.996000 | 1.000000 | 0.000000 | 6.471500 | 1.471500 |
| ADVERSARIAL_GATE_CONFUSION | 0.997450 | 0.993500 | 0.993750 | 0.504750 | 1.000000 | 0.000000 | 8.735750 | 3.735750 |
| EXTERNAL_TEST_REQUIRED | 0.999800 | 0.999000 | 1.000000 | 0.996000 | 1.000000 | 0.000000 | 8.828300 | 3.828300 |
| INDISTINGUISHABLE_SUPPORT | 0.999800 | 0.999000 | 1.000000 | 0.996000 | 1.000000 | 0.000000 | 8.828300 | 3.828300 |
| NOISY_CONTEXT | 0.997450 | 0.993500 | 0.993750 | 0.504750 | 1.000000 | 0.000000 | 7.665050 | 2.665050 |
| HIDDEN_BUDGET_CONTEXT | 0.995600 | 0.988500 | 0.989500 | 0.012500 | 1.000000 | 0.000000 | 6.471500 | 1.471500 |

Additional checks:

```text
hard_gain_vs_D58 = +0.390550
saturated_regression_vs_D59 = +0.000625
min_seed_exact = 0.994750
fallback_rows = 0
failed_jobs = []
```

## Interpretation

D62 confirms that the controller can route among more than two Rust sparse ECF action-policy modules. The result is stronger than the D61 two-policy gate because the learned gate handles:

```text
hard budget pressure
external-test-required rows
indistinguishable-support abstain rows
noisy context
hidden budget context without explicit budget flag
```

The important boundary remains:

```text
Rust sparse path = action-controller path
formula discovery solver = controlled symbolic stack
```

## Validation

```text
python -m py_compile scripts/probes/run_d62_policy_ensemble_ecf_controller_with_learned_gate.py
python -m py_compile scripts/probes/run_d62_policy_ensemble_ecf_controller_with_learned_gate_check.py
python scripts/probes/run_d62_policy_ensemble_ecf_controller_with_learned_gate_check.py --check-only --out target/pilot_wave/d62_policy_ensemble_ecf_controller_with_learned_gate/smoke
git diff --check
```

All validation commands passed.

## Boundary

D62 only tests learned policy-ensemble gating for a Rust sparse ECF action-controller in controlled symbolic joint formula discovery. It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
