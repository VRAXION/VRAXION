# Phase D10r-v5 Artifact Null Gate

Date: 2026-04-30

## Purpose

D10r-v5 separates evaluator artifact/null controls from alternate H=384
baseline checkpoints. Alternate seeds are no longer treated as null negatives.
They are reported separately and cannot by themselves produce
`CONTROL_LEAK_FAIL`.

The artifact gate remains:

```text
trusted_mo = real_mo - max(artifact_control_mo)
```

## Code Changes

`tools/_scratch/d10r_hardened_eval.py` now supports:

- `--artifact-controls`
- `--alternate-baseline-checkpoints`
- `--alternate-baseline-mode report_only`
- top-level artifact verdicts:
  - `D10R_V5_ARTIFACT_GATE_PASS`
  - `D10R_V5_ARTIFACT_READOUT_BLOCKED`
  - `D10R_V5_POSITIVE_CONTROL_FAIL`
- alternate-baseline verdicts:
  - `D10R_V5_ALTERNATE_BASELINE_SIGNAL`
  - `D10R_V5_NO_ALTERNATE_BASELINE_SIGNAL`

The generated report now includes:

- artifact null margin table
- alternate baseline trusted-score table
- final release gate decision table

## Smoke

Run root:

```text
output/phase_d10r_v5_artifact_gate_20260430/smoke
```

Result:

```text
verdict                    D10R_V5_ARTIFACT_READOUT_BLOCKED
alternate_baseline_verdict D10R_V5_ALTERNATE_BASELINE_SIGNAL
```

This confirms the implementation split works: alternate seeds are no longer
classified as negative leak failures.

## Bounded Main

The originally planned full main was started, but runtime projected too high, so
it was stopped before report generation and is not used as evidence.

Bounded main run root:

```text
output/phase_d10r_v5_artifact_gate_20260430/bounded_main
```

Setup:

- `eval_len=1000`
- eval seeds `970001..970008`
- `control_repeats=4`
- artifact controls:
  - `random_label`
  - `random_projection_null`
  - `state_shuffle_shared`
  - `no_network_random_state`

Result:

```text
verdict                    D10R_V5_ARTIFACT_READOUT_BLOCKED
alternate_baseline_verdict D10R_V5_ALTERNATE_BASELINE_SIGNAL
artifact_gate_pass         false
D10s_unlocked              false
```

Beta.8:

```text
real_mo_delta_mean +0.014526
trusted_mo_mean    -0.030054
trusted_mo_ci_low  -0.042609
```

The strongest artifact blocker was `random_projection_null_03`:

```text
random_projection_null_03 margin_mean   -0.023773
random_projection_null_03 margin_ci_low -0.023830
```

Alternate baselines:

```text
seed_1042 trusted_mo_pass true
```

This is reported as an alternate-baseline signal, not as a null-control leak.

## Verdict

`D10R_V5_ARTIFACT_READOUT_BLOCKED`

The evaluator split is correct and useful, but beta.8 still does not beat the
artifact/null controls under worst-control scoring. D10s, H512, and H8192 remain
blocked.

## Next Step

D10r-v6 should focus specifically on the random projection null:

- measure whether `random_projection_null_03` wins through smooth, unigram, or
  overconfident softmax collapse
- test calibrated/centered projection scores before softmax
- add a projection-null ensemble bound as a first-class statistic
- only unlock D10s if beta.8 beats that projection-null bound

