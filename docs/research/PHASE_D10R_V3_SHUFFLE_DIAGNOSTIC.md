# Phase D10r-v3 Shuffle Diagnostic

Date: 2026-04-30

## Question

D10r-v2 stopped the release-ready autopilot with
`D10R_V2_PROJECTION_READOUT_BLOCKED`. The immediate question was whether this
was a simple bug in the `state_shuffle` control or a real projection/readout
sensitivity.

## Finding

The blocker has two layers:

1. The original `state_shuffle` control was too aggressive for paired
   baseline-vs-candidate evaluation because it applied independent state
   permutations per checkpoint. That can introduce random readout luck that is
   not a fair paired comparison.
2. After adding a fair `state_shuffle_shared` variant that applies the same
   permutation across baseline and candidates, the beta.8 checkpoint still has
   at least one shuffle repeat that approaches or beats the real MO score on
   some eval seeds. This means the remaining blocker is not only the independent
   permutation bug; the projection/readout is genuinely sensitive to state
   reindexing.

## Evidence

Run root:

```text
output/phase_d10r_v3_shuffle_diagnostic_20260430/shared_probe
```

Setup:

- `eval_len=1000`
- eval seeds `970001..970008`
- controls: `state_shuffle_shared`, `random_label`
- `control_repeats=4`
- `max_charge=7`

Beta.8 summary:

```text
real_mo_delta_mean      +0.014526
worst_selectivity_mean  -0.010946
failed_controls         state_shuffle_shared_02
```

The failing shared shuffle was not uniformly good. It was mostly noisy, but it
had several readout-luck spikes:

```text
seed 970002: shared02_mo +0.030618
seed 970004: shared02_mo +0.088535
seed 970008: shared02_mo +0.018535
```

The spike is driven by the same metrics the real result uses, especially
`smooth` and `unigram`, not by an isolated echo artifact.

## Code Changes

`tools/_scratch/d10r_hardened_eval.py` now supports:

- `state_shuffle_shared`: fair paired shuffle; one state permutation is shared
  across all checkpoint outputs for the same eval seed/control repeat.
- `state_shuffle_projection_consistent`: diagnostic no-op sanity; state rows and
  projection rows are permuted consistently, so scores should match the real
  path.

`state_shuffle_projection_consistent` is treated as a diagnostic control, not a
trust-gate control.

## Verdict

`D10R_V3_READOUT_SENSITIVITY_CONFIRMED`

The old control had a fairness issue, but fixing the paired permutation does not
fully clear the blocker. The evaluator remains too sensitive to projection/state
alignment. D10s, H512, and H8192 promotion-style runs should remain blocked until
the readout is redesigned or the score is explicitly control-adjusted.

## Next Step

Implement D10r-v4 readout hardening:

- Use `trusted_mo = real_mo - max(control_mo)` as the primary promotion score.
- Add a random-projection null distribution and require the real score to beat
  its upper confidence bound.
- Add entropy/calibration diagnostics so overconfident random readouts cannot
  pass as semantic signal.
- Keep `state_shuffle_shared` as the adversarial control and keep
  `state_shuffle_projection_consistent` as a no-op regression test.

