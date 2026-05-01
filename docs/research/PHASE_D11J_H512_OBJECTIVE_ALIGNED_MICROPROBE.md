# Phase D11j: H512 Objective-Aligned Microprobe

Date: 2026-05-01

## Summary

D11j tested the direct fix implied by D11i:

```text
If smooth/diversity-only search moves H512 away from robust paired eval,
add an opt-in accuracy term to the local fitness.
```

Code change:

```text
evolve_mutual_inhibition --accuracy-guard-lambda <float>
```

Default is `0.0`, so existing runs are unchanged unless the flag is explicitly
enabled.

Verdict:

```text
D11J_ACCURACY_GUARD_STABILIZES_BUT_DOES_NOT_IMPROVE
```

## Code Change

The local candidate objective now supports:

```text
fitness =
  smooth * (1.0 + 0.1 * alive_frac)
  + diversity_guard_lambda * diversity
  + accuracy_guard_lambda * accuracy
```

This is deliberately opt-in because older D9-D11 runs must remain reproducible.

## Run Shape

Start scaffold:

```text
H=512
seed=2042
FineWeb-Edu 1M
embedding_anchored_highways=4
diversity_guard_lambda=0.2
accuracy_guard_lambda=5.0
steps=500
```

Output:

```text
output/phase_d11j_h512_objective_aligned_20260501/acc5_500/final.ckpt
```

Raw result:

- final accuracy: `2.70%`
- peak accuracy: `2.70%`
- accept rate: `35.2%`
- edges: `10454`
- quick diversity: `1/4`
- average charge diff: `45.6%`

The raw score improved, but quick diversity showed a constant-output shortcut.

## Paired Confirm

D11j used the same D11i H512 init baseline:

```text
output/phase_d11i_h512_fineweb_confirm_20260501/init_baseline/final.ckpt
```

Confirm command shape:

```text
d10r_hardened_eval.py
  --eval-len 1000
  --eval-seeds 970001..970008
  --artifact-controls state_shuffle_projection_consistent
```

Result:

- verdict: `D10R_V5_POSITIVE_CONTROL_FAIL`
- real MO delta mean: `~0.0`
- real MO CI: overlaps zero
- accuracy delta: `0.0`
- unigram delta: `0.0`
- state-shuffle projection-consistent margin: `0.0`

Interpretation:

The accuracy guard prevented the harmful D11h drift, but it did not create a
paired-eval improvement. The checkpoint is effectively tied with the same-seed
H512 init baseline under robust eval.

## Chain Diagnosis

The D11j checkpoint is not dead:

- input `0..31` reaches output: `32/32`
- total input-to-output impact: `4595`
- output charge diff: `21.5%`
- unique predictions: `3/8`
- context-dependent predictions: `1/4`

But the readout is still dominated by class `0`, so it is not a release-quality
generalist candidate.

## Interpretation

D11j tells us the next blocker is not H512 activation. That was fixed.

The blocker is objective shape:

```text
accuracy guard alone is too conservative;
smooth/diversity alone is too loose;
H512 needs a richer local objective or a different proposal policy.
```

The current H512 path is useful research infrastructure, but not yet a release
candidate path.

## Next Gate

Recommended next step:

```text
D11k objective redesign or return to H384 release packaging
```

Two viable branches:

- `release branch`: package the proven H384 top_01 checkpoint as a research
  release candidate with explicit H512 blocker notes.
- `research branch`: implement a real multi-metric local objective using the
  same components as D10r: smooth, accuracy, echo, unigram, and artifact-safe
  paired eval seeds.

Do not run H1024/H8192 until H512 has a paired-eval positive candidate.

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 research checkpoint
    DONE

[2] D10 artifact/state hardening
    DONE

[3] H512 activation/readout bottleneck
    DONE

[4] H512 FineWeb searchability
    DONE

[5] H512 paired confirm
    FAIL: D11i

[6] H512 objective alignment
    CURRENT RESULT: WEAK
    D11j prevents negative drift but does not beat init baseline

[7A] H384 research release package
    AVAILABLE NOW

[7B] richer H512 multi-metric objective
    NEXT RESEARCH PATH

[8] release-ready AI candidate
    H384 package can move forward as research release;
    high-H release remains blocked
```
