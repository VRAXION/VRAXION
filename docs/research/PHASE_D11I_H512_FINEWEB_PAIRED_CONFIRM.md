# Phase D11i: H512 FineWeb Paired Confirm

Date: 2026-05-01

## Summary

D11i checked whether the D11h H512 FineWeb 10k checkpoint survives paired
evaluation against a same-seed H512 init baseline.

Input checkpoint:

```text
output/phase_d11h_fineweb_variant_sweep_20260501/anchor4_lambda02_10k/final.ckpt
```

Same-seed H512 init baseline:

```text
output/phase_d11i_h512_fineweb_confirm_20260501/init_baseline/final.ckpt
```

Verdict:

```text
D11I_H512_FINEWEB_CONFIRM_FAIL_OBJECTIVE_MISMATCH
```

## Baseline Sanity

The D11i init baseline was generated with the same H512 anchor recipe:

```text
--H 512
--embedding-anchored-highways 4
--diversity-guard-lambda 0.2
--seed 2042
--steps 0
```

Init result:

- final accuracy: `0.40%`
- peak accuracy: `0.40%`
- edges: `10334`
- quick diversity: `1/4`
- average charge diff: `56.9%`

This is a valid weak baseline: same scaffold, no search.

## Paired Smoke

Command shape:

```text
d10r_hardened_eval.py
  --baseline D11i init baseline
  --positive D11h anchor4/lambda0.20 10k checkpoint
  --corpus FineWeb-Edu 1M
  --eval-len 256
  --eval-seeds 970001..970004
  --artifact-controls random_projection_null,state_shuffle_projection_consistent,no_network_random_state
```

Result:

- verdict: `D10R_V5_POSITIVE_CONTROL_FAIL`
- real MO delta mean: `-0.05084`
- trusted MO mean: `-0.06080`
- failed control families: `random_projection_null`, `no_network_random_state`

Interpretation:

The checkpoint did not pass artifact-selective evaluation at smoke budget.

## Real-Only Confirm

To make sure the failure was not caused by an artifact-control bug, D11i ran a
minimal paired confirm with only the projection-consistent invariant control.

Command shape:

```text
d10r_hardened_eval.py
  --eval-len 1000
  --eval-seeds 970001..970008
  --artifact-controls state_shuffle_projection_consistent
```

Result:

- verdict: `D10R_V5_POSITIVE_CONTROL_FAIL`
- real MO delta mean: `-0.05417`
- real MO CI low: `-0.05422`
- state-shuffle projection-consistent margin: exactly `0.0`

Primary metric pattern on real paired seeds:

| Metric | Init baseline | D11h 10k checkpoint | Delta |
|---|---:|---:|---:|
| accuracy | 0.80% | 0.00% | -0.80% |
| smooth | ~0.0351 | ~0.0050 | ~-0.0301 |
| echo | ~0.0083 | 0.00% | ~-0.0083 |
| unigram | ~0.0991 | ~0.0101 | ~-0.0890 |
| MO | 0.0 | ~-0.0542 | negative |

This confirms the blocker is not the control wrapper. The D11h checkpoint is
worse than the same-seed init baseline under the robust paired evaluator.

## Interpretation

D11h was still valuable because it proved H512 can be activated and searched on
FineWeb-Edu:

- input reaches output;
- recurrent context exists;
- strict mutation accept rate is nonzero;
- the network does not remain dead.

But D11i shows the current H512 search objective is not aligned with the release
gate. The `3.90%` D11h peak was a local/progress-sample signal, not a stable
paired-eval improvement.

The failure mode is:

```text
H512 infrastructure works,
but smooth/diversity-only search can move away from robust accuracy/unigram.
```

## Next Gate

Do not run H1024/H8192 yet.

Recommended next step:

```text
D11j objective-aligned H512 microprobe
```

Required change:

- add an opt-in accuracy or multi-metric guard to `evolve_mutual_inhibition`;
- keep the H512 anchor4/lambda0.20 scaffold;
- run a short FineWeb-Edu smoke;
- accept only if paired D10r eval improves over the same-seed init baseline.

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 research checkpoint
    DONE

[2] D10 artifact/state hardening
    DONE

[3] H512 activation/readout bottleneck
    DONE: fixed by D11e

[4] H512 FineWeb searchability
    DONE: D11g/D11h prove H512 is alive and searchable

[5] H512 paired confirm
    CURRENT RESULT: FAIL
    D11i shows the D11h checkpoint does not beat its same-seed H512 init baseline

[6] Objective-aligned H512 search
    NEXT: D11j

[7] H512 confirm / H1024 decision
    BLOCKED until D11j produces a paired-eval positive candidate

[8] release-ready AI candidate package
    BLOCKED until H512 confirm or independent H384 reproduction passes
```
