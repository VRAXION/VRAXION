# Phase D11h: FineWeb H512 Variant Sweep

Date: 2026-05-01

## Summary

D11h tested whether the weak D11g FineWeb-Edu H512 signal could be improved by
changing only the H512 bootstrap/search knobs:

- `embedding_anchored_highways`
- `diversity_guard_lambda`

This was still a scaling pilot, not a release checkpoint run.

Goal:

```text
Can H512 on FineWeb-Edu tiny exceed the D11g 0.70% short signal
without collapsing into a constant or artifact-like readout?
```

## Shared Setup

Corpus:

```text
output/phase_d11g_fineweb_edu_tiny_20260501/fineweb_edu_1m.txt
```

Packed table:

```text
output/block_c_bytepair_champion/packed.bin
```

Common run shape:

```text
--H 512
--jackpot 9
--ticks 6
--accept-policy strict
--operator-policy baseline
--accept-ties false
```

## 500-Step Variant Sweep

| Arm | Final | Peak | Accept | Quick diversity | Avg diff | Verdict |
|---|---:|---:|---:|---:|---:|---|
| anchor2 lambda0.05 | 0.70% | 0.70% | 28.8% | 3/4 | 12.2% | ties D11g |
| anchor2 lambda0.20 | 0.40% | 0.40% | 20.6% | 3/4 | 14.8% | weak |
| anchor4 lambda0.10 | 0.50% | 0.50% | 29.6% | 1/4 | 49.2% | reject: collapse |
| anchor4 lambda0.20 | 1.60% | 1.60% | 33.4% | 2/4 | 44.3% | promote to follow-up |

The best 500-step arm was:

```text
embedding_anchored_highways = 4
diversity_guard_lambda = 0.2
```

Chain diagnosis for this arm:

- input `0..31` reaches output: `32/32`
- total input-to-output impact: `4498`
- output charge diff: `20.7%`
- unique predictions: `3/8`
- context effect: present

Verdict: `D11H_FINEWEB_H512_SHORT_SIGNAL`

## 2k Follow-Up

Arm:

```text
anchor4 lambda0.20
```

Result:

- step 2000 accuracy: `2.40%`
- peak accuracy: `2.40%`
- final accuracy: `2.10%`
- accept rate: `29.7%`
- edges: `10717`
- quick diversity: `4/4`
- average charge diff: `38.4%`

Chain diagnosis:

- input `0..31` reaches output: `32/32`
- total input-to-output impact: `4391`
- output charge diff: `17.7%`
- unique predictions: `6/8`
- context-dependent predictions: `2/4`

Verdict: `D11H_FINEWEB_H512_2K_SIGNAL_CONFIRMED`

## 10k Bounded Follow-Up

Arm:

```text
anchor4 lambda0.20
```

Result:

- step 10000 accuracy: `3.90%`
- peak accuracy: `3.90%`
- final accuracy: `1.60%`
- accept rate: `24.09%`
- edges: `11636`
- quick diversity: `4/4`
- average charge diff: `39.7%`

Chain diagnosis on the final checkpoint:

- input `0..31` reaches output: `32/32`
- total input-to-output impact: `3902`
- output charge diff: `17.5%`
- unique predictions: `7/8`
- context-dependent predictions: `3/4`

Verdict: `D11H_FINEWEB_H512_SCALING_SIGNAL_NEEDS_CONFIRM`

The 10k run proves a stronger H512 FineWeb state is reachable, but the
progress-sample score and final-sample score disagree. Both scores are from the
same end-of-run network state, so this should be treated as eval-sample variance
until a paired multi-seed confirm proves otherwise.

## Interpretation

D11h materially improves the H512 scaling picture:

```text
D11g best FineWeb short signal: 0.70%
D11h best 500-step signal:      1.60%
D11h best 2k signal:            2.40%
D11h best 10k observed peak:    3.90%
```

This is not a dead-H512 or Alice-only artifact. FineWeb-Edu H512 now has:

- real input-to-output reach;
- recurrent context dependence;
- diverse predictions;
- accepted strict mutations;
- a best observed state more than 5x above the D11g 0.70% short signal.

The remaining blocker is stability under paired evaluation, not basic H512
viability.

## Next Gate

Recommended next step:

```text
D11i H512 FineWeb paired confirm
```

Required run shape:

- create a same-seed H512 anchor4/lambda0.20 init checkpoint as baseline;
- compare the D11h 10k checkpoint against that baseline on paired eval seeds;
- include artifact controls where the H512 evaluator supports them;
- confirm first at short budget, then at `eval_len=4000` if the short gate passes.

Do not promote the 10k checkpoint directly. It is useful evidence, but the
`3.90%` versus `1.60%` split means it must be checked with paired seeds before
it can become a scaling proof.

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 research checkpoint
    DONE

[2] D10 artifact/state hardening
    DONE

[3] H512 dead baseline diagnosis
    DONE

[4] H512 embedding-anchored bootstrap
    DONE: D11e

[5] FineWeb-Edu corpus ladder
    DONE: D11g weak signal

[6] H512 FineWeb variant sweep
    CURRENT RESULT: PASS-NEEDS-CONFIRM
    D11h finds a strong H512 signal, peak sample 3.90%, final sample 1.60%

[7] H512 paired confirm
    NEXT: D11i

[8] H512 long confirm / H1024 decision
    BLOCKED until D11i confirms the saved D11h checkpoint against an H512 init baseline

[9] release-ready AI candidate package
    BLOCKED until H512 confirm or independent H384 reproduction passes
```
