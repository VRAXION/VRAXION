# Phase D11y: Structured High-H Semantic Scout

Date: 2026-05-01

## Summary

D11y followed the D11x GPU frontier with a structured high-H scout. The goal was
to test whether H8192/H16384 become useful when we stop using purely random
sparse perturbations and instead use beta8/motif/projection/threshold starts.

This was still a GPU scout, not a release candidate run.

Verdict:

```text
D11Y_HIGH_H_STRUCTURED_SIGNAL_SEMANTIC_BLOCKED
```

## Setup

Evaluator:

```text
tools/_scratch/d10p_semantic_projection_hardening.py
```

Controls enabled for every arm:

- `random_label`
- `random_bigram`
- `unigram_decoy`
- `projection_shuffle`

Pass rule:

```text
real safe rate must beat all controls by a meaningful margin.
```

## H8192 / 100k

Settings:

```text
H=8192
active_edges=100000
eval_len=128
eval_seeds=990001,990002
proposals_per_arm=8
```

| arm | verdict | real safe | max control safe | adjusted |
|---|---|---:|---:|---:|
| beta8_lifted_v2 | `SEMANTIC_FAIL` | 0.000 | 0.500 | -0.500 |
| motif_no_echo | `SEMANTIC_FAIL` | 0.750 | 0.500 | 0.250 |
| block_local_projection | `NO_SIGNAL` | 0.250 | 0.250 | 0.000 |
| frozen_beta8_rows | `SEMANTIC_FAIL` | 0.375 | 0.750 | -0.375 |
| threshold_mid | `SEMANTIC_FAIL` | 0.375 | 0.375 | 0.000 |

Interpretation:

```text
Raw signal exists, especially motif_no_echo,
but controls are too positive.
```

## H16384 / 100k

Settings:

```text
H=16384
active_edges=100000
eval_len=128
eval_seeds=990001,990002
proposals_per_arm=8
```

| arm | verdict | real safe | max control safe | adjusted |
|---|---|---:|---:|---:|
| beta8_lifted_v2 | `SEMANTIC_FAIL` | 0.375 | 0.750 | -0.375 |
| motif_no_echo | `SEMANTIC_FAIL` | 0.000 | 0.625 | -0.625 |
| block_local_projection | `SEMANTIC_FAIL` | 0.125 | 0.625 | -0.500 |
| frozen_beta8_rows | `SEMANTIC_FAIL` | 0.250 | 0.500 | -0.250 |
| threshold_mid | `SEMANTIC_FAIL` | 0.000 | 0.500 | -0.500 |

Interpretation:

```text
H16384/100k is worse than H8192/100k under semantic controls.
Projection-shuffle and decoy controls dominate the apparent signal.
```

## H8192 / 400k

Settings:

```text
H=8192
active_edges=400000
eval_len=128
eval_seeds=990001,990002
proposals_per_arm=8
```

| arm | verdict | real safe | max control safe | adjusted |
|---|---|---:|---:|---:|
| motif_no_echo | `NO_SIGNAL` | 0.000 | 0.125 | -0.125 |
| block_local_projection | `NO_SIGNAL` | 0.000 | 0.000 | 0.000 |
| threshold_mid | `WEAK_PASS` | 0.375 | 0.250 | 0.125 |

The only apparent survivor was `threshold_mid`, so it received a stronger
confirm.

## H8192 / 400k Threshold-Mid Confirm

Settings:

```text
H=8192
active_edges=400000
eval_len=1000
eval_seeds=990001,990002,990003,990004
proposals_per_arm=16
arm=threshold_mid
```

Result:

| arm | verdict | real safe | max control safe | adjusted |
|---|---|---:|---:|---:|
| threshold_mid | `SEMANTIC_FAIL` | 0.4375 | 0.6250 | -0.1875 |

The control failure came primarily from projection/semantic decoy behavior. The
real arm had signal, but the control signal was stronger.

## Interpretation

D11y gives a deterministic answer:

```text
Structured high-H starts do create raw positive candidates.
But the current projection/readout/eval setup creates equal or stronger
false positives under controls.
```

This is not a GPU capacity failure and not a proof that high-H is impossible.
It is a semantic trust failure.

The high-H path remains blocked until projection/readout controls are redesigned.

## Long-Horizon Meaning

The current release-ready state is unchanged:

- H384 research checkpoint remains the only confirmed release-package candidate.
- H512/H8192/H16384 are research infrastructure, not release candidates.
- Bigger H is runnable, but not trustworthy under current semantic controls.

Next useful high-H work:

```text
D12 projection/readout redesign:
  non-tiled projection,
  stronger no-label baselines,
  fixed validation slices,
  paired semantic controls,
  then rerun H8192/400k threshold_mid.
```

Do not spend more time on larger random H until this semantic blocker is fixed.

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 research checkpoint
    DONE

[2] H384 artifact/state hardening
    DONE

[3] H512 infrastructure
    DONE but objective mismatch

[4] high-H GPU frontier
    DONE: H8192-H524288 evaluable

[5] structured high-H semantic scout
    CURRENT RESULT: BLOCKED
    raw signal exists, controls are stronger

[6] next high-H gate
    projection/readout redesign, not larger H

[7] release-ready candidate
    H384 research package is the only deterministic path today
```
