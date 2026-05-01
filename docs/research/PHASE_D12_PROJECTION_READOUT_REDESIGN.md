# Phase D12: Projection/Readout Redesign Microprobe

Date: 2026-05-01

## Summary

D11y showed that structured high-H starts can produce raw signal, but semantic
controls are stronger. D12 tested the smallest projection/readout redesign that
could plausibly reduce those false positives.

Verdict:

```text
D12_CLEANER_PROJECTION_FLAT_SIGNAL
```

## Code Change

`tools/_scratch/d10p_semantic_projection_hardening.py` now supports three extra
projection/start arms:

- `copy_zero_threshold_mid`
- `block_local_threshold_mid`
- `signed_threshold_mid`

These keep the useful `threshold_mid` high-H structure from D11y but change the
projection/readout pattern:

- `copy_zero_threshold_mid`: copy only the source projection rows, leave the
  extra high-H rows zeroed.
- `block_local_threshold_mid`: copy source projection rows into separated
  blocks with gaps instead of dense tiling.
- `signed_threshold_mid`: use signed tiled projection to break exact row-copy
  symmetry.

All are scratch/prototype arms; no release API is changed.

## Run Shape

```text
H = 8192
active_edges = 400000
eval_len = 128
eval_seeds = 990001,990002
proposals_per_arm = 8
controls = random_label, random_bigram, unigram_decoy, projection_shuffle
```

## Results

| arm | verdict | real safe | max control safe | adjusted |
|---|---|---:|---:|---:|
| copy_zero_threshold_mid | `NO_SIGNAL` | 0.125 | 0.125 | 0.000 |
| block_local_threshold_mid | `NO_SIGNAL` | 0.000 | 0.125 | -0.125 |
| signed_threshold_mid | `NO_SIGNAL` | 0.000 | 0.125 | -0.125 |

## Interpretation

D12 reduced the semantic leak, but also removed the useful real signal.

This narrows the blocker:

```text
tiled projection creates raw high-H signal but leaks to controls;
cleaner projection variants reduce false positives but go flat.
```

The next high-H approach must not only change projection layout. It needs a
co-designed readout/training objective where the real signal is learned or
selected against the controls, not merely copied or tiled from beta.8.

## Release-Ready Meaning

High-H remains blocked.

The deterministic release path today is still:

```text
H384 top_01 research release package
```

High-H research should continue only after a stronger D12b design:

- train/select projection rows under real-vs-control objective;
- freeze a validation slice for controls;
- stop using tiled projection as the default high-H readout;
- require control-adjusted safe rate > 0 at `eval_len=1000`.

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
    DONE: large H is runnable

[5] structured high-H semantic scout
    DONE: raw signal but semantic leak

[6] projection/readout redesign
    CURRENT RESULT: CLEANER BUT FLAT
    false positives reduced, real signal also disappears

[7] release-ready candidate
    H384 research package is the only deterministic path today
```
