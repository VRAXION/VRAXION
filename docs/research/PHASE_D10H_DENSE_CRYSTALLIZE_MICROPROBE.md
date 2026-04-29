# Phase D10h Dense Crystallize Microprobe

Date: 2026-04-29

Verdict: `D10H_DENSE_START_TOO_CLIFFY`

## Summary

D10h tested a short dense -> crystallize path around the beta.8 H=384
generalist checkpoint. The question was whether adding a large random edge
fill and then pruning buckets could preserve or improve the current
multi-objective EQ-bar behavior.

The short answer is no for global random fill: both 10% and 25% directed
edge density immediately collapsed the beta.8 behavior before any useful
crystallize loop could begin.

## Run

Command shape:

```text
python tools/_scratch/d10h_dense_crystallize_probe.py
  --device cuda
  --eval-len 1000
  --eval-seeds 981001,981002,981003,981004
  --noise-seeds 981001..981008
  --densities 0.10,0.25
  --rounds 10
  --buckets 64
```

Output root:

```text
output/phase_d10h_dense_crystallize_probe_20260429/main
```

Generated output remains uncommitted.

## Results

Noise floor for the beta.8 reference was small enough for this as a scout:

| metric | std | sem95 |
|---|---:|---:|
| smooth | 0.000152 | 0.000105 |
| accuracy | 0.000354 | 0.000245 |
| echo | 0.000000 | 0.000000 |
| unigram | 0.000337 | 0.000233 |

Dense-start sanity:

| density | added edges | total edges | class | smooth delta | accuracy delta | echo delta | unigram delta |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.10 | 4,567 | 14,707 | `CLIFF` | -0.11338 | -0.04300 | +0.00500 | -0.30283 |
| 0.25 | 26,628 | 36,768 | `CLIFF` | -0.11338 | -0.04300 | +0.00500 | -0.30283 |

Because both dense starts crossed the cliff gate, the run stopped before
ranked pruning. This follows the pre-registered stop rule rather than spending
the full 45-75 minute budget on a broken start state.

## Interpretation

The beta.8 circuit is not tolerant to global random overfill. Adding thousands
of arbitrary edges swamps the output distribution enough that the smooth
EQ-bar, accuracy, echo safety, and unigram behavior all fail immediately.

This does not falsify crystallization as a general idea. It falsifies the
specific short path:

```text
beta.8 + global random dense fill -> prune back to a better circuit
```

The next crystallize test should be more local:

```text
beta.8 + local overfill near causal edge/threshold zones
```

or should start from a model trained under dense/overfilled conditions from the
beginning, rather than injecting many random edges into an already tuned sparse
checkpoint.

## Progress Map

```text
GLOBAL AI PLAN MAP

[1] beta.8 generalist found
    DONE

[2] mechanism explained
    DONE: edge + threshold co-adaptation

[3] seed replication
    RUNNING: D10b CPU main

[3.5] GPU evaluator
    DONE: useful for batched scout

[4] dense -> crystallize/prune
    D10h short global-fill probe: DONE
    result: DENSE_START_TOO_CLIFFY
        |
        |-- next crystallize test
        |      local overfill around causal zones, not global random fill
        |
        '-- do not spend long runs on global random dense fill

[5] H512/H1024 scaling
    still blocked until D10b or a safer crystallize path gives repeatable signal
```
