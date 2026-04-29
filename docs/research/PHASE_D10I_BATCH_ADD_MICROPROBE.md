# Phase D10i Batch Add Microprobe

Date: 2026-04-29

Verdict: `D10I_ADD_TOO_CLIFFY`

## Summary

D10i tested whether the beta.8 H=384 generalist checkpoint can be improved by
adding small edge batches instead of using global dense fill. This was the
direct follow-up to D10h, where global dense fill plus pruning produced an
echo-cliffy attractor.

The short scout found many positive-looking candidates at `eval_len=128`, but
the top candidates did not survive the longer `eval_len=1000` micro-confirm.
The apparent gains were mostly unigram/MO improvements paired with small
smooth/accuracy regressions, not valid generalist improvements.

## Run

Main scout:

```text
device: cuda
eval_len: 128
eval_seeds: 982001,982002
policies: global_random, local_existing, motif_closure
batch_sizes: 1,4,16,64
proposals_per_arm: 64
total proposals: 768
```

Main scout classes:

```text
ADD_STRONG: 63
ADD_WEAK: 5
ADD_NEEDS_THRESHOLD_POLISH: 13
```

Best scout arm:

```text
local_existing_b16
smooth_delta:   +0.001923
accuracy_delta: +0.000000
echo_delta:     +0.000000
unigram_delta:  +0.010554
mo_score:       +0.017753
```

Micro-confirm:

```text
device: cuda
eval_len: 1000
eval_seeds: 982001,982002,982003,982004
confirmed candidates: top 12
```

Confirm classes:

```text
ADD_REJECT: 12
```

Best confirm row:

```text
source: global_random_b4
smooth_delta:   -0.001572
accuracy_delta: -0.001000
echo_delta:     +0.000000
unigram_delta:  +0.008074
mo_score:       +0.010039
```

## Interpretation

Small edge-add has a real signal, but not yet a valid generalist signal. The
short eval found many candidates that improved the composite score, but longer
eval showed the improvement was mostly unigram-driven and came with smooth and
accuracy regression.

This is not the same failure mode as D10h:

```text
D10h dense fill:
  too much edge mass -> echo-cliff attractor

D10i small add:
  local edge grafts -> unigram/MO lure, but smooth/accuracy do not hold
```

The next useful version is not bigger edge batches. Batch size 64 was already
cliffy in the scout. The next version should combine small edge-add with
immediate threshold polish, or change the acceptance score so unigram cannot
dominate a smooth/accuracy regression.

## Progress Map

```text
GLOBAL AI PLAN MAP

[1] beta.8 generalist
    DONE

[2] causal mechanism
    DONE: edge + threshold co-adaptation

[3] seed replication
    RUNNING: D10b CPU main

[3.5] GPU evaluator
    DONE: batch scout works

[4] dense fill/prune
    DONE: global dense too cliffy

[4.1] batch-add graft
    D10i DONE
    result: short signal, confirm rejects all top candidates
        |
        |-- next option A
        |      edge-add + threshold polish on the same proposal
        |
        '-- next option B
               stricter smooth/accuracy-first acceptance

[5] H512/H1024
    still blocked until repeatable H384 signal beyond beta.8
```
