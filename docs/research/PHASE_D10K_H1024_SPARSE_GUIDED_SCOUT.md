# Phase D10k H1024 Sparse Guided Scout

Date: 2026-04-30

Verdict: `D10K_BETA8_PATTERN_SCALES`

## Summary

D10k tested the independent GPU/high-H lane opened by D10j. The question was
not whether we have a deployable H1024 network, but whether a large sparse
space is searchable enough to justify a fair H512/H1024 science gate.

Result: H1024 with 25k active edges produced repeatable safe-positive scout
signals after a longer eval_len=1000 confirm. The best beta8-lifted candidate
had a stronger multi-objective score than the best random-sparse candidate,
while the random-label control produced zero safe-positive candidates.

## Setup

```text
H: 1024
active_edges: 25000
usage: 2.39%
eval_len: 1000
eval_seeds: 984001,984002,984003,984004
proposals_per_arm: 32
edge_swaps: 16
threshold_mutations: 16
```

Generated output:

```text
output/phase_d10k_h1024_sparse_guided_scout_20260430/confirm_1k/
```

## Confirm Results

| arm | safe | unsafe | echo trap | no signal | best MO | best smooth | best accuracy | best echo | best unigram |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_sparse | 16 | 4 | 3 | 10 | 0.006179 | 0.001038 | 0.001000 | 0.001000 | 0.003261 |
| beta8_lifted | 12 | 1 | 7 | 13 | 0.025020 | 0.005668 | 0.001750 | 0.001177 | 0.012514 |
| motif_guided | 0 | 0 | 26 | 7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Interpretation:

```text
H1024 sparse space is not flat.
Random sparse already has many small safe-positive local moves.
Beta8-lifted has fewer safe moves, but the best move is much stronger.
The current motif-guided policy is too echo-cliffy and should not be reused
unchanged.
```

## Adversarial Checks

```text
NOOP_ZERO_DELTA: passed in smoke
RANDOM_LABEL_CONTROL: 0/32 safe-positive
ECHO_TRAP_REJECT: motif-guided rejected as trap-heavy
GPU_ONLY_NO_PROMOTION: enforced by interpretation
LOW_USAGE_NOT_DENSE: H1024/25k = 2.39% active usage
```

The random-label control produced:

```text
safe: 0
unsafe: 0
echo_trap: 28
no_signal: 4
```

This means the D10k positive signal is not trivially reproduced under shuffled
semantic targets. It is still GPU scout evidence only, because scatter-based
GPU propagation is not a promotion gate and there is no fair trained H1024
checkpoint yet.

## Long-Horizon Meaning

D10j proved that H1024 sparse evaluation is feasible. D10k adds that H1024
sparse search has a measurable signal and that beta.8 structure may partially
scale when lifted into the larger space.

This changes the next gate:

```text
Before D10j/D10k:
  H512/H1024 was blocked by infra uncertainty.

After D10j/D10k:
  H512/H1024 is blocked by fair checkpoint/projection quality, not by GPU eval.
```

## Progress Map

```text
GLOBAL AI PLAN MAP

[1] H384 beta.8 generalist
    DONE

[2] causal mechanism
    DONE: edge + threshold co-adaptation

[3] H384 seed replication
    RUNNING: D10b CPU main

[4] dense/add local probes
    DONE:
      D10h dense/prune -> echo/cliff
      D10i edge-add -> short signal, confirm reject

[5] sparse high-H GPU
    DONE:
      D10j -> H1024/25k evaluable
      D10k -> H1024/25k searchable scout signal

[6] fair H512/H1024 science gate
    NEXT:
      build or locate fair high-H checkpoint/projection
      rerun multi-objective scout
      confirm winners with CPU/dense-reference before any promotion
```

## Next Step

Run a fair high-H gate instead of another synthetic-only scout:

```text
D10l Fair H512/H1024 Checkpoint/Projection Gate

1. create or locate H512 checkpoint with matching projection
2. run sparse GPU scout at low active usage
3. compare random_sparse vs beta8_lifted vs trained_highH_start
4. confirm any candidate with dense-reference/CPU-compatible evaluator
```

No release claim should be made from D10k alone.
