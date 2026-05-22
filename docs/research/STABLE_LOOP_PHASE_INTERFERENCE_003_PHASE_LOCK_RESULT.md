# STABLE_LOOP_PHASE_INTERFERENCE_003_PHASE_LOCK Result

## Status

Implemented and ran the phase-lock discriminator after `DYNAMIC_CANCEL` showed that additive dynamic message passing was still sufficient.

This is the first run in this branch that gives a clean Prismion-positive result.

## Why This Run Exists

`STABLE_LOOP_PHASE_INTERFERENCE_002_DYNAMIC_CANCEL` was a valid non-monotonic phase task, but it still used an additive local vector update:

```text
state_next = decay * state + incoming
```

Canonical dynamic message passing solved that exactly. This was a measurement-orientation problem for testing Prismion specifically.

`PHASE_LOCK` changes the primitive:

```text
incoming phase is multiplied by the destination cell's local phase gate
target phase depends on local phase transport / rotation along the path
```

This makes the phase/interference primitive the core operation rather than a decorative representation.

## Run

```text
out=target/pilot_wave/stable_loop_phase_interference_003_phase_lock/k4_3seed
phase_classes=4
seeds=2026,2027,2028
train_examples=4096
eval_examples=4096
epochs=10
width=32
jobs=4
device=cuda
completed_jobs=21
```

## Verdict

```text
PHASE_LOCK_TASK_VALID
PRISMION_PHASE_LOCK_POSITIVE
```

## Summary Table

| Arm | PhaseLockAcc | PhaseBucket | LongPath | PairAcc | WallLeak | Params |
|---|---:|---:|---:|---:|---:|---:|
| `ORACLE_PHASE_LOCK_S` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0` |
| `HARD_WALL_PRISMION_PHASE_LOCK_LOOP` | `0.974` | `0.983` | `0.961` | `0.952` | `0.000` | `0` |
| `LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK` | `0.400` | `0.263` | `0.456` | `0.219` | `0.000` | `0` |
| `HARD_WALL_ABC_PHASE_LOCK_LOOP` | `0.218` | `0.067` | `0.272` | `0.096` | `0.000` | `2402` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK` | `0.208` | `0.116` | `0.234` | `0.083` | `0.000` | `353058` |
| `SUMMARY_DIRECT_HEAD` | `0.206` | `0.258` | `0.188` | `0.045` | `0.000` | `13253` |
| `TARGET_MARKER_ONLY` | `0.210` | `0.264` | `0.197` | `0.054` | `0.000` | `5` |

## Matched Seed Deltas

```text
Prismion - ABC:
  mean_delta = +0.7562
  std_delta  =  0.0041
  min_delta  = +0.7509
  max_delta  = +0.7610
  positive   =  3/3 seeds

Prismion - GNN:
  mean_delta = +0.5743
  std_delta  =  0.0003
  min_delta  = +0.5738
  max_delta  = +0.5746
  positive   =  3/3 seeds

Prismion - UntiedLocal:
  mean_delta = +0.7656
  std_delta  =  0.0889
  min_delta  = +0.7026
  max_delta  = +0.8913
  positive   =  3/3 seeds
```

## Interpretation

This is the first clean result in this sequence where Prismion wins decisively.

The important distinction:

```text
Dynamic cancel additive task:
  canonical message passing sufficient
  Prismion not uniquely required

Phase-lock multiplicative task:
  Prismion positive
  ABC/GNN/summary/target controls fail
```

The result supports the narrower primitive hypothesis:

```text
Prismion-style phase operations are useful when the required local update is phase transport / rotation / multiplication.
```

It does not show that Prismion is needed for ordinary reachability, first-arrival phase-BFS, or additive dynamic cancellation.

## Sanity Checks

Shortcut controls are weak:

```text
SUMMARY_DIRECT_HEAD = 0.206
TARGET_MARKER_ONLY  = 0.210
majority_baseline   = 0.262
random_baseline     = 0.200
```

Wall leakage is clean:

```text
Prismion wall leak = 0.000
ABC wall leak      = 0.000
GNN wall leak      = 0.000
```

The result is not driven by a single seed:

```text
Prismion beats ABC/GNN/UntiedLocal on 3/3 paired seeds.
```

## Claim Boundary

```text
does not prove consciousness
does not prove full VRAXION
does not prove language grounding
does not prove general reasoning
does not show Prismion is required for reachability or additive dynamic cancel
does support Prismion as a useful primitive for local phase-lock / phase-rotation updates
```

## Next Step

The next useful test is transfer/composition, not another toy win:

```text
STABLE_LOOP_PHASE_LOCK_002_TRANSFER
```

Question:

```text
Can the same Prismion phase-lock primitive transfer to longer paths,
noisier gate fields, mixed cancellation + phase-lock tasks,
and learned/update-limited variants where the gate operation is not hard-coded?
```
