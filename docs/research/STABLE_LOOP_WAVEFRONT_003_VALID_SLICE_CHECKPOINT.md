# STABLE_LOOP_WAVEFRONT_003_VALID_SLICE_CHECKPOINT

Date: 2026-05-16

## Status

The CPU `valid_slice` run was stopped intentionally after it became clear that the exact 156-job validation queue is not compatible with the intended fast research loop.

Run root:

```text
target/pilot_wave/stable_loop_wavefront_003_long_s_curve/valid_slice
```

Stopped state:

```text
completed_jobs: 27 / 156
device: cpu
jobs: 12
train_examples: 4096
eval_examples: 4096
epochs: 80
```

The run was healthy before termination:

```text
stderr: empty
workers: alive
repo: clean
```

The monitor heartbeat automation was deleted after the run was stopped.

## Why The CPU Run Was Stopped

This was not a model failure. It was a throughput failure.

The valid slice is much larger than the smoke run:

```text
smoke:
  1024 examples
  20 epochs

valid_slice:
  4096 examples
  80 epochs

per-job multiplier:
  about 16x before accounting for long S and width32
```

The full queue contains:

```text
total_jobs: 156
HARD_WALL_ABC_LOOP: 36
HARD_WALL_PRISMION_PHASE_LOOP: 36
UNTIED_LOCAL_CNN_TARGET_READOUT: 36
LOCAL_MESSAGE_PASSING_GNN: 15
ORACLE_TRUNCATED_BFS_S: 15
control/oracle/global arms: 18
```

The completed rows were only `HARD_WALL_ABC_LOOP`, because the queue order front-loaded that arm. Therefore the partial CPU run cannot answer the main comparative question:

```text
Prismion vs GNN vs ABC vs untied-local
```

Observed CPU rate:

```text
27 jobs in about 6 hours
overall: about 4.5 jobs/hour
recent heavy configs: about 2 jobs/hour
```

Expected CPU completion if continued:

```text
optimistic: 30-36 hours total
realistic: 40-50 hours total
pessimistic heavy-tail: 60+ hours total
```

This is too slow for the desired research loop.

## Partial CPU Evidence

The partial rows do support one narrow point:

```text
HARD_WALL_ABC_LOOP works cleanly on the completed S/bucket cases.
```

Observed partial pattern:

```text
post-mask wall leak: 0.000
pre-mask frontier wall pressure: mostly 0.0015-0.011
S=4 / S=8 truncated accuracy: about 0.986-1.000
S=24 / S=32 truncated accuracy: 1.000 on completed width16 rows
same-weights ABC width16 S=24: 1.000 on all three seeds
```

Important caveat:

```text
The partial CPU run does not include Prismion, GNN, or untied-local rows.
It cannot confirm or reject Prismion-specific benefit.
```

## GPU Probe

CUDA is available:

```text
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
VRAM: 16 GB
torch: 2.5.1+cu121
cuda_available: true
```

Small GPU throughput probes were run under separate `target/` roots.

Representative config:

```text
arm: HARD_WALL_ABC_LOOP
width: 32
S: 24
train_examples: 4096
eval_examples: 4096
epochs: 5
```

Results:

```text
cuda jobs=1:
  1 job in about 153 seconds

cuda jobs=2:
  2 jobs in about 150 seconds
  stable and better throughput

cuda jobs=4:
  BrokenProcessPool
  not stable
```

Interpretation:

```text
One GPU worker under-utilizes the GPU.
Two GPU workers are currently the measured stable point.
Four GPU workers are not safe for this runner.
```

The low GPU utilization seen during probes is expected for small local-loop PyTorch models: the workload is many small kernels plus CPU/Python orchestration, not a large dense transformer-style batch.

## Current Scientific Checkpoint

What is solid so far:

```text
1. The wavefront task killed the summary shortcut.
2. The target-marker shortcut is controlled.
3. The hard-wall channel design removes wall leakage.
4. The long-S label split is correct:
   TRUNCATED_REACHABLE_S is used for S-curve logic,
   FULL_REACHABLE is only meaningful at large enough S.
5. ABC loop partial rows are clean and stable.
```

What is not established:

```text
1. Prismion is not yet confirmed on the valid slice.
2. GNN/ABC may be sufficient.
3. Untied-local compute comparison is not complete.
4. The 156-job exact valid slice is too slow for the desired <=10 minute loop.
```

## Runtime Lesson

The current validation design is scientifically clean but operationally too large.

The next runner shape should be:

```text
single question
single or few matched configs
<= 10 minutes wall time
GPU jobs=2 max
only escalate if the micro-result is informative
```

Do not repeat broad 156-job validation as the default.

## Recommended Next Experiment Shape

Use a narrow GPU micro-battery instead of the full valid slice.

Minimum decision battery:

```text
arms:
  HARD_WALL_ABC_LOOP
  HARD_WALL_PRISMION_PHASE_LOOP
  LOCAL_MESSAGE_PASSING_GNN
  UNTIED_LOCAL_CNN_TARGET_READOUT

width:
  16 only first

S:
  8, 24

train_mode:
  SAME_WEIGHTS_S_CURVE first

seeds:
  2026 only first

epochs:
  10-20

device:
  cuda

jobs:
  2
```

Decision:

```text
If Prismion is clearly not above ABC/GNN in the micro-battery:
  stop Prismion-specific claim.

If ABC/GNN are already near-perfect:
  declare canonical local message passing sufficient for this task.

If Prismion beats ABC/GNN with clean wall pressure:
  then run a slightly larger paired-seed confirmation.
```

## Copy Prompt For GPT Online

```text
We are testing VRAXION stable-loop / attractor mechanics using a deterministic 2D wavefront reachability task.

Current status:
  STABLE_LOOP_WAVEFRONT_001 killed summary shortcut but learned loops leaked through walls.
  STABLE_LOOP_WAVEFRONT_002 added hard-wall reached/frontier channels and fixed wall leakage.
  STABLE_LOOP_WAVEFRONT_003 fixed the S-vs-distance label bug:
    use TRUNCATED_REACHABLE_S for S-curve scoring,
    use FULL_REACHABLE only at sufficiently large S.

Smoke result was strong:
  FULL_REACHABILITY_POSITIVE
  PROPAGATION_CURVE_POSITIVE
  SAME_WEIGHTS_S_CURVE_POSITIVE
  SUMMARY_DIRECT_HEAD long/truncated near baseline
  TARGET_MARKER_ONLY controlled
  HARD_WALL_PRISMION_PHASE_LOOP same-weights average about 0.983
  post-mask wall leak 0.000
  pre-mask wall pressure about 0.003-0.008

We launched the exact valid slice:
  156 jobs
  4096 train examples
  4096 eval examples
  80 epochs
  3 seeds
  CPU jobs=12

After about 6 hours only 27/156 jobs were done, all HARD_WALL_ABC_LOOP.
The run was healthy but too slow, so we killed it.

Partial CPU evidence:
  ABC rows are clean:
    post-mask wall leak 0.000
    pre-mask wall pressure mostly 0.0015-0.011
    truncated accuracy generally about 0.986-1.000
    same-weights ABC width16 S=24 reached 1.000 on all three seeds
  But no Prismion/GNN/untied-local valid comparison was completed.

GPU probes:
  RTX 4070 Ti SUPER 16GB
  torch cuda available
  w32/S24/4096 train+eval/5 epochs:
    cuda jobs=1: 1 job about 153 sec
    cuda jobs=2: 2 jobs about 150 sec, stable
    cuda jobs=4: BrokenProcessPool, not stable

Constraint:
  We now want <=10 minute research loops, not 12-60 hour sweeps.

Question:
  What is the best next narrowed experiment?

Please reason adversarially.
Do not recommend another broad 156-job sweep.
We need a micro-battery that answers:
  1. Is stable tied local loop confirmed beyond summary/target shortcuts?
  2. Is Prismion actually better than ABC/GNN under matched S/bucket/seed?
  3. Or is canonical local message passing sufficient?
  4. What exact arms/S/seeds/epochs should run under 10 minutes on GPU jobs=2?
```

## Provisional Verdict

```text
CPU_VALID_SLICE_ABORTED_FOR_RUNTIME
PARTIAL_ABC_WAVEFRONT_CLEAN
PRISMION_NOT_VALIDATED_BY_PARTIAL_CPU_RUN
FULL_156_JOB_VALID_SLICE_TOO_EXPENSIVE_FOR_CURRENT LOOP
NEXT: GPU_MICRO_BATTERY <= 10 MIN
```

## GPU Failure Diagnostic

The GPU failure was diagnosed with target-only harnesses that import the wavefront runner without calling its `main()` path. This avoided tracked result-doc writes.

GPU diagnostics:

```text
CUDA fanout sanity:
  jobs=1 OK
  jobs=2 OK
  jobs=3 OK
  jobs=4 OK

runner eval-only:
  jobs=1 OK
  jobs=2 OK
  jobs=3 OK
  jobs=4 OK

runner train, 1 epoch:
  jobs=1 OK
  jobs=2 OK
  jobs=3 OK
  jobs=4 OK

throughput, 5 epochs:
  jobs=1  75.57s  75.57 sec/job
  jobs=2  86.00s  43.00 sec/job
  jobs=3  91.32s  30.44 sec/job
  jobs=4 102.70s  25.67 sec/job
```

Interpretation:

```text
The previous jobs=4 crash was not reproduced in isolation.
The likely cause was resource interference / launch context:
  CPU valid_slice was still running with 12 worker processes
  Windows ProcessPool + CUDA context startup pressure
  concurrent diagnostic launch pressure

GPU jobs=4 is usable when no large CPU sweep is running in parallel.
```

One diagnostic false alarm was also identified:

```text
Python launched from stdin can fail under Windows multiprocessing spawn
because child processes cannot reload the <stdin> main module.
Diagnostics must be launched from a real .py file.
```

## GPU Micro-Battery 3-Seed Result

Run:

```text
root: target/pilot_wave/stable_loop_wavefront_003_long_s_curve/gpu_micro_battery_3seed_001
device: cuda
jobs: 4
seeds: 2026,2027,2028
epochs: 10
train_examples: 4096
eval_examples: 4096
```

Runner labels:

```text
FULL_REACHABILITY_POSITIVE
SAME_WEIGHTS_S_CURVE_POSITIVE
```

Matched rows:

| Arm | Seed | TruncAcc | SameWeights | Unreachable false reach | Post-wall leak | Pre-wall pressure |
|---|---:|---:|---:|---:|---:|---:|
| `HARD_WALL_ABC_LOOP` | 2026 | 0.9954 | 0.9872 | 0.0056 | 0.0000 | 0.0031 |
| `HARD_WALL_ABC_LOOP` | 2027 | 1.0000 | 0.9911 | 0.0000 | 0.0000 | 0.0032 |
| `HARD_WALL_ABC_LOOP` | 2028 | 0.9717 | 0.9639 | 0.0530 | 0.0000 | 0.0039 |
| `HARD_WALL_PRISMION_PHASE_LOOP` | 2026 | 0.9990 | 0.9903 | 0.0020 | 0.0000 | 0.0037 |
| `HARD_WALL_PRISMION_PHASE_LOOP` | 2027 | 0.9995 | 0.9906 | 0.0010 | 0.0000 | 0.0039 |
| `HARD_WALL_PRISMION_PHASE_LOOP` | 2028 | 0.9905 | 0.9827 | 0.0036 | 0.0000 | 0.0033 |
| `LOCAL_MESSAGE_PASSING_GNN` S=24 | 2026 | 0.8511 | n/a | 0.3084 | n/a | n/a |
| `LOCAL_MESSAGE_PASSING_GNN` S=24 | 2027 | 0.8621 | n/a | 0.2858 | n/a | n/a |
| `LOCAL_MESSAGE_PASSING_GNN` S=24 | 2028 | 0.8528 | n/a | 0.3072 | n/a | n/a |
| `UNTIED_LOCAL_CNN_TARGET_READOUT` | 2026 | 0.5574 | 0.6271 | 0.2508 | n/a | n/a |
| `UNTIED_LOCAL_CNN_TARGET_READOUT` | 2027 | 0.8345 | 0.6841 | 0.2762 | n/a | n/a |
| `UNTIED_LOCAL_CNN_TARGET_READOUT` | 2028 | 0.5823 | 0.6443 | 0.5313 | n/a | n/a |

Paired deltas:

```text
Prismion - ABC, truncated_accuracy_by_S:
  seed 2026: +0.0037
  seed 2027: -0.0005
  seed 2028: +0.0188
  mean:      +0.0073

Prismion - ABC, same_weights_s_curve_accuracy:
  seed 2026: +0.0031
  seed 2027: -0.0005
  seed 2028: +0.0188
  mean:      +0.0071

Prismion - ABC, unreachable_false_reach_all_S:
  seed 2026: -0.0035
  seed 2027: +0.0010
  seed 2028: -0.0494
  mean:      -0.0173

Prismion - GNN S=24, truncated_accuracy_by_S:
  seed 2026: +0.1479
  seed 2027: +0.1375
  seed 2028: +0.1377
  mean:      +0.1410

Prismion - untied-local, same_weights_s_curve_accuracy:
  seed 2026: +0.3632
  seed 2027: +0.3065
  seed 2028: +0.3383
  mean:      +0.3360
```

Interpretation:

```text
GPU_JOBS4_CONFIRMED_IN_ISOLATED_RUNS
PRISMION_HINT_SURVIVES_3SEED
CANONICAL_GNN_NOT_SUFFICIENT_IN_THIS_MICRO_BATTERY
UNTIED_LOCAL_CNN_NOT_SUFFICIENT_IN_THIS_MICRO_BATTERY
ABC_IS_STRONG_BUT_PRISMION_HAS_SMALL POSITIVE_DELTA
```

Claim boundary:

```text
This is still not a full validation slice.
It is a strong micro-battery signal.
The Prismion edge over ABC is small but replicated in mean over 3 seeds.
The edge over GNN and untied-local is large in this battery.
```

Reproducibility note:

```text
CUDA runs emitted PyTorch cuBLAS determinism warnings.
Future confirmation runs should set:
  CUBLAS_WORKSPACE_CONFIG=:4096:8
before Python starts.
```

## GPU Micro-Battery 5-Seed Confirmation

Run:

```text
root: target/pilot_wave/stable_loop_wavefront_003_long_s_curve/gpu_micro_battery_5seed_001
device: cuda
jobs: 4
seeds: 2026,2027,2028,2029,2030
epochs: 10
train_examples: 4096
eval_examples: 4096
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Runner labels:

```text
FULL_REACHABILITY_POSITIVE
SAME_WEIGHTS_S_CURVE_POSITIVE
```

Mean truncated accuracy:

```text
HARD_WALL_PRISMION_PHASE_LOOP:   0.9974
HARD_WALL_ABC_LOOP:              0.9897
LOCAL_MESSAGE_PASSING_GNN:       0.8995
UNTIED_LOCAL_CNN_TARGET_READOUT: 0.6845
SUMMARY_DIRECT_HEAD:             0.0922
TARGET_MARKER_ONLY:              0.0372
```

Prismion vs ABC paired deltas:

```text
truncated_accuracy_by_S:
  seed 2026: +0.0037
  seed 2027: -0.0005
  seed 2028: +0.0188
  seed 2029: -0.0010
  seed 2030: +0.0173
  mean:      +0.0077
  positive:  3 / 5

same_weights_s_curve_accuracy:
  seed 2026: +0.0031
  seed 2027: -0.0005
  seed 2028: +0.0188
  seed 2029: -0.0010
  seed 2030: +0.0171
  mean:      +0.0075
  positive:  3 / 5

unreachable_false_reach_all_S:
  mean delta: -0.0170
  interpretation: Prismion has fewer false reaches on average.

pre_mask_frontier_wall_write_norm:
  mean delta: +0.00035
  interpretation: Prismion has slightly higher pre-mask wall pressure, still very low.
```

Prismion vs other baselines:

```text
Prismion - GNN S=24 truncated_accuracy_by_S:
  mean: +0.1416
  min:  +0.1375
  max:  +0.1479

Prismion - untied-local same_weights_s_curve_accuracy:
  mean: +0.3336
  min:  +0.2985
  max:  +0.3632
```

5-seed verdict:

```text
PRISMION_HINT_SURVIVES_5SEED
STABLE_LOOP_SIGNAL_CONFIRMED_IN_MICRO_BATTERY
GNN_NOT_SUFFICIENT_IN_THIS_MICRO_BATTERY
UNTIED_LOCAL_NOT_SUFFICIENT_IN_THIS_MICRO_BATTERY
SUMMARY_AND_TARGET_CONTROLS_REMAIN_WEAK
```

Claim boundary:

```text
The Prismion edge over ABC is real but small at width16/SAME_WEIGHTS/epochs10.
The edge is not seed-universal: ABC slightly wins on seeds 2027 and 2029.
The next falsification is whether the edge survives width32 or longer training.
```

## GPU Micro-Battery Width32 3-Seed Confirmation

Run:

```text
root: target/pilot_wave/stable_loop_wavefront_003_long_s_curve/gpu_micro_battery_w32_3seed_001
device: cuda
jobs: 4
seeds: 2026,2027,2028
width: 32
epochs: 10
train_examples: 4096
eval_examples: 4096
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Runner labels:

```text
FULL_REACHABILITY_POSITIVE
SAME_WEIGHTS_S_CURVE_POSITIVE
```

Mean truncated accuracy:

```text
HARD_WALL_PRISMION_PHASE_LOOP:   0.9998
HARD_WALL_ABC_LOOP:              0.9875
LOCAL_MESSAGE_PASSING_GNN:       0.8995
UNTIED_LOCAL_CNN_TARGET_READOUT: 0.5476
SUMMARY_DIRECT_HEAD:             0.2436
TARGET_MARKER_ONLY:              0.0368
```

Prismion vs ABC paired deltas:

```text
truncated_accuracy_by_S:
  seed 2026: +0.0132
  seed 2027: +0.0103
  seed 2028: +0.0137
  mean:      +0.0124
  positive:  3 / 3

same_weights_s_curve_accuracy:
  seed 2026: +0.0132
  seed 2027: +0.0103
  seed 2028: +0.0131
  mean:      +0.0122
  positive:  3 / 3

unreachable_false_reach_all_S:
  mean delta: -0.0243
  interpretation: Prismion has fewer false reaches.

pre_mask_frontier_wall_write_norm:
  mean delta: -0.00027
  interpretation: Prismion has slightly lower pre-mask wall pressure.
```

Prismion vs other baselines:

```text
Prismion - GNN S=24 truncated_accuracy_by_S:
  mean: +0.1445

Prismion - untied-local same_weights_s_curve_accuracy:
  mean: +0.3735
```

Width32 verdict:

```text
PRISMION_HINT_STRENGTHENS_AT_WIDTH32
PRISMION_BEATS_ABC_3_OF_3_SEEDS_AT_WIDTH32
PRISMION_HAS_LOWER_FALSE_REACH_AND_LOWER_PRE_WALL_PRESSURE_AT_WIDTH32
GNN_AND_UNTIED_LOCAL_REMAIN_BEHIND
```

Claim update:

```text
The width16 result showed a small average Prismion edge.
The width32 result strengthens the signal: Prismion wins all tested seeds against ABC.
The next confirmation is width32 with 5 seeds before any broader run.
```
