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
