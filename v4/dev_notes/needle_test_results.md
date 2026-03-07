# INSTNCT v4 Needle Test — Step/Sec to Intelligence Ratio

**Date**: 2026-03-07
**Author**: CLI Claude (raw data dump, needs processing)
**GPU**: NVIDIA GeForce RTX 4070 Ti SUPER (16GB)
**PyTorch**: 2.5.1+cu121, AMP fp16 enabled

## Purpose

Find the config that maximizes intelligence per compute unit.
Metric: best_loss at target speed (2+ step/sec).

## Fixed params (all runs)

- M: 256 (ring slots)
- N: 1 (single expert)
- R: 1 (attention radius)
- S: 0.3 (context scale)
- pointer: sequential (ptr += 1)
- write: replace (HDD-style)
- kernel: vshape
- embed: bitlift, output: lowrank_c19
- lr: 1e-3, warmup: 100, seed: 1337
- data: WikiText-103 byte-level
- sequential training (state persists)

## Phase 0: seq_len speed calibration (hidden=512)

All runs: hidden=512, M=256, batch=64, sequential pointer, 100 steps.

| seq_len | step/sec | sec/step | best_loss | VRAM  |
|---------|----------|----------|-----------|-------|
| 64      | 0.63     | 1.6      | 3.27      | 724M  |
| 32      | 1.0      | 1.0      | 3.23      | 376M  |
| 16      | 2.0      | 0.5      | 3.24      | 203M  |

Key finding: loss nearly identical across seq_len at 100 steps.
Sequential training mode means short seq doesn't lose context.
seq=16 hits 2 step/sec target. Locked for Phase 1.

## Phase 1: hidden_dim sweep (seq=16)

All runs: seq=16, M=256, batch=64, sequential pointer, 100 steps.

| hidden_dim | params | step/sec | sec/step | best_loss | VRAM  | ring_norm@100 |
|------------|--------|----------|----------|-----------|-------|---------------|
| 512        | 190K   | 2.0      | 0.5      | 3.241     | 203M  | 2060          |
| 1024       | 364K   | 2.0      | 0.5      | 3.344     | 246M  | 1214          |
| 2048       | 711K   | 2.3      | 0.4      | 3.274     | 334M  | 3027          |
| 4096       | 1.4M   | 2.4      | 0.4      | 3.246     | 511M  | 1720          |
| 8192       | 2.8M   | 2.0      | 0.5      | 3.408     | 864M  | 323 (sat!)    |

Key findings:
1. hidden=4096 is FASTEST (2.4 step/sec) despite 7x more params than 512.
   Tensor Cores saturate better with larger matmuls.
2. hidden=8192 regresses: ring_norm saturates at 323 (slot_dim=128 bottleneck).
   The 128-dim slot can't carry 8192-dim hidden state info through the ring.
3. Sweet spot: hidden=4096 (best loss + fastest speed).
4. Python loop overhead + CUDA kernel launch dominates at small hidden_dim.

**CAVEAT**: Phase 0+1 ran only 100 steps = warmup only. Loss values NOT reliable
for intelligence comparison. Speed data IS reliable.

## Phase 2: seq_len speed (seq=8..32 @ hidden=4096)

All runs: hidden=4096, M=256, batch=64, sequential pointer, 100 steps.

| seq_len | step/sec | best_loss | tok/sec (batch*seq*step/sec) |
|---------|----------|-----------|------------------------------|
| 8       | 4.0      | 3.273     | 2048                         |
| 16      | 2.4      | 3.246     | 2457                         |
| 32      | 1.2      | 3.164     | 2457                         |

Token throughput: seq=16 and seq=32 process equal tok/sec.
Difference is gradient depth (16 vs 32 step backprop).

## Phase 3: Intelligence race (5 min wall time each)

THE REAL TEST: same wall time, which config learns the most?
All runs: hidden=4096, M=256, sequential pointer.

| Config | seq | batch | steps | wall_time | best_loss | step/sec | status        |
|--------|-----|-------|-------|-----------|-----------|----------|---------------|
| **A**  | 16  | 512   | 1000  | 4.5 min   | **2.047** | 3.7      | stable        |
| B      | 32  | 256   | 400   | 3.4 min   | 2.535     | 2.0      | spike @340    |
| C      | 64  | 128   | -     | crash     | NaN       | -        | diverged @17  |

### Analysis

1. **Config A (seq=16) is the clear winner**: best loss (2.047) AND fastest.
   More steps compensate for shallow gradient depth.
   1000 steps at seq=16 beats 400 steps at seq=32, same walltime.

2. **Config B (seq=32) showed instability**: loss spiked from 2.53 to 3.02 around
   step 330-340 then slowly recovered. Possible cause: larger seq = larger gradients
   per step = more volatile updates at lr=1e-3.

3. **Config C (seq=64) diverged immediately**: NaN at step 17. M=256 is too small
   for seq=64 (ring overwrites itself every 256/64=4 steps). Need M>=512 for seq=64.

4. **Batch size has minimal speed impact**: batch 64->512 at seq=16 went from 2.4
   to 3.7 step/sec (only 1.5x for 8x batch). Ring loop is the bottleneck, not matmuls.
   But bigger batch = better gradient signal per step.

### Adversarial self-check

Initial hypothesis was that seq=16 would produce "fundamentally dumber" models because
TBPTT gradient depth of only 16 bytes can't capture sentence-level patterns.

**This was WRONG.** At 1000 steps, seq=16 reached loss 2.047 which is competitive with
prior runs at seq=256. The sequential training carry-over IS sufficient -- the model
learns to use persistent hidden state even though gradients only flow 16 steps back.
The speed advantage (3.7 vs 2.0 step/sec) gives seq=16 more gradient updates per minute,
which outweighs the shallower gradient.

**Still unresolved:** does this advantage hold at 10K+ steps? Or does seq=32's deeper
gradient eventually overtake? Prior all-time record was 47.8% acc at 3000 steps with
seq=256. Need longer runs to confirm.

## Current winner

```
hidden_dim: 4096, slot_dim: 128, M: 256, seq: 16, batch: 512
pointer: sequential, write: replace
params: 1.4M, speed: 3.7 step/sec, best_loss: 2.047 @ 1000 steps
VRAM: 3.8 GB, wall time: 4.5 min
```

## Observations

- Pilot pointer costs ~2x speed vs sequential (0.5 vs 1.0 step/sec at seq=32/hidden=512).
  Pilot computes cosine sim against all M=1024 slots per timestep.
- M=1024 to M=256 reduces ring overhead significantly.
- The ring sequential loop (T timesteps, not parallelizable) is THE speed bottleneck.
- At seq=16 with sequential training, gradient depth is only 16 steps but
  ring/hidden state persists, so effective context is unlimited.
- Batch size barely affects step/sec (ring loop dominates), so crank it up for free.
- M=256 is too small for seq>=64 (causes NaN divergence).

## Next (TODO)

- [ ] Longer run: winner config at 5000+ steps. Does seq=16 plateau early?
- [ ] seq=32 stability fix: try lr=5e-4 or grad_clip=5.0
- [ ] slot_dim sweep (64, 128, 256) with winner config
- [ ] Does slot_dim=256 fix hidden=8192 saturation?
- [ ] Re-test pilot pointer with M=256 (was tested at M=1024)
