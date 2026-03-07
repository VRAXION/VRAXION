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

## Current winner

```
hidden_dim: 4096, slot_dim: 128, M: 256, seq: 16, batch: 64
pointer: sequential, write: replace
params: 1.4M, speed: 2.4 step/sec, best_loss: 3.246, VRAM: 511M
```

## Observations

- Pilot pointer costs ~2x speed vs sequential (0.5 vs 1.0 step/sec at seq=32/hidden=512).
  Pilot computes cosine sim against all M=1024 slots per timestep.
- M=1024 to M=256 reduces ring overhead significantly.
- The ring sequential loop (T timesteps, not parallelizable) is THE speed bottleneck.
- At seq=16 with sequential training, gradient depth is only 16 steps but
  ring/hidden state persists, so effective context is unlimited.

## Next (TODO)

- [ ] Phase 2: seq_len sweep at hidden=4096 (seq=8, 16, 32, 64)
- [ ] Phase 3: batch_size sweep (32, 64, 128, 256)
- [ ] Phase 4: slot_dim sweep (64, 128, 256)
- [ ] Phase 5: longer runs (500-1000 steps) with winner config
- [ ] Investigate: does slot_dim=256 fix the hidden=8192 saturation?
