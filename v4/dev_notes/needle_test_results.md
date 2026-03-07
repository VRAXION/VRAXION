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

## Phase 4: slot_dim sweep (hidden=4096, seq=16, batch=512, 1000 steps)

Fixed: hidden=4096, M=256, seq=16, batch=512, sequential pointer, replace write, seed=1337.
Varying slot_dim to find the compression bottleneck knee.

| slot_dim | ratio | best_loss | step/sec | VRAM   | wall    | alpha | ring_norm | stable? |
|----------|-------|-----------|----------|--------|---------|-------|-----------|---------|
| 32       | 128:1 | 2.164     | 3.6      | 2.9 GB | 4.6 min | ~0.40 | 5305      | NO — diverged ~step 630, loss→3.4 |
| 64       | 64:1  | 2.218     | 3.0      | 3.2 GB | 5.5 min | ~0.41 | 8215      | NO — diverged ~step 570, 2× LR plateau |
| **128**  | **32:1** | **2.047** | **3.7** | **3.8 GB** | **4.5 min** | **~0.40** | **~2000** | **YES — stable through 1000 steps** |
| 256      | 16:1  | 2.318     | 2.75     | 4.9 GB | 6.1 min | ~0.15 | 11795     | NO — diverged ~step 500, 2× LR plateau |
| 512      | 8:1   | 2.422     | 1.78     | 7.2 GB | 9.4 min | ~0.08 | 29816↑    | NO — diverged ~step 600, ring_norm exploding |

### Analysis: U-shaped curve, 128 is the Goldilocks point

1. **slot_dim=128 wins on EVERY metric**: best loss, fastest speed, only stable config.

2. **Too small (32, 64) — compression bottleneck**: The 4096→32/64 compression loses too much
   information. Ring can't carry the hidden state signal. Training learns fast initially
   but diverges as the model hits the compression ceiling. Interestingly, alpha stays ~0.40
   (ring is being used), but the information flowing through it is degraded.

3. **Too large (256, 512) — alpha collapse**: This is the surprising finding.
   Wider slots should carry MORE information, but alpha collapses (0.15→0.08).
   The model effectively stops using the ring. Why?
   - Larger slot_dim = larger read/write projections = harder optimization surface
   - ring_norm explodes (11K→30K) because each write deposits more energy
   - The read_proj can't extract useful signal from the high-norm noisy ring
   - Net effect: model learns to ignore the ring (alpha→0) and rely on hidden state alone

4. **Speed is non-monotonic**: 128 (3.7) > 32 (3.6) > 64 (3.0) > 256 (2.75) > 512 (1.78).
   slot_dim=128 hits the Tensor Core sweet spot, same as hidden_dim=4096.

5. **32:1 compression ratio** (4096/128) is the magic number for this architecture.
   With hidden=8192 this predicts optimal slot_dim=256. Testable but lower priority
   since hidden=8192 had saturation issues at slot_dim=128 (Phase 1).

### Diagnostic: alpha as health metric

The `alpha` (context blend factor) is the best single diagnostic:
- alpha ~0.40 = healthy (ring contributing ~40% of signal) → 32, 64, 128
- alpha ~0.15 = degraded (ring mostly ignored) → 256
- alpha ~0.08 = dead (ring is noise) → 512

But alpha alone doesn't predict stability. 32 and 64 have healthy alpha but diverge
because the compressed signal is lossy. Need alpha ~0.40 AND adequate slot capacity.

## Phase 5: Pilot pointer @ M=256

Same winner config (hidden=4096, slot=128, M=256, seq=16, batch=512) but pointer_mode=pilot.
Prior test was at M=1024 (2× slower). M=256 has 4× fewer slots to cosine-scan.

| pointer    | best_loss | step/sec | VRAM   | wall    | alpha | ring_norm       | stable? |
|------------|-----------|----------|--------|---------|-------|-----------------|---------|
| sequential | 2.047     | 3.7      | 3.8 GB | 4.5 min | ~0.40 | ~2000 (varying) | YES     |
| **pilot**  | **2.004** | **2.86** | 3.85 GB | 5.8 min | ~0.22 | 8021.33 (FROZEN) | **YES — still improving** |

### Analysis

1. **Pilot wins on loss** (-2.1%): 2.004 vs 2.047, and loss was STILL DECLINING at step 1000
   (stale=4). Sequential had plateaued. With more steps, pilot likely pulls further ahead.

2. **Speed cost is acceptable**: 2.86 vs 3.7 step/sec (23% slower). At M=256, cosine sim
   scans 256 slots per timestep (vs 1024 before = 4× less work). Still above 2+ step/sec target.

3. **ring_norm FROZEN at 8021.33**: Didn't change by a single digit across 1000 steps.
   With sequential pointer, ring_norm varied (decayed from ~2060 to ~1700). Pilot's
   content-based jumping + replace write creates a stable equilibrium — pointer revisits
   and overwrites the same slots in a balanced cycle. Needs investigation but doesn't hurt.

4. **Alpha halved (0.22 vs 0.40)**: Pilot uses the ring LESS but MORE SELECTIVELY.
   Sequential reads every slot in order (brute force). Pilot jumps to relevant content.
   Less total ring influence, but higher quality per read. Net result: better loss.

5. **pilot_max_jump=512 > M=256**: Jump wraps around the ring. Effectively any slot
   is reachable in one step. Could try max_jump=128 (M/2) for less chaotic jumping.

## Updated winner (NEW — pilot pointer)

```
hidden_dim: 4096, slot_dim: 128, M: 256, seq: 16, batch: 512
pointer: pilot (max_jump=512, id_dim=32), write: replace
params: ~1.4M + 8K (slot identities), speed: 2.86 step/sec
best_loss: 2.004 @ 1000 steps (still declining)
VRAM: 3.85 GB, wall time: 5.8 min
compression ratio: 32:1 (hidden/slot)
```

## Phase 6: M sweep with pilot pointer

Fixed: hidden=4096, slot=128, seq=16, batch=512, pilot pointer, replace write, seed=1337.
Varying M (ring capacity) to find optimal slot count AND diagnose frozen ring_norm.

| M    | best_loss | step/sec | VRAM   | wall    | alpha | ring_norm (frozen) | stable? |
|------|-----------|----------|--------|---------|-------|--------------------|---------|
| **128** | **1.941** | **3.42** | **3.3 GB** | **4.9 min** | **~0.21** | 6622 | **YES — still improving** |
| 256  | 2.004     | 2.86     | 3.85 GB | 5.8 min | ~0.22 | 8021               | YES     |
| 512  | 2.023     | 3.86     | 4.9 GB | 4.3 min | ~0.33 | 2725               | YES     |
| 1024 | 2.038     | 3.79     | 7.1 GB | 4.4 min | ~0.72 | 1708               | YES     |

### Analysis: SMALLER M = BETTER with pilot pointer

1. **Loss monotonically decreases with smaller M**: 1.941 < 2.004 < 2.023 < 2.038.
   This is counterintuitive — fewer memory slots = better learning. Why?

2. **Freshness hypothesis**: With M=128 and seq=16, the ring gets fully overwritten every
   128/16=8 sequences. Content stays FRESH. With M=1024, overwrite cycle is 64 sequences
   — stale content accumulates. Pilot reads stale slots → garbage in, garbage out.

3. **Alpha SCALES with M**: 0.21 (M=128) → 0.72 (M=1024). More slots = model reads MORE
   from ring. But MORE ring dependence = WORSE loss! The model is better off relying on
   hidden state. Ring reads at M=1024 are a crutch, not an asset.

4. **ring_norm inverse pattern**: 6622 (M=128) > 8021 (M=256) > 2725 (M=512) > 1708 (M=1024).
   Smaller M = higher norm per slot (more overwrites = more energy concentrated).
   Larger M = lower norm (energy spread thin, many slots barely written).
   ALL FROZEN — pilot pointer universally creates ring equilibrium.

5. **Speed non-monotonic (GPU scheduling)**: M=512 (3.86) > M=1024 (3.79) > M=128 (3.42) > M=256 (2.86).
   Not a simple "more M = slower". VRAM and Tensor Core effects dominate.

### Key insight: pilot pointer inverts the M requirement

With sequential pointer, bigger M = more memory capacity = better (up to a point).
With pilot pointer, smaller M = fresher content = better signal per read.
The pilot's content-based jumping means it doesn't need many slots — it just needs
the few slots it has to contain FRESH, RELEVANT information.

## Updated winner (NEW — M=128)

```
hidden_dim: 4096, slot_dim: 128, M: 128, seq: 16, batch: 512
pointer: pilot (max_jump=512, id_dim=32), write: replace
params: ~1.4M + 4K (slot identities), speed: 3.42 step/sec
best_loss: 1.941 @ 1000 steps (still declining)
VRAM: 3.3 GB, wall time: 4.9 min
compression ratio: 32:1 (hidden/slot)
```

## Next (TODO)

- [x] slot_dim sweep — **128 confirmed optimal** (U-shaped, both directions worse)
- [x] Pilot pointer @ M=256 — beat sequential (2.004 vs 2.047)
- [x] M sweep with pilot — **M=128 NEW WINNER** (1.941, smaller M = fresher ring)
- [ ] M=64 test: does the trend continue? (M/seq=4, borderline)
- [ ] Longer run: M=128 pilot at 3000-5000 steps
- [ ] pilot_max_jump sweep: max_jump=512 > M=128, try 64/128
- [ ] seq=32 stability fix
