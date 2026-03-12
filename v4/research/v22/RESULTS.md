# VRAXION v22 — Research Results Summary

## Self-Wiring Graph Network — Consolidated Findings

**Last updated:** 2026-03-13
**Branch:** v4.1

---

## Canonical Configuration (2026-03-13)

| Component | Choice | Why |
|-----------|--------|-----|
| **Neuron model** | Capacitor (integrate-and-fire) | 2-3x better than leaky_relu on all tasks |
| **Threshold** | 0.5 | Config sweep winner (tested 0.3, 0.5, 0.7, 1.0) |
| **Leak** | 0.85 | Config sweep winner (tested 0.8, 0.85, 0.9, 0.92, 0.95) |
| **Reset** | 0.0 (simple) / -0.2 AHP (64-class) | AHP helps at 64-class, hurts at 16-class |
| **Mask** | Ternary int8 (-1/0/+1) | Sign (inhibit/excite) is essential. int8 = float32 accuracy, 4x less memory |
| **Weights** | NONE | Weight matrix is unnecessary — mask-only is BETTER (+19% on 64-class) |
| **Topology** | Flat graph (no layers) | Structural search > parametric search |
| **I/O** | Shared (first V neurons) | Same accuracy, fewer neurons |
| **Input injection** | First tick only, gain=2.0 | Simplest, works best |
| **Ticks** | 6 | Sweet spot for cost/accuracy |
| **Density** | 6% | Proven empirically |
| **Flip rate** | 30% | Most powerful mutation operator |

---

## Neuron Model Comparison

### 3-Way A/B (seed=42, 256 neurons, 4K steps)

| Mode | 16-class | 64-class | Sparsity | Notes |
|------|----------|----------|----------|-------|
| **Capacitor** | **75.0%** | 20.3% | 83-85% | Best for 16-class |
| **Bio-capacitor** (AHP=-0.2) | 68.8% | **26.6%** | 90-97% | Best for 64-class (interference reduction) |
| LeakyReLU | 37.5% | 7.8% | 39-65% | Baseline — too many active neurons |

### Why Capacitor Wins

- **Natural sparsity**: 17-42 active neurons per input (vs 50-157 for leaky_relu)
- **Threshold creates pathway isolation**: only strongly-driven neurons fire
- **Fire+reset prevents runaway**: charge resets after firing, no overflow
- **AHP (-0.2)**: post-fire inhibition reduces cross-class interference at scale

### What FAILED in Neuron Model

| Attempt | Result | Why Failed |
|---------|--------|-----------|
| Full bio params (τ=20ms, gain=0.033) | 43.8%, 99% sparse | Designed for 7K+ synapses/neuron, we have ~hundreds |
| Refractory period (1-2 ticks) | No effect | At 6-tick timescale, neurons don't fire fast enough for it to matter |
| AHP on 16-class | -6.2% vs simple cap | Over-regulates where there's no interference problem |

---

## Weight Representation Sweep

### 64-class, 256 neurons, capacitor

| Representation | Accuracy | Kept | Memory | Bytes/elem |
|---------------|----------|------|--------|-----------|
| **Mask-only (ternary)** | **28.1%** | **17** | 0.06 MB | 1 |
| Ternary int8 + binary int8 weight | 10.9% | 7 | 0.12 MB | 2 |
| Binary float32 (original) | 9.4% | 5 | 0.50 MB | 8 |

**Key discovery**: The weight matrix HURTS. It doubles the search space without adding useful information. Mask-only (ternary) is 3x better accuracy AND 8x less memory.

### Mask Type Comparison (16-class)

| Mask type | Accuracy | Bits/connection |
|-----------|----------|----------------|
| **Ternary** (-1/0/+1) | **62.5%** | 2 |
| Binary (0/1) | 43.8% | 1 |

Sign (inhibit/excite) is essential. The network needs inhibitory connections to say "no" — without them, every signal only reinforces, causing interference.

---

## Logic Benchmarks (256 neurons, capacitor)

| Task | Accuracy | Baseline | Notes |
|------|----------|----------|-------|
| AND 4-bit | **100%** | 50% | Solved |
| OR 4-bit | **100%** | 50% | Solved |
| XOR 4-bit | **93.8%** | 50% | Near-perfect |
| NAND 4-bit | **100%** | 50% | Solved |
| XNOR 4-bit | **93.8%** | 50% | Near-perfect |
| A > B (range 0-3) | **87.5%** | 62.5% | Good, above majority baseline |
| A > B (range 0-5) | **80.6%** | 58.3% | Good |
| A > B (range 0-7) | **78.1%** | 56.2% | Good |
| A+B mod 4 | 56.2% | 25.0% | 2.25x random — multi-class output is hard |
| A+B mod 6 | 50.0% | 16.7% | 3x random |
| A+B mod 8 | 29.7% | 12.5% | 2.4x random — same wall as 64-class lookup |

**Pattern**: Binary output (2-class) → works great. Multi-class output → hits interference wall.

---

## The 64-Class Wall — Diagnosis

### Root Cause: Interference from Shared Activation

| Metric | 16-class | 64-class |
|--------|----------|----------|
| Active neurons / input | 42 | 157 (leaky) / 33-38 (capacitor) |
| Sparsity | 83% | 39% (leaky) / 85-97% (capacitor) |
| Connection overlap | 2.1 | 9.84 |
| Mutation acceptance rate | 0.25% | 0.08% |
| Interference (mixed) | 12% | 52.5% |

At 64-class with leaky_relu: 98% of neurons fire for every input. Every mutation that helps one class hurts ~10 others. Capacitor reduces this dramatically but doesn't eliminate it.

### What FAILED to Break the Wall

| Attempt | Delta | Why |
|---------|-------|-----|
| Diff-guided mutation | -8.3% (16-class) | Only targets last-hop connections, but signal travels 6 ticks deep |
| More steps (8K→16K) | Marginal | Acceptance rate near 0 — more attempts ≠ more progress |

### What HELPS (but doesn't solve)

| Approach | 64-class result | vs baseline |
|----------|----------------|-------------|
| Capacitor | 20.3% | +12.5% over leaky |
| Bio-capacitor (AHP) | 26.6% | +18.8% over leaky |
| Mask-only (no weights) | 28.1% | +20.3% over leaky |

---

## Scaling Analysis

### Hardware: AMD Ryzen 9 3900X (12C/24T, 64MB L3)

### Memory with mask-only int8 (canonical config)

| Neurons | Mask size | Fits in | Neurons/class (64-class) |
|---------|-----------|---------|-------------------------|
| 256 | 64 KB | L2 | 3 (too few) |
| 512 | 256 KB | L2 | 7 |
| 1024 | 1 MB | L2 | 15 (= 16-class@256 that scores 75%) |
| 2048 | 4 MB | L3 | 31 |
| 4096 | 16 MB | L3 | 63 |
| 8192 | 64 MB | L3 (limit) | 127 |

### Comparison: old (float32 mask + weight) vs new (int8 mask-only)

| Neurons | Old (3× N² × 4B) | New (1× N² × 1B) | Savings |
|---------|-------------------|-------------------|---------|
| 256 | 768 KB | 64 KB | 12x |
| 1024 | 12 MB | 1 MB | 12x |
| 4096 | 192 MB (RAM!) | 16 MB (L3) | 12x |

**8x more neurons fit in cache** with the new representation.

---

## Architecture Diagram (Updated)

```
INPUT → one-hot [0,0,...,1,...,0]
      │
      ▼ (first tick only, gain=2.0)
╔═══════════════════════════════╗
║  I/O NEURONS (first V)        ║ ← shared input+output
╠═══════════════════════════════╣
║                               ║
║   FLAT GRAPH (N-V internal)   ║
║   ternary mask int8 (-1/0/+1) ║
║   NO weight matrix            ║
║   capacitor neuron model      ║
║     threshold=0.5, leak=0.85  ║
║     fire → reset to 0.0       ║
║   6 ticks propagation         ║
║                               ║
╠═══════════════════════════════╣
║  I/O NEURONS (same V)         ║ ← read accumulated output
╚═══════════════════════════════╝
      │
      ▼ softmax → prediction
      │
  ┌───┴────────────────┐
  │  LEARNING           │
  │  Mutation+Selection  │
  │                      │
  │  1. Mutate mask      │
  │     (flip -1↔0↔+1)  │
  │  2. Forward all      │
  │     inputs (eval)    │
  │  3. Keep if better   │
  │     Revert if worse  │
  │                      │
  │  Self-wiring:        │
  │  inverse arousal     │
  └──────────────────────┘
```

---

## Open Questions / Next Steps

1. **Neuron scaling**: Does 1024 neurons on 64-class match 256 on 16-class (75%)?
2. **Timing**: How slow is 1K step at 1024/2048/4096 neurons?
3. **Sparse matmul**: At 85-97% sparsity, can we use scipy.sparse for the forward pass?
4. **Population**: Parallel eval of multiple mutants (exploit 12 cores)?
5. **Modular structure**: Class-group sub-networks to reduce interference?
6. **Ring buffer integration**: Temporal tasks need memory beyond 6-tick state decay
