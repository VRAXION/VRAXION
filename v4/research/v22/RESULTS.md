# VRAXION v22 — Research Results Summary
## Self-Wiring Graph Network — Empirical Findings

**Date:** 2026-03-12
**Session:** Full-day ablation study + architecture search

---

## Best Configuration (87.5% on 16-class)

| Component | Choice | Alternatives Tested | Why This Won |
|-----------|--------|-------------------|-------------|
| **Topology** | Flat graph (no layers) | MLP, RNN, Pipeline | Graph beats all — structural search > parametric |
| **Mask** | Ternary (-1/0/+1) | Binary (0/1) | Flip mutation (+1↔-1) is most powerful operator |
| **Weights** | Binary (0.5/1.5) | Float32, None, Ternary, 4-level | Same accuracy as float, 11x smaller |
| **Activation** | Leaky ReLU | abs, ternary, binary, tanh, C19, capacitor | Best at 160n scale, continuous = better for selection |
| **Inhibition** | Group WTA k=5 | No inhibition, Local WTA, Soft WTA | Forces orthogonal patterns, +6% over baseline |
| **Self-wiring** | Inverse arousal | Normal, Aggressive, None | More wiring when good (exploit), less when bad (explore) |
| **I/O** | Shared (first V neurons) | Dedicated layers, Flat projection | Same accuracy, fewer neurons |
| **Input** | First tick only | Every tick, Blend, Soft inject | Fewest connections needed |
| **Diff** | NOT in input (learning signal only) | In input (2V neurons) | Redundant with persistent state |
| **Flip rate** | 30% | 5%, 15% | Sweet spot for search efficiency |

---

## All Test Results

### Lookup Tasks
| Task | Best Config | Accuracy | Connections |
|------|------------|----------|-------------|
| 16-class (80n) | Ternary mask + binary W + flip=30% | **87.5%** | 890 |
| 32-class (160n) | v18 leaky_relu + self-wire | **100%** | 231 |
| 64-class (320n) | All configs | **55% (wall)** | varies |

### Diverse Tasks
| Task | Accuracy | Note |
|------|----------|------|
| Pattern completion | **100%** | Routing task, solved easily |
| Sorting 4 elements | **100%** | Algorithmic, comparison+reorder |
| XOR 4-bit | 59% | Weak at nonlinear combination |
| Sequence (prev+curr mod N) | 47% | Temporal dependency, needs memory |
| English bigram prediction | **28% (7.8x random)** | Real data! |

### Architecture Comparisons
| Comparison | Winner | Score | Loser | Score |
|-----------|--------|-------|-------|-------|
| Graph vs MLP | Graph | 81% | MLP | 50% |
| Graph vs RNN | Graph | 81% | Elman | 38% |
| Binary vs Soft mask | Binary | 100% | Sigmoid | 81% |
| Sparse vs Distributed coding | Sparse | 88% | Distributed | 50% |
| Pipeline vs Single graph | Single | 81% | 4-chip pipe | 50% |
| Build vs Sculpt (prune-only) | Build | 100% | Sculpt | 69% |

---

## Key Discoveries

### 1. Ternary Mask + Flip Mutation
The sign of connections (excitatory/inhibitory) living in the mask instead of weights, combined with a flip mutation that toggles +1↔-1, is the single most impactful architectural choice. 87.5% vs 75% for binary mask.

The system learns **3x more inhibitory connections** than excitatory — matching biology.

### 2. Inverse Arousal
Counter-intuitive: self-wire MORE when accuracy is HIGH (exploit), LESS when low (explore via mutation). The connection curve shows biological pruning→growth: 818→374→290→209→474→1610→3555.

### 3. Weights Are (Almost) Unnecessary
No-weight (ternary mask + adaptive scaling) achieves 75%. Binary weight (0.5/1.5) achieves 87.5%. Float32 weight also 87.5%. The topology carries most of the information, weights are just fine-tuning.

### 4. Capacitor Neurons Compensate for Missing Weights
When weights are removed entirely, capacitor threshold neurons (75%) beat leaky_relu (50%). The temporal integration replaces what weights would do.

### 5. Diff as Input is Redundant
The persistent state (decay 0.5) already carries the information that the diff would provide. Removing diff from input saves V neurons and doesn't hurt accuracy (26% vs 28%).

### 6. The 64-Class Wall is Interference
Not a capacity problem — 480 neurons scored same as 160. The issue is that random mutations help some classes while hurting others. Solution needs local credit assignment.

### 7. Graph Architecture is Fundamental
Can't be replaced by standard networks. The ability to add/remove/rewire connections (structural search) is qualitatively different from and superior to parametric search (weight perturbation).

---

## Hardware Feasibility

### Memory per model (80 neurons, 16-class)
| Precision | Size | Fits in |
|-----------|------|---------|
| Float32 | ~34 KB | L2 cache |
| Binary weight (3 bit/conn) | ~330 bytes | L1 cache! |
| No weight (2 bit/conn) | ~158 bytes | Anywhere |

### Swarm estimates (8x RPi4, 32-class)
- 25 cycles/sec per Pi
- 190 total cycles/sec combined
- 32-class solved in ~42 seconds

### Sparse compute opportunity
With 3% neurons active, only active neurons need cache.
2560-neuron model in RAM, but forward pass uses 24KB in L1.

---

## Architecture Diagram

```
INPUT "t" → one-hot [0,0,...,1,...,0]
      │
      ▼ (first tick only)
╔═══════════════════════════╗
║  I/O NEURONS (first V=28) ║ ← shared input+output
╠═══════════════════════════╣
║                           ║
║   FLAT GRAPH (64 internal)║
║   ternary mask (-1/0/+1)  ║
║   binary weight (0.5/1.5) ║
║   leaky_relu activation   ║
║   6 ticks propagation     ║
║                           ║
╠═══════════════════════════╣
║  I/O NEURONS (same V=28)  ║ ← read output here
╚═══════════════════════════╝
      │
      ▼ softmax → prediction "h"
      │
      ▼ compare with actual → diff
      │
  ┌───┴───────────────┐
  │  LEARNING          │
  │  (diff NOT input)  │
  │                    │
  │  Mutation: add/rem/│
  │  rewire/FLIP(30%)  │
  │  Keep if better    │
  │  Revert if worse   │
  │                    │
  │  Self-wire:        │
  │  inverse arousal   │
  └────────────────────┘
```

---

## What Failed
- Binary/ternary activation at scale (threshold too high)
- Soft/continuous masks (too noisy for mutation+selection)
- Pipeline/chained architectures (information bottleneck)
- Sculpt-only/prune-only (irreversible greedy trap)
- Inhibitory neuron TYPES (hurts with random mutation)
- Dense distributed coding (bad with mutation+selection)
- Emergent topology from addresses alone (too coarse)
- Combined "best of everything" (components interfere)

## Open Questions
1. 64-class wall — needs local credit assignment
2. Self-wiring is only 3% of topology changes
3. Scaling to real language tasks (character LM)
4. Ternary weight deployment on ESP32/RPi
5. Ring buffer integration for temporal tasks
6. Reward-modulated Hebbian learning (v19b fusion)
