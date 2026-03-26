<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Resonator Theory

> **Inference emerges as the fixed point of destructive interference.**

## What This Page Is

This page is the public-facing summary of the Resonator Chamber hypothesis: the theoretical model behind INSTNCT's architecture. It explains why a self-wiring graph with passive I/O, persistent state, and mutation-selection training can compute — and what biology says about the same design.

For the full technical treatment with all toy model results and validation code: [`instnct/RESONATOR_THEORY.md`](https://github.com/VRAXION/VRAXION/blob/main/instnct/RESONATOR_THEORY.md).

## Core Hypothesis

The network acts as a **resonator chamber** — a wave-interference medium. Input signals enter through fixed projections, propagate as spike waves through the hidden graph, and destructive interference eliminates most paths. What survives at readout is the computation.

No quantum mechanics is needed. This is classical wave interference in a discrete spike-based medium.

## The Resonator Metaphor

Think of the network as a room:

- **Input neurons** = speakers playing a signal
- **Excitatory connections** = walls that reflect sound
- **Inhibitory hub neurons** = acoustic dampeners at key positions
- **Reciprocal E-I pairs** = tuned resonance filters
- **Ticks** = how long you let the sound bounce
- **Readout** = placing a microphone after the right number of bounces

The room's shape (topology) determines what frequencies survive. A well-designed room produces clean filtered output. A badly designed room kills all sound.

**The topology IS the computation.** The structure of connections determines what gets filtered and what survives.

## Six Empirical Findings

All findings are from deterministic toy tests (no training, no randomness) on networks from 16 to 1024 neurons, validated against the complete FlyWire fruit fly connectome (139,255 neurons, 16.8M connections).

### 1. Inhibitory Architecture

**10% inhibitory hub neurons** with 2x fan-out is optimal — not 40% uniform. Each I neuron acts as a wide-broadcast signal canceller. FlyWire data: 10.2% inhibitory (GABA), 2x out-degree, identical per-synapse strength.

### 2. Weight Resolution Is Irrelevant

**Binary weights (0/1) are sufficient.** Topology determines computation, not edge precision. Tested: binary, 2-level, 3-level, int3, int4, float32 — all equivalent when the architecture is correct.

### 3. Optimal Ticks = Network Diameter

Signal passes through three phases: explosion (spreading), interference (filtering), death (over-cancellation). The sweet spot is `ticks ≈ 1.0 × diameter` — when the wave has just reached the far side.

### 4. Diameter Scales Logarithmically

Diameter grows as **log₂(N)** due to small-world topology. Doubling the network adds only +1 tick. Extrapolated: 1024 neurons → ~10 ticks, 139K (fly) → ~20, 86B (human) → ~43.

### 5. Reciprocal E-I Pairs Are Filters (at Scale)

Reciprocal excitatory-inhibitory pairs act as band-pass filters: some frequencies survive, others don't. They require the correct hub-inhibitor architecture (Finding 1) and sufficient scale (H >= 64).

### 6. Universal vs Scale-Dependent Parameters

**Universal** (same at all scales): 10% inhibitory fraction, binary weights, tick/diameter ≈ 1.0.
**Scale-dependent** (shifts with N): maximum reciprocal fraction, inhibitory fraction tolerance.

## Biology Validation

| Prediction | FlyWire Data | Match |
|---|---|---|
| ~10% inhibitory neurons | 10.2% | Yes |
| I neurons = hubs (high degree) | 2x out-degree | Yes |
| E and I same per-synapse strength | ratio 1.013 | Yes |
| High reciprocal E-I fraction | 50%+ of reciprocal pairs are E-I | Yes |
| Giant strongly connected component | 92.6% in one SCC | Yes |
| Small-world (log diameter) | Predicted ~20 for 139K | Plausible |
| Binary weights sufficient | 50% of connections = 1 synapse | Yes |

## Implications for INSTNCT

| Parameter | Current INSTNCT | Theory-Optimal | Status |
|---|---|---|---|
| I neuron fraction | 20% | **10%** | Should decrease |
| I fan-out | 1x (uniform) | **2x** | Should add hub topology |
| Reciprocal E-I | 0% | **30-50%** | Missing entirely |
| Weight type | int4 | **binary sufficient** | Could simplify |
| Ticks (H=1024) | 8 | **10-12** | Slightly too few |
| Tick rule | fixed | **≈ diameter** | Should be adaptive |

The most impactful changes would be introducing **hub inhibitors** and **reciprocal E-I pairs** — structural changes to the mutation operators, not parameter tuning.

## Open Questions

1. **Neuromodulation (C19 waves):** How does rho/wave modulation interact with the resonator? Does it shift the chamber's tuning over time?
2. **Learning:** If topology IS computation, then learning = rewiring. Is evolutionary search the right algorithm for resonator chambers?
3. **Hierarchical resonators:** The fly brain has distinct neuropils. Are these nested resonator chambers?
4. **Consciousness:** What makes certain interference patterns self-referential?

## Test Code

- [`instnct/tests/resonator_toy_deterministic.py`](https://github.com/VRAXION/VRAXION/blob/main/instnct/tests/resonator_toy_deterministic.py) — deterministic toy model
- [`instnct/tests/resonator_multi_scale.py`](https://github.com/VRAXION/VRAXION/blob/main/instnct/tests/resonator_multi_scale.py) — multi-scale sweep (32–512 neurons)
- [`instnct/tests/resonator_weight_resolution.py`](https://github.com/VRAXION/VRAXION/blob/main/instnct/tests/resonator_weight_resolution.py) — weight resolution test

*Data from FlyWire Connectome (Dorkenwald et al., 2024).*

## Read Next

- [VRAXION Home](Home)
- [INSTNCT Architecture](INSTNCT-Architecture)
- [Validated Findings](Validated-Findings)
- [Engineering Protocol](Engineering)
- [Project Timeline](Release-Notes)
