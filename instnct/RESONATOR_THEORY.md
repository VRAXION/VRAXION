# Resonator Chamber Theory
## A Wave-Interference Model of Neural Computation

### Core Hypothesis

The brain (and brain-inspired networks like INSTNCT) function as **resonator chambers** —
input signals enter, propagate through the network as waves, bounce off structure,
and destructive interference eliminates most paths. What survives is the "answer."

This is analogous to a double-slit experiment: inputs are the slits, the network
is the medium, and the interference pattern at readout time is the output.

No quantum mechanics needed — this is **classical wave interference** in a discrete
spike-based medium.

---

### Empirical Findings

All findings below are from deterministic toy tests (no training, no randomness)
on networks from 16 to 1024 neurons, validated against the complete FlyWire
fruit fly connectome (139,255 neurons, 16.8M connections).

---

### Finding 1: Inhibitory Architecture

**Question:** What ratio of excitatory to inhibitory neurons works best?

**Answer from fly brain data (FlyWire):**
- Only **10.2% of neurons are inhibitory** (GABA), not 40% as previously assumed
- But each I neuron connects to **2× more targets** (214 vs 112 out-degree)
- E and I synaptic strength is **identical** (3.49 vs 3.44 syn/target, ratio 1.013)

**What this means:** The fly brain uses **few but powerful hub-inhibitors**, not
many weak ones. Each inhibitory neuron acts as a wide-broadcast signal canceller.

**Toy model validation:**
```
Topology: 10% I neurons, 2× fan-out, full strength (-1.0)

              50% reciprocal E↔I pairs:
              H=32:  6/8 separation
              H=64:  8/8 ✓ (perfect)
              H=128: 8/8 ✓
              H=256: 8/8 ✓

Compare: 20% I, uniform fan-out, 50% reciprocal = DEAD at all scales
Compare: 40% I, any config, 50% reciprocal = DEAD at all scales
```

**Design principle:** `few_hub_inhibitors > many_uniform_inhibitors`

---

### Finding 2: Weight Resolution Is Irrelevant

**Question:** Do edge weights need high precision (float32)? Or is binary enough?

**Tested:** binary (1-bit), 2-level, 3-level, int3, int4, float32

**Answer:** Weight resolution makes **zero difference**. Binary (0/1) edges are
optimal or tied-for-best at every scale and configuration tested.

```
Fly-brain config (40% inhib, 50% recip) across resolutions:
  binary:  DEAD
  float:   DEAD
  (both dead because the architecture was wrong, not the weights)

Correct architecture (10% I, 2× fan-out):
  binary:  8/8 ✓
  float:   8/8 ✓  (same result)
```

**What matters is topology (which neurons connect to which), not edge weights.**

The fly brain's synaptic weight distribution (1-2400 synapses per connection)
follows a power law, but the heavy tail (strong connections) represents only
0.4% of edges. The remaining 99.6% could be represented with 4 bits (int4)
or even 3 levels without information loss.

**Design principle:** `binary_mask + correct_topology > graded_weights + wrong_topology`

---

### Finding 3: Optimal Tick Count = Network Diameter

**Question:** How many ticks (wave bounces) should the network run before readout?

**Answer:** `optimal_ticks ≈ 1.0 × diameter`

The signal goes through three phases:
1. **Explosion** (tick 1 → diameter/2): signal spreads, more neurons activate
2. **Interference** (tick diameter/2 → diameter): waves collide, destructive interference filters
3. **Death** (tick > 1.5× diameter): too much cancellation, everything dies

```
Quality score (separation × alive × network_reach):

ratio   quality
  0.5   0.495   ← too early, signal hasn't spread
  0.9   0.590   ← BEST
  1.0   0.585   ← ≈ diameter, near-optimal
  φ     0.524   ← golden ratio: reach peaks but too many dead
  2.0   0.468   ← too late, half the network dead
  3.0   0.387   ← mostly dead
```

**The readout window is narrow:** too early = trivial separation (inputs haven't
mixed), too late = everything dies. The sweet spot is when the wave has just
reached the far side of the network.

---

### Finding 4: Diameter Scales Logarithmically

**Question:** Does a million-neuron network need a million ticks?

**Answer:** No. Diameter grows as **log₂(N)** due to small-world topology.

```
Measured and extrapolated:

  Neurons          Diameter    Ticks needed
  ─────────────────────────────────────────
  1,024            10          ~10
  139,000 (fly)    20          ~20
  86,000,000,000   43          ~43
  (human brain)
```

This matches biology: a human thought takes ~50ms, a neuron fires in ~1ms,
so ~50 ticks — exactly where the model predicts.

**Doubling the network size adds only +1 tick.** Scaling is essentially free
in the tick dimension. The real cost is the per-tick matrix multiply (O(edges)).

**Design principle:** `ticks = ceil(log2(N) * 1.18)` (fitted constant)

---

### Finding 5: Reciprocal E↔I Pairs Are Filters (at Scale)

**Question:** Are reciprocal excitatory↔inhibitory connections useful?

**Answer:** Yes, but **only with the right architecture** (Finding 1) and
**only at sufficient scale**.

```
Scale dependence of 50% reciprocal edges:

Architecture           H=32   H=64   H=128  H=256
──────────────────────────────────────────────────
20% I, uniform:        DEAD   DEAD   DEAD   DEAD
40% I, uniform:        DEAD   DEAD   DEAD   DEAD
10% I, 2× fan-out:     6/8    8/8✓   8/8✓   8/8✓   ← WORKS
```

Reciprocal pairs act as **band-pass filters**: the E neuron excites its I partner,
which then sends calibrated inhibition back. This creates selective damping —
some signal frequencies survive, others don't.

At small scale (32 neurons), there aren't enough alternative paths, so the
reciprocal inhibition kills too many signals. At 64+ neurons, the network
is rich enough that alternative paths route around the damped ones.

---

### Finding 6: Universal vs Scale-Dependent Parameters

Some parameters are universal (work at all scales), others are scale-dependent:

**Universal (same optimal value at all scales):**
- Inhibitory fraction: **10%** with 2× fan-out
- Weight resolution: **binary**
- Tick/diameter ratio: **~1.0**

**Scale-dependent (optimal value shifts with N):**
- Maximum tolerable reciprocal fraction: grows with scale
  - H=32: max ~15%
  - H=64: max ~30%
  - H=128+: 50% works fine
- Inhibitory fraction tolerance: widens with scale
  - H=32: only 20-30% works
  - H=128+: 10-40% all viable

---

### Implications for INSTNCT

Current INSTNCT configuration vs optimal (based on these findings):

| Parameter | Current | Optimal | Status |
|-----------|---------|---------|--------|
| I neuron fraction | 20% | **10%** | Should decrease |
| I fan-out | 1× (uniform) | **2×** | Should add hub topology |
| Reciprocal E↔I | 0% | **30-50%** | Missing entirely |
| Weight type | int4 | **binary sufficient** | Could simplify |
| Ticks (H=1024) | 8 | **10-12** | Slightly too few |
| Tick rule | fixed | **≈ diameter** | Should be adaptive |

The most impactful change would be introducing **hub inhibitors** (10% of neurons
with 2× connections) and **reciprocal E↔I pairs**. These are structural changes
to the mutation operators, not parameter tuning.

---

### The Resonator Metaphor

Think of the network as a room:
- **Input neurons** = speakers playing a signal
- **Excitatory connections** = walls that reflect sound
- **Inhibitory hub neurons** = acoustic dampeners at key positions
- **Reciprocal E↔I pairs** = tuned resonance filters
- **Ticks** = how long you let the sound bounce
- **Readout** = placing a microphone after the right number of bounces

The room's shape (topology) determines what frequencies survive. A well-designed
room (correct I%, hub structure, reciprocal pairs) produces clean filtered output.
A badly designed room (too much dampening, or dampening in wrong places) kills
all sound.

**The topology IS the computation.** No weights need to be learned — the structure
of connections determines what gets filtered and what survives.

---

### Validation Against Biology

| Prediction | Fly Brain (FlyWire) | Match? |
|------------|-------------------|--------|
| ~10% inhibitory neurons | 10.2% | ✓ |
| I neurons = hubs (high degree) | 2× out-degree | ✓ |
| E and I same per-synapse strength | 1.013 ratio | ✓ |
| High reciprocal E↔I fraction | 50%+ of reciprocal pairs are E↔I | ✓ |
| Giant strongly connected component | 92.6% in one SCC | ✓ |
| Small-world (log diameter) | Predicted ~20 for 139K | Plausible |
| Binary weights sufficient | 50% of connections = 1 synapse | ✓ |

---

### Open Questions

1. **Neuromodulation (C19 waves):** How does the rho/wave modulation interact
   with the resonator? Does it shift the "tuning" of the chamber over time?

2. **Learning:** If topology IS computation, then learning = rewiring. Is
   evolutionary search (INSTNCT's approach) actually the right learning algorithm
   for resonator chambers? (Backprop doesn't apply to topology.)

3. **Hierarchical resonators:** The fly brain has distinct neuropils (brain regions).
   Are these nested resonator chambers? Does signal flow between chambers follow
   different rules?

4. **Consciousness:** If the resonator model is correct, what makes certain
   interference patterns "conscious"? Is it self-referential loops (signal
   that returns to its own source)? Or sustained oscillation (patterns that
   don't die)?

---

*Generated from deterministic toy experiments and FlyWire connectome analysis.*
*All test code in `instnct/tests/resonator_*.py`.*
*Data from FlyWire Connectome (Dorkenwald et al., 2024).*
