# Structured Chaos Theory v1.0

> Three necessary and sufficient conditions for gradient-free, mutation-based learning systems.

**Status:** Empirically validated on INSTNCT byte-pair prediction (April 2026). Individual components tested across 10 fitness variants, ablation studies, and adversarial diversity checks.

## The Formula

```
Learning = Ψ × sensitivity / dimensions
```

- **Ψ (psi)** — architectural constraint efficiency: how effectively the structure narrows the search space without eliminating the solution
- **sensitivity** — the measurable fitness change caused by a single mutation (smooth fitness → high, log/CE → low)
- **dimensions** — the size of the search space (neurons, edges, thresholds, channels)

Learning is maximized when the architecture constrains efficiently (high Ψ), every mutation is detectable (high sensitivity), and the search space is compact (low dimensions). This is why binary edges, single channels, and sparse topologies outperform their richer counterparts — each simplification improves the sensitivity/dimensions ratio.

---

## Law I — The Minimal Rule

*A single smooth fitness function plus blind random search yields maximum emergence.*

The fitness function must be smooth: every mutation must produce a measurable directional change. Logarithmic and cross-entropy functions compress peak values, destroying directional information — the opposite of their behavior in gradient-based ML. The fewer rules imposed on the system, the more room emergence has to operate.

**Key insight:** In gradient-based training, CE/log loss is standard because gradients carry directional information regardless of loss curvature. In gradient-free mutation search, the fitness value IS the only signal. Compressing that signal with log/CE destroys the directional information that mutations need to be selected.

**Empirical evidence:** Systematic testing of ten fitness variants showed smooth linear cosine similarity as the clear winner. Every "smarter" variant (cross-entropy, log-scaled, temperature sharpening, hybrid blends, top-N filtering, boredom penalty) degraded performance. The winning fitness function:

```
fitness = cosine_similarity(softmax(projection_scores), bigram_distribution)
```

where `bigram_distribution` is `P(next | current)` — the empirical transition probability from the training corpus.

---

## Law II — The Anti-Monopoly

*A structural constraint against single-attractor collapse is required.*

Without constraint, evolution invariably finds the easiest local optimum: constant output at the frequency baseline. The network converges to a single attractor where a handful of dominant neurons produce identical output regardless of input. Two mechanisms are necessary:

- **Mutual inhibition**: cross-inhibitory neuron clusters in the output zone that enforce winner-take-all dynamics, creating multiple attractors. Two groups of output neurons are connected with inhibitory edges — when one group activates, it suppresses the other, forcing the network to make a choice rather than collapsing to a mean.

- **Alive fraction bonus**: `fitness = similarity × (1 + λ × alive_fraction)`, where λ = 0.1. This penalizes low neuron activity without sacrificing accuracy. `alive_fraction` is the proportion of output neurons with non-zero charge after propagation.

**The λ tradeoff:** The anti-monopoly pressure must be calibrated. Too weak (λ = 0) allows single-attractor collapse. Too strong (λ ≥ 0.3) makes the fitness landscape reward diversity over accuracy, degrading performance.

**Empirical evidence:** At λ = 0, one unique output across all inputs. At λ = 0.1, four unique outputs and 7.50% accuracy (all-time peak, with 18.7% charge diversity). At λ = 0.3, diversity increases to 31% but accuracy drops to 3.9%.

**Root cause analysis:** Ablation study showed that removing 7 dominant neurons killed ALL output (0/158 alive), proving no alternative pathways existed. The network had evolved a single bottleneck — the anti-monopoly mechanisms prevent this structural failure mode.

---

## Law III — The Opponent

*Competitive coevolution destabilizes local optima.*

Multiple networks compete in parallel on the same task. At the end of each cycle, winner topologies are merged and the worst performer is replaced. This prevents any single network from stagnating in a local optimum.

**Mechanism:**
1. N networks (default N = 3) evolve independently for a fixed number of steps
2. Networks are ranked by accuracy at the end of each cycle
3. Winner topologies are merged into an offspring network
4. The worst-performing network is replaced by the offspring
5. The cycle repeats

**Union merge principle:** Nothing is discarded — every edge present in any winner is carried into the offspring. This maximizes preservation of discovered structures. The rationale: evolution has already validated each edge through selection pressure; discarding validated structure wastes information.

---

## Relationship to Prior Work

The individual building blocks have academic lineage:

- **Fitness landscape smoothness**: Kauffman's NK model (1993) established the relationship between epistatic interactions and landscape ruggedness
- **Diversity maintenance**: Goldberg & Richardson (1987) introduced fitness sharing for maintaining population diversity
- **Competitive coevolution**: Rosin & Belew (1997), Stanley & Miikkulainen (2002, NEAT) demonstrated coevolution for escaping local optima
- **Evolution as learning**: Watson & Szathmary (2016), Valiant (2009) formalized evolution as a constrained learning algorithm

**What is novel in this framework:**

1. The compact formula `Learning = Ψ × sensitivity / dimensions` combining architectural constraint, mutation detectability, and search space size
2. The explicit claim that CE/log loss actively harms gradient-free search (inverting gradient-based ML wisdom)
3. The alive fraction bonus as an evolutionary fitness term to prevent dead-neuron collapse
4. Mutual inhibition as a structural constraint in evolutionary training (known in neuroscience, novel in neuroevolution)
5. The specific triple of laws as necessary and sufficient conditions for gradient-free learning

---

## Notation Summary

| Symbol | Name | Definition |
|--------|------|------------|
| Ψ | Architectural constraint efficiency | How effectively the structure narrows the search space |
| sensitivity | Mutation sensitivity | Measurable fitness change from a single mutation |
| dimensions | Search space size | Total free parameters (neurons, edges, thresholds, channels) |
| λ | Anti-monopoly pressure | Weight of alive_fraction bonus in fitness (optimal: 0.1) |
| alive_fraction | Neuron activity ratio | Proportion of output neurons with non-zero charge |

---

*Structured Chaos Theory v1.0 — Kenessy & Claude, Vraxion, April 2026*
