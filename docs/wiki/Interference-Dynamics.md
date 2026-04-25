# Interference Dynamics

> *Signal-level mechanism inside the [Local Constructability Framework](Local-Constructability-Framework).*

**Status: Mechanism description, partially evidenced.** The signal-level component of the framework. Inherits the destructive-interference thesis from the prior "Theory of Thought" page, narrowed to claims that current data supports.

---

## Core claim

> **Inference is the residual pattern that survives destructive cancellation in a recurrent spiking substrate.**

A single token-processing pass:

1. **Inject.** Input is encoded and projected into the input zone of the recurrent substrate.
2. **Propagate.** Signal spreads through the directed graph for a fixed tick window.
3. **Collide.** Excitatory and inhibitory contributions meet on shared targets; some reinforce, many cancel.
4. **Gate.** Phase-channel structure makes some ticks easier or harder, so not every path competes under identical timing.
5. **Read out.** The surviving state pattern is projected to the readout surface and interpreted as inference.

The thesis is **not** that recurrent inhibition exists. It is that **the answer is the residue that remains after conflicting paths have been suppressed**, not the sum of everything that fired.

---

## Mechanism details

| Element | Concrete realisation in INSTNCT |
|---|---|
| Substrate | binary ±1 polarity edges, sparse topology, edge cap = 7%·H² |
| Time | 6 ticks of propagation, single phase channel |
| Threshold | per-neuron θ ∈ [0, 7], fires when accumulated charge crosses θ |
| Cancellation | inhibitory edges subtract from target charge; below-θ states are silent |
| Read-out | output-zone charge vector × Int8 projection → softmax over 397 classes |

Compact spike-decision form (current implementation; not a permanent fixture):

```
charge * 10 >= (theta + 1) * PHASE_BASE[(tick + 9 - channel) & 7]
```

This expresses an accumulated signal, firing threshold, temporal preference, and current phase schedule in one local decision. The mechanism-level claim does not depend on this specific equation.

---

## Resonator framing (subordinate metaphor)

The resonator picture is an explanatory lens, not a rival theory:

- input neurons are the speakers,
- excitatory paths are reflective surfaces,
- inhibitory paths are dampeners,
- reciprocal loops are tuned filters,
- ticks are how long signal is allowed to bounce,
- readout is where you place the microphone.

What survives the bouncing is the residue. The metaphor matters because it makes one thing legible: **topology shapes what survives**.

No quantum claim is intended. The picture is classical interference inside a discrete spike-style medium.

---

## What we measure

These are diagnostics for the signal level. They do not, individually, prove the destructive-interference claim — they constrain it.

| Measurement | What it indicates | Where measured |
|---|---|---|
| `accept_rate` | per-step viable-mutation density (proxy for signal sensitivity) | every run, summary log |
| `R_neg` | average magnitude of destructive (Δ < −ε) candidates — the "scariness" of bad moves | Phase B candidate logs |
| `alive_frac` | fraction of output neurons that fire on probe inputs — non-zero readout coverage | every run, panel summary |
| `kernel_rank`, `separation_SP` | functional capacity of the residual representation | Phase B panel summary |
| Derrida slope | (proposed) Hamming divergence of two 1-bit-perturbed inputs over T ticks | future experiment, gated for "edge-of-chaos" language |

---

## Empirical evidence

- Phase B's B3 arm (2× ticks) is *correlated* with arm-mean `R_neg` ~2.7× baseline and a worst-seed event with R_neg ≈ 0.16. We report this as co-occurrence; the causal claim that "deeper propagation produces destructive oversensitivity" remains a hypothesis (see § Open questions). Phase D's per-arm KS-test on ΔU histograms will provide the first direct evidence.
- Phase B's B4 arm (input scatter) is *correlated* with `alive_frac` ~0.05 and `accept_rate` ~1%. The simplest reading — "signal coherence depends on a focused input channel" — is consistent with the data but not *proved* by it; alternative explanations (including operator-schedule interaction at the scattered embedding, or a bottleneck in the readout) have not been ruled out.
- Per-input `unique_predictions` and the four-pair adversarial check distinguish multi-attractor from single-attractor regimes; the original mutual-inhibition seeding produces multi-attractor outputs in trained nets.

These do not prove the destructive-interference thesis as a general claim. They are consistent with it.

---

## Open questions

- **Derrida-style perturbation test.** Slope of log-Hamming-distance between two near-identical inputs over the 6-tick window, computed across H. A slope crossing 1 specifically at a candidate "critical" H would warrant edge-of-chaos language; until measured, that language is not used here.
- **Causal influence vs co-occurrence.** Cancellation is hypothesised to be the *productive* mechanism; current evidence shows it is *present*. A targeted ablation (suppress mutual inhibition during inference, measure task-performance drop) would tighten the claim.
- **Information retention.** dCor(input, output) and CKA values are reported in panel summaries but not yet compared across H-conditions in a controlled cross-H sweep.

---

## Read next

- [Local Constructability Framework](Local-Constructability-Framework) — umbrella.
- [Mutation-Selection Dynamics](Mutation-Selection-Dynamics) — how the substrate that supports productive interference is built.
- [Constructed Computation](Constructed-Computation) — what computation emerges when interference and selection both operate within regime.
- [INSTNCT Architecture](INSTNCT-Architecture) — implementation surface.
