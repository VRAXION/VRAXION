# Constructed Computation

> *Emergence-level claims of the [Local Constructability Framework](Local-Constructability-Framework) — what arises when the signal and structure mechanisms operate within their effective regimes.*

**Status: Empirical and bounded.** This page covers what we can measure as outcomes. Cognition-level claims that exceed our measurements are kept in a [separate appendix](Cognitive-Emergence-Speculative) and labelled speculative.

---

## Core claim

> **When [interference dynamics](Interference-Dynamics) and [mutation-selection dynamics](Mutation-Selection-Dynamics) both operate within their effective regimes, the substrate produces measurable computation: above-baseline task accuracy, above-baseline representational capacity, and above-baseline distinctness across inputs.**

Three parts:

1. The substrate has been *constructed* (by mutation-selection) into a state where interference is productive rather than destructive.
2. That state is then capable of actual computation on its task.
3. "Effective regime" is jointly conditioned on the recipe; outside the regime, the same H or the same training schedule produces collapse, bimodality, or below-baseline performance.

The emergence is the **combined outcome** of two mechanisms operating in regime, not a property of either alone.

---

## Operational measures

| Class | Measurement | Where computed |
|---|---|---|
| Task | `peak_acc`, `final_acc` | every run, summary log |
| Capacity | `kernel_rank` of charge-on-probe matrix | Phase B `panel_summary.json` |
| Capacity | `separation_SP` (input-distance vs output-distance preservation) | Phase B `panel_summary.json` |
| Capacity | `participation_ratio` over tick × neuron activity | Phase B `panel_summary.json` |
| Distinctness | `unique_predictions` over a probe set, `collision_rate` | Phase B `panel_summary.json` |
| I/O retention | distance correlation `dCor(input, output)`, linear `CKA` | Phase B `panel_summary.json` |
| Coverage | output-zone `f_active`, `Var_i[H_output]` | Phase B `panel_summary.json` |

Three of the proposed measures (`D_eff_sensitivity`, `motif_z3`, `branching_σ`) are not yet computed. They are addable post-hoc from the saved final checkpoints; they are not gating any current claim.

---

## Empirical anchors

### Recipe-dependent H-profile

The signature emergence finding is that **the H-profile is recipe-specific**. This is the cleanest cross-fixture observation in the dataset.

```
                              peak_acc%
H        mutual_inhibition    bytepair_proj   verdict
128       3.76 ± 0.91          5.24 ± 1.07     bytepair_proj wins (Δ +1.48, p≈0.05)
256       5.28 ± 1.79          3.62 ± 0.81     mutual_inhibition wins (Δ +1.66)
384       3.52 ± 1.14          3.16 ± 2.33     comparable means; bytepair_proj bimodal (range 0.0–6.0 %)
```

The "champion recipe" at one H is not the champion at another. At fixed-H sweeps with one fixture, this looks like an architectural verdict; across fixtures, it does not.

### Capacity correlates with peak across arms (Phase B, H=384)

In the 25-cell Phase B, capacity proxies (`kernel_rank`, `separation_SP`, `participation_ratio`) are highest in B1 (the arm that also reaches highest `peak_acc`) and collapse in B3/B4 (arms that fail). The capacity panel and the task panel are not independent.

### Bimodality at the edge of regime

`bytepair_proj` H=384 produces five seeds with peaks of `[6.00, 0.00, 2.30, 2.70, 4.80]`. One seed reaches a working substrate, one collapses entirely, three sit in between. The grow-prune cycle is in a knife-edge regime at this H: the same recipe with the same input produces qualitatively different outcomes depending on initial conditions. This is itself a finding about the emergence boundary.

---

## What the framework lets us claim

- **Useful task computation requires both mechanisms.** A substrate with productive interference but no construction (i.e. fixed random topology) cannot be tuned by gradient-free mutation; a substrate with construction but degenerate signal dynamics (B3, B4 in Phase B) does not produce useful predictions despite mutation effort.
- **Emergence is conditional on regime.** A network at H=384 with `mutual_inhibition` recipe and 20 k step budget is *not* the same emergent system as the same H=384 with 40 k steps. Treating the architecture as the only variable misattributes capacity.
- **Capacity proxies and peak accuracy track together within an arm but not across recipes.** Cross-recipe capacity comparison (e.g. `mutual_inhibition` H=256 vs `bytepair_proj` H=128) is sensitive to the readout mechanism; raw `kernel_rank` differences do not directly imply task performance differences.

---

## What the framework does not let us claim

- That emergence is **architecturally explained** without specifying the training recipe. Any single-fixture H-sweep is a confound.
- That a network's capacity panel **predicts** peak accuracy in a new recipe. Cross-recipe extrapolation has not been tested.
- That **cognition** is achieved. We measure task performance and capacity proxies on a 397-class byte-pair prediction task. A claim about cognitive emergence in a stronger sense (multi-task generalisation, transfer, abstraction) requires evidence on more than one task.

---

## Open questions

- Cross-recipe **capacity normalisation**: does there exist a transformation of `kernel_rank` or `separation_SP` that is comparable across `mutual_inhibition` and `bytepair_proj`? Without this, capacity panels are within-recipe.
- The `bytepair_proj` H=384 **knife-edge bimodality**: is this a discontinuous phase transition (suggesting a control parameter exists), or a continuous distribution that happens to land in two basins under the current setup? More seeds (n ≥ 20) needed.
- **Compute-economic emergence**: B1's 5.50 % at H=384 used ~3× the compute of `mutual_inhibition` H=256's 5.28 %. Architecturally H=384 reaches higher peak, but only with more budget. The compute-normalised emergence boundary is a separate question.

---

## Read next

- [Local Constructability Framework](Local-Constructability-Framework) — umbrella.
- [Interference Dynamics](Interference-Dynamics) — signal mechanism (one of two prerequisites).
- [Mutation-Selection Dynamics](Mutation-Selection-Dynamics) — structure mechanism (the other).
- [Speculative Extension — Cognitive Emergence](Cognitive-Emergence-Speculative) — extrapolations beyond what is measured here.
- [INSTNCT Architecture](INSTNCT-Architecture) — the substrate.
