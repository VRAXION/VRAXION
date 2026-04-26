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

| H | mutual_inhibition (n=5) | bytepair_proj (n=5) | Δ (bp − MI) pp | Welch t | p (two-tail, df ≈ 8) |
|---|---|---|---|---|---|
| 128 | 3.76 ± 0.91 | 5.24 ± 1.07 | +1.48 | 2.36 | 0.046 |
| 256 | 5.28 ± 1.79 | 3.62 ± 0.81 | −1.66 | 1.89 | 0.111 |
| 384 | 3.52 ± 1.14 | 3.16 ± 2.33 | −0.36 | 0.31 | 0.764 |

`bytepair_proj` H=384 is bimodal: peaks `[6.00, 0.00, 2.30, 2.70, 4.80]`, std 2.33pp. p-values are uncorrected; with Bonferroni correction over 3 cross-fixture tests (α/3 ≈ 0.017), no comparison reaches strict significance. The directional crossover (bytepair_proj > MI at H=128, MI > bytepair_proj at H≥256) is suggestive but underpowered at n=5.

The "champion recipe" at one H is not the champion at another. At fixed-H sweeps with one fixture, this looks like an architectural verdict; across fixtures, it does not.

### Capacity correlates with peak across arms (Phase B, H=384)

In the 25-cell Phase B, capacity proxies (`kernel_rank`, `separation_SP`, `participation_ratio`) are highest in B1 (the arm that also reaches highest `peak_acc`) and collapse in B3/B4 (arms that fail). The capacity panel and the task panel are not independent.

### Architecture rehabilitation under combined conditions (Phase B.1, H=384)

The Phase B.1 30-cell horizon × tie-policy ablation found that the H=384 substrate **can exceed the Phase A H=256 reference (5.28%)** under the combined conditions of long horizon (80k steps) AND Zero-Drive acceptance (`accept_ties=true`):

```
H=384 + 80k + ties:  6.78 ± 1.52% peak (max-seed 8.50%)   ← beats H=256
H=384 + 80k + strict: 5.60 ± 1.58%                        ← matches H=256
H=384 + 20k + ties:   4.50 ± 1.96%                        ← below H=256
```

This is **architecture rehabilitation under combined conditions** — neither horizon alone nor neutral acceptance alone reaches the rehabilitated peak. The implication for the framework is *not* that H=384 is intrinsically a better architecture, but that the H-profile previously interpreted as an architectural ceiling reflected the joint constraint of the (H, horizon, acceptance-policy) operating point.

Two caveats limit the strength of this claim:
- **Statistical power**: n=5 per cell. The 80k+ties vs 20k+ties contrast has Welch p ≈ 0.076 (not Bonferroni-significant). Replication at n ≥ 10 is needed for strict claims.
- **Tested-plateau bound**: strict acceptance plateaus *within the 80k tested range* (40k strict 5.50%, 80k strict 5.60%, p=0.92), but we cannot rule out further gains beyond 80k.

### H-dependence of (K, policy) — Phase D2 cross-H verdict (2026-04-26)

The Phase D1 result (K=9 strict is the optimal aperture at H=384, n=5, peak mean 5.50%) was tested for H-generalization in Phase D2 by re-running the full (K, policy) grid at H=128 and H=256 with n=5 per cell (see `docs/research/PHASE_D2_CROSS_H_VERDICT.md`).

**Per-H winners (n=5):**

| H | winner K | winner policy | peak mean (%) | peak std |
|---:|---:|---|---:|---:|
| 128 | 9 | strict | 4.62 | 0.98 |
| 256 | 9 | strict | 5.28 | 1.79 |
| 384 | 9 | strict | 5.50 | 1.47 |

**K=9 strict generalizes.** The same `(K=9, strict)` aperture point wins at every tested H. Peak accuracy increases monotonically with H under this aperture, in contrast to the Phase A inverted-U with peak at H=256 — i.e. the H=256 peak was a recipe artefact of the default `(K, policy)`, not an architectural ceiling.

**Sub-saturated K shows H-dependent ranking.** At K=1 and K=3 the policy winner shifts across H:

- **K=1 ranking flips between H=128 and H≥256.** H=128 strict wins (4.00 vs 3.74 ties, +0.26pp); H=256 ties wins (3.72 vs 3.48 strict, +0.24pp); H=384 ties wins decisively (4.02 vs 1.92 strict, +2.10pp). Mechanistic candidate: per-candidate `p_pos` is higher at smaller H (denser positives), so a single-candidate jackpot occasionally lands a useful mutation at H=128 and uniformly starves at H=384.
- **K=3 has a softer gradient.** H=128 ties wins (+0.32pp), H=256 essentially tied (+0.02pp strict), H=384 strict wins (+0.46pp). Same activation-function point, opposite winners at the H extremes.
- **K=9 saturates.** Strict wins at all three H (+0.18 / +0.74 / +0.20pp). At the saturated jackpot, the ties valve provides ~99% accept-rate drift on top of an already permissive (17–78%) strict accept regime — it substitutes plateau wandering for useful selection.

**Framework consequence.** The "Acceptance Aperture" `A(K, τ, s)` is K-independent only in the saturated regime; in the sub-saturated K range it is a function of H as well, i.e. effectively `A(K, τ, s; H)` for K ≲ 3. None of these comparisons reach Bonferroni-corrected significance at n=5; the directional claim is robust across both the D1 H=384 and D2 H={128,256} sweeps.

A precursor n=3 H=128 sandbox pilot motivated the formal D2 design and is preserved at tag `archive/research-sandbox-h128-d1-20260426`. The pilot's K=9 ties-wins reading at n=3 was not reproduced at n=5 (D2 finds K=9 strict wins at H=128 too); it appears to have been a small-n artefact.

### Bimodality at the edge of regime

`bytepair_proj` H=384 produces five seeds with peaks of `[6.00, 0.00, 2.30, 2.70, 4.80]`. One seed reaches a working substrate, one collapses entirely, three sit in between. The grow-prune cycle is in a knife-edge regime at this H: the same recipe with the same input produces qualitatively different outcomes depending on initial conditions. This is itself a finding about the emergence boundary.

### CSP-clustering interpretation (analogy, not proof)

This bimodal seed-outcome pattern is *structurally analogous* to the **dynamic-threshold transition in random constraint-satisfaction problems** (Mézard, Parisi, Zecchina, *Science* 2002; Mertens, Mézard, Zecchina 2006; Liao et al. 2024 [arXiv:2407.20724] for DNN-loss-landscape RSB analogy), where the solution space fragments into exponentially many isolated clusters near the SAT/UNSAT boundary. In that regime:

- Backbone fraction (variables fixed across all solutions) becomes large near the SAT/UNSAT threshold (Monasson, Zecchina, Kirkpatrick, Selman, Troyansky, *Nature* 1999, "Determining computational complexity from characteristic 'phase transitions'"), and bimodal solver-outcome distributions are observed near that boundary.
- "Frozen variables" cause solvers to either succeed or get stuck — the same shape as our `[6.0, 0.0, 2.3, 2.7, 4.8]`.
- Bouchaud's trap model (1992) predicts trapping time ∼ exp(barrier/T), so deep-trap seeds (e.g. our seed=1042 0.0% case) are essentially unreachable on a fixed budget — exactly what we observe.
- The **training-horizon recovery** (Phase B, B1: 5.50% at 40k steps vs 3.52% at 20k) is consistent with the CSP picture: longer search time allows escape from isolated clusters. Background on annealing limits in glassy regimes: Angelini & Ricci-Tersenghi 2022 (arXiv:2206.04760).

This is a structural analogy, not a proof. Two falsifiable post-hoc tests on existing data would tighten the connection (no new compute required):

1. **Avalanche size distribution** from the Phase B candidate logs. If P(s) ~ s^{−3/2} with branching ratio σ ≈ 1 specifically at H=256 (where variance peaks), this is the Beggs–Plenz (2003) self-organised-criticality signature.
2. **Two-time fitness correlation** C(t_w, t) from training logs. If the correlation depends on the ratio t/t_w (aging), the system is glassy in the Bouchaud sense; if it depends only on t (stationary), it is not.

Until these are run, the framework states the analogy as a guiding interpretation, not a validated mechanism. We do not currently use the language of "phase transition" or "edge of chaos" in primary claims.

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
