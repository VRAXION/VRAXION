# Mutation-Selection Dynamics

> *Structure-level mechanism inside the [Local Constructability Framework](Local-Constructability-Framework). Carries forward and refines the prior "Structured Chaos Theory" content under stricter empirical bounds.*

**Status: Empirically anchored.** Two preregistered experimental phases (Phase A baseline and Phase B confound test) and a 12.6M-row per-candidate log substantiate the structure-level claims under stated boundaries.

---

## Core claim

> **A working substrate is built, not searched.** Local mutation under three organising laws, with selection biased toward useful improvement, accumulates topology that supports productive interference. The framework characterises this construction process and proposes a single per-step measurable quantity, **C_K (local constructability)**.

The substrate is not navigated to a fixed pre-existing optimum; the mutation trajectory is the construction process. Each accepted mutation is a unit of structure added.

---

## The Three Laws

These are the necessary organising principles for productive mutation-selection in this regime. Two of them have direct counterparts in the predecessor "Structured Chaos Theory v1.0"; the third is renamed and broadened.

### Law I — Smoothness (Single Constraint)

> A learning system should face exactly one binding fitness signal at a time; multi-objective compounds prevent attribution and degrade selection to random walk.

Empirical anchor: in the 10-fitness-variant sweep that produced the original "champion" smooth-cosine fitness, compound objectives uniformly underperformed. Cross-entropy and log-scaled variants compress σ_μ below the detection threshold.

### Law II — Anti-Monopoly

> No single neuron, pathway, or attractor basin should be permitted to dominate the network's computational capacity. Single-attractor convergence is the dominant failure mode of unconstrained mutation-selection on this substrate.

Empirical anchor: the 7-dominant-neuron ablation (April 2026) collapsed the network to one attractor. The mutual-inhibition seed and the λ=0.1 alive-fraction penalty in the fitness function both implement Law II.

### Law III — Opponent

> Productive learning requires opponent dynamics — excitatory and inhibitory forces that keep each other in check, and constructive and destructive operations that bound each other.

This subsumes the prior "Competitive Coevolution" framing as one specific instantiation. Concrete observed instances:

- E/I balance at the signal level (excitation vs inhibition; see [Interference Dynamics](Interference-Dynamics)).
- Grow-vs-prune cycle at the structure level (the `bytepair_proj` fixture; see Phase A boundary findings below).
- Multi-population breeding with union merge (Structured Chaos's "competitive coevolution"; not exercised in Phase A or Phase B).

---

## Primary measure: C_K

Per-step expected useful improvement, normalised by the cost of evaluating *K* jackpot candidates:

$$C_K(g)\;=\;\frac{\mathbb{E}\bigl[\max\bigl(0,\;\max_{i \le K} \Delta U_i \;-\; \varepsilon\bigr)\bigr]}{\mathbb{E}[\mathrm{cost}_K]}$$

Operational reconstruction: from a per-candidate log emitting `(step, candidate_id, operator_id, before_U, after_U, Δ_U, accepted, eval_ms)`, compute window-ratio aggregates over rolling step windows.

**C_K is a diagnostic, not a frozen scalar objective.** Phase B shows C_K(B0) ≈ C_K(B1) at H=384, even though peak accuracy differs by ~2 pp; the difference is explained by accumulation length, not per-step productivity.

---

## Explanatory decomposition (hypothesis)

The framework proposes the following multiplicative decomposition. It is **a hypothesis under regression test**, not an axiom:

$$C_K \;\stackrel{?}{\approx}\; \frac{V_{\mathrm{raw}}\,\cdot\,M_{\mathrm{pos}}\,\cdot\,A\,\cdot\,I_{\mathrm{proxy}}}{D_{\mathrm{eff}}\,\cdot\,\mathrm{cost}_{\mathrm{eval}}\,\cdot\,R_{\mathrm{neg}}}$$

Component definitions:

| Component | Definition | Standard analog |
|---|---|---|
| `V_raw` | `P(ΔU > ε)` over all candidates (not just accepted) | beneficial mutation rate, positive DFE mass |
| `M_pos` | `E[ΔU | ΔU > ε]` | positive tail of the distribution of fitness effects |
| `R_neg` | `E[|ΔU| | ΔU < −ε]` | deleterious tail / destructive risk |
| `A` | composite anti-collapse index = `f_active · (1 − exp(−Var_i[H_i])) · stable_rank_norm` | novelty/diversity measures (Lehman-Stanley, MAP-Elites) |
| `I_proxy` | composite I/O retention = panel of `dCor`, collision rate, CKA, output entropy | input-output mutual information; raw MI is undersampled at our N |
| `D_eff` | empirical sensitivity: `E[‖Δoutput‖ : 1 random mutation]` | effective rank of mutation Jacobian; not the raw parameter count |
| `cost_eval` | wall-clock per candidate evaluation | standard ES cost normalisation |

The decomposition is evaluated post-hoc by regressing `log C_K` against the log-components across the 25 Phase B runs. Adequate fit (R² > 0.8) supports the form; poor fit triggers revision.

---

## Empirical anchors

### Phase A — 30-cell baseline

```
fixture=mutual_inhibition (Law I + II + flat training):
  H=128 (n=5)   3.76 ± 0.91 % peak     accept 78%   alive 0.72
  H=256 (n=5)   5.28 ± 1.79 %          accept 41%   alive 0.44
  H=384 (n=5)   3.52 ± 1.14 %          accept 17%   alive 0.45

fixture=bytepair_proj (Law I + grow-prune cycle):
  H=128 (n=5)   5.24 ± 1.07 %          accept 26%   alive 0.66
  H=256 (n=5)   3.62 ± 0.81 %          accept 8%    alive 0.19
  H=384 (n=5)   3.16 ± 2.33 %          accept 0.6%  alive 0.05  ← bimodal, knife-edge
```

Crossover finding: `bytepair_proj > mutual_inhibition` at H=128 (Δ=+1.48 pp, p≈0.05); `mutual_inhibition > bytepair_proj` at H=256 (Δ=+1.66 pp). The "champion recipe" is H-specific.

### Phase B — 25-cell confound test at H=384

```
B0 baseline                3.52 ± 1.14 %   ↳ replicates Phase A H=384 MI
B1 2× horizon (40 k step)  5.50 ± 1.47 %   ← recovers H=256 reference
B2 2× jackpot (K=18)       3.26 ± 2.93 %   heavy-tailed, outlier-driven
B3 2× ticks (12)           3.16 ± 2.07 %   higher destructive risk (R_neg ↑ ~2.7×)
B4 input scatter           2.00 ± 1.71 %   collapses signal coherence
```

Welch B1 vs B0: t = 2.37, df ≈ 8, p = 0.047. **Directional support for the training-horizon-confound interpretation; not formally Bonferroni-significant** at α/4 = 0.0125. Replication at n ≥ 10 required for strict significance.

### Per-operator findings (operator schedule misalignment)

Across 12.6M candidate rows, operator productivity (V_raw × M_pos) is markedly non-uniform:

| Operator | Schedule % | Productivity share | Note |
|---|---|---|---|
| `theta` (threshold) | 5% | **22.3%** | most productive; under-allocated |
| `channel` | 10% | 15.9% | second; well-allocated |
| `loop3` | 5% | 14.1% | under-allocated |
| `add_edge` | 22% | 7-8% | over-allocated |
| `projection_weight` | 5% | < 0.01% | **inert at H=384** |

The schedule is plausibly misaligned. A theta- and channel-heavy schedule with `projection_weight` removed is a Phase C hypothesis; it has not been measured and is not claimed as an improvement here.

---

## Open hypotheses

- **Phase B.1** — `accept_ties × horizon × seed` ablation (2 × 3 × 5 = 30 cells) on `mutual_inhibition` only. Tests whether the B1 horizon effect survives strict Bonferroni at n ≥ 10, whether 80 k steps lets H=384 surpass H=256, and whether neutral-accept policy substantively affects peak. Pre-registration to be filed before launch.
- **Phase C** — operator schedule retuning (theta-heavy, channel-heavy, drop `projection_weight`). Pre-registered as a predictive improvement test, not a post-hoc narrative.
- **`bytepair_proj` collapse ablation.** Separate from Phase B.1, because the failure mode at H=384 is grow-prune-aggressiveness, not horizon. Candidate knobs: prune rate, keep-best restore, minimum-alive constraint.

---

## Boundaries

- The C_K decomposition is *partly* validated. Component-level reconstruction works; closed-form predictivity across regimes has not been demonstrated.
- "B1 wins" is directional, not Bonferroni-significant. Don't overclaim.
- Per-operator findings reflect distributions on H=384 specifically; cross-H operator productivity will likely differ. Phase A had no per-candidate logging.

---

## Read next

- [Local Constructability Framework](Local-Constructability-Framework) — umbrella.
- [Interference Dynamics](Interference-Dynamics) — signal-level mechanism this structure supports.
- [Constructed Computation](Constructed-Computation) — what emerges in regime.
- [Research Process & Archive](Timeline-Archive) — full chronology including the predecessor "Structured Chaos Theory v1.0".
