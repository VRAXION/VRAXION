# Mutation-Selection Dynamics

> *Structure-level mechanism inside the [Local Constructability Framework](Local-Constructability-Framework). Carries forward and refines the prior "Structured Chaos Theory" content under stricter empirical bounds.*

**Status: Empirically anchored for Laws I + II; Law III is a structural hypothesis pending direct test.** Two preregistered experimental phases (Phase A baseline and Phase B confound test) and a 12.6M-row per-candidate log substantiate Laws I and II at H=384 specifically. Per-operator findings reflect H=384 distributions; cross-H operator productivity is extrapolated, not measured (Phase A had no per-candidate logging). Law III is named for completeness and structural symmetry but is not separately validated by Phase A or Phase B.

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

Crossover finding (Welch t-tests, n=5 per cell, df ≈ 8, *uncorrected for multiple comparisons*; interpret cautiously):

| H | Δ (bytepair − MI) pp | t | p (two-tailed) |
|---|---|---|---|
| 128 | +1.48 | 2.36 | 0.046 |
| 256 | −1.66 | 1.89 | 0.111 |
| 384 | −0.36 | 0.31 | 0.764 |

At α=0.05 uncorrected, only H=128 reaches nominal significance; with Bonferroni-correction over 3 comparisons (α/3 ≈ 0.017), no comparison passes. The "champion recipe" is H-specific *directionally*; we do not claim it formally significant at strict thresholds.

### Phase B — 25-cell confound test at H=384

```
B0 baseline                3.52 ± 1.14 %   ↳ replicates Phase A H=384 MI
B1 2× horizon (40 k step)  5.50 ± 1.47 %   ← recovers H=256 reference
B2 2× jackpot (K=18)       3.26 ± 2.93 %   heavy-tailed, outlier-driven
B3 2× ticks (12)           3.16 ± 2.07 %   accept rate ↓ to 8% (8 vs B0 17%)
B4 input scatter           2.00 ± 1.71 %   collapses signal coherence
```

Welch tests (n=5 per arm, df ≈ 8, vs B0):

| Arm | Δ peak pp | t | p | verdict |
|---|---|---|---|---|
| B1 (2× horizon) | +1.98 | 2.37 | 0.047 | directional, not Bonferroni-significant at α/4 = 0.0125 |
| B2 (2× jackpot) | −0.26 | −0.18 | 0.86 | inconclusive; high variance σ=2.93 (outlier-driven) |
| B3 (2× ticks)   | −0.36 | −0.34 | 0.74 | inconclusive |
| B4 (input scatter) | −1.52 | −1.65 | 0.14 | inconclusive (directionally negative) |

**B1 is the only arm with directional support** for the training-horizon-confound interpretation. Replication at n ≥ 10 required for strict significance. **Per-arm separability**: B3's mean R_neg(accepted) is ~2.7× B0's at the arm level (range 1.46–3.25× across operators); this is reported as a *correlate* of the B3 outcome, not the *cause* — disambiguation requires direct perturbation testing (Phase D / future work).

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

## Acceptance Aperture and the Gaussian Null Model

The acceptance rule is not a tuning convenience — it defines the topology of the directed reachability graph over network states. We name three regimes:

- **Strict (ε < 0):** accept ΔU > 0 only.
- **Neutral / Zero-Drive Search (ε = 0):** accept ΔU ≥ 0. Established under "neutral drift on neutral networks" (van Nimwegen, Crutchfield, Huynen 1999; Wagner 2005).
- **Tolerant / Threshold-Drive Search (ε > 0):** accept ΔU ≥ −ε. Established as "Threshold Accepting" (Dueck & Scheuer 1990).

The acceptance rule alters the topology of the reachable graph. The *magnitude* of that alteration is empirical and substrate-specific; theoretical upper bounds (e.g. permutation-equivalent configurations under neutral) characterise the reachable *phenotype-equivalence class*, not the *operator-reachable subset*.

**Primary measured quantity** (unchanged): C_K = E[max(0, max_{i≤K} ΔU_i − η)] / E[cost_K], where η is a small detection threshold (≈ 1e-4) distinct from the acceptance tolerance ε.

**Gaussian / isotropic null model** for the acceptance-volume function:

$$A_\pi(\varepsilon) \;=\; P(\Delta U \ge -\varepsilon) \;=\; \Phi\!\Big(\tfrac{\mu+\varepsilon}{\sigma}\Big) \;=\; \tfrac{1}{2}\operatorname{erfc}\!\Big(-\tfrac{\mu+\varepsilon}{\sigma\sqrt 2}\Big)$$

This is the closed form *if* ΔU is approximately Gaussian(μ, σ²) at the local state. **π appears here as the standard Gaussian normaliser, not as a forced geometric ornament**: it is the natural consequence of assuming locally isotropic mutation effects. The model is testable by a Kolmogorov–Smirnov test on the empirical per-arm ΔU histogram.

**Predicted regime split**:
- *Easy regime* (e.g. mutual_inhibition H=128, possibly H=256): ΔU empirical histogram fits Gaussian; A_π(ε) tracks empirical accept_rate(ε); π-formula applies.
- *Rugged regime* (e.g. bytepair_proj H=384 knife-edge bimodality, replica-symmetry-breaking-like landscapes per Urbani et al. 2024): ΔU heavy-tailed or multimodal; A_π(ε) breaks; π-formula does not apply; empirical CDF used instead.

The break of the Gaussian null is itself a positive empirical signature of a CSP-clustering / RSB landscape topology (Mézard, Montanari, Zecchina 2002; Mannelli, Zdeborová et al. 2022; Urbani et al. 2024).

**Recent theoretical anchors** (literature context for the acceptance-aperture framing; none is a direct prediction for our deterministic threshold-accepting grower):

- **Li, Wang, Dou, Rosenthal (2024), arXiv:2408.06894** — the asymptotic 0.234 acceptance-rate result for *random-walk Metropolis* / parallel tempering kernels (with ESJD optimization, probabilistic acceptance) is robust under Gaussian-like proposal kernels. This is a *literature anchor* for the probabilistic class; our deterministic threshold-accepting search may have a different optimum, and the 0.234 figure is not a predicted target for our system.
- **Chen, Mikulincer, Reichman, Wein (2023), arXiv:2312.13554** — time lower bounds for SA establish that on certain hard instances no ε schedule (including adaptive) can reach within ratio Ω(1/n^{1−δ}). Acceptance-aperture tuning has theoretical limits in worst-case regimes.
- **Ma et al. (2024), GECCO 2024, arXiv:2404.08239 (GLEET)** — meta-learned adaptive ε schedules deliver substantial improvements over static ε in evolutionary algorithms; the empirical frontier is landscape-adaptive ε, not a single fixed value.
- **Ren et al. (2023), AISTATS 2024, arXiv:2311.13159** — Wasserstein–Fisher–Rao gradient flows decompose mutation-selection into Wasserstein transport + Fisher–Rao birth/death; ε plays the role of the Fisher–Rao reweighting temperature in this framing.
- **Discrete NES (2024), arXiv:2404.00208** — extends the natural-gradient view of evolution strategies to discrete binary domains, the closest existing match to our binary-spike substrate.
- **Liao et al. (2024), arXiv:2407.20724** — spin-glass / replica-symmetry-breaking analogy for DNN loss landscapes; provides background for the rugged-landscape interpretation of our H=384 bimodality but does not prove it for our system.
- **Angelini & Ricci-Tersenghi (2022), arXiv:2206.04760** — limits of simulated annealing on sparse hard inference problems (e.g. planted CSPs) where SA is trapped by glassy states. Background for "hard landscape + annealing limits"; we do not claim a closed-form ε* from this work.

We claim *not* that any of these results have been validated on our substrate; we claim that the framework they jointly form is the right reading-list for Phase D analysis.

**Phrasing of the null model** (per GPT review):

> *"We use A_π(ε) as a Gaussian acceptance-volume null model, not as an assumed law. The model is accepted only if empirical ΔU distributions and accept-rate curves fit it. If not, we replace A_π with the empirical CDF A_emp(ε)."*

---

## Open hypotheses

- **Phase B.1** — `accept_ties × horizon × seed` ablation (2 × 3 × 5 = 30 cells) on `mutual_inhibition` only. Tests whether the B1 horizon effect survives strict Bonferroni at n ≥ 10, whether 80k steps lets H=384 surpass H=256, and whether neutral-accept policy substantively affects peak.
- **Phase D** — continuous ε-sweep over acceptance tolerance (e.g. ε ∈ {strict, 0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2}) × horizon × 5 seeds × `mutual_inhibition` H=384. Pre-registered hypotheses include: (H1) existence of ε* with peak gain > 1.5pp; (H2) Li 2024 calibration: at ε*, acceptance rate ≈ 0.234 (testable null); (H3) Bouchaud-style trap rescue: seed=1042 H=384 bytepair_proj 0.0% trap escapes for ε ≥ ε_trap; (H4) Gaussian null fit (KS test) per arm; (H5) Chen 2023 impossibility class detection. Plus post-hoc analyses on the *existing* candidate logs (no new compute): avalanche size distribution + branching ratio σ (Beggs–Plenz 2003 criticality test) and two-time fitness correlation (Bouchaud aging signature). Pre-registration to be filed before launch.
- **Phase E** — operator schedule retuning (theta-heavy, channel-heavy, drop `projection_weight`). Pre-registered as a predictive improvement test, not a post-hoc narrative.
- **`bytepair_proj` collapse ablation.** Separate from Phase B.1 / D, because the failure mode at H=384 is grow-prune-aggressiveness, not horizon or acceptance tolerance. Candidate knobs: prune rate, keep-best restore, minimum-alive constraint.
- **External baseline comparisons** — random topology, gradient-trained networks of comparable parameter count, NEAT — are not currently run. They are needed before any architectural claim of the form "X% on this task is meaningful in absolute terms" can be made. Future work.

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
