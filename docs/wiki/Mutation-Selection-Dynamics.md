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

### Phase B.1 — 30-cell horizon × tie-policy ablation at H=384

```
                       peak_acc%        accept%      verdict
20k strict             3.52 ± 1.14      17%          Phase A baseline reproduction
20k ties               4.50 ± 1.96      99.8%        +0.98 pp directional
40k strict             5.50 ± 1.47      18%          horizon recovery
40k ties               5.88 ± 1.72      99.6%        +0.38 pp directional
80k strict             5.60 ± 1.58      17%          tested strict plateau within 80k
80k ties               6.78 ± 1.52      99.6%        +1.18 pp directional ← BEST
```

Welch tests (n=5 per cell, df ≈ 8, *uncorrected*; Bonferroni at α/9 ≈ 0.0056):

| comparison | Δ pp | t | p (two-tail) | verdict |
|---|---|---|---|---|
| 80k vs 40k strict | +0.10 | 0.10 | 0.92 | tested plateau within 80k — strict does not recover further |
| 80k vs 20k ties | +2.28 | 2.06 | 0.076 | directional, not Bonferroni-significant |
| 80k ties vs strict | +1.18 | 1.20 | 0.26 | directional, not Bonferroni-significant |

**Three findings, all directional** (n=5 underpowered for strict significance):

1. **Tested strict plateau within 80k**: 40k vs 80k strict differ by 0.10pp (p=0.92). We cannot rule out further gains beyond 80k; the claim is bounded to the tested range.
2. **80k + ties (6.78 ± 1.52%) exceeds the Phase A H=256 reference (5.28 ± 1.79%)**: the H=384 substrate is *not* inherently inferior; under longer horizon AND Zero-Drive acceptance, it surpasses the H=256 baseline. This is **architecture rehabilitation under combined conditions** — neither horizon alone (strict 80k = 5.60% only matches H=256) nor neutral alone (20k ties = 4.50%) achieves it.
3. **`alive_frac` candidate signature**: under strict acceptance, alive_frac grows with horizon (44.7% → 46.2% → 63.2%); under ties it shrinks (61.4% → 51.3% → 44.2%). The Zero-Drive regime appears to *specialise* output activity over time, not diversify. This is a **candidate microscopic signature** consistent with the topology-gate framing; we do not yet claim it causally.

The direction of the tie-policy effect is consistent across horizons (+0.98, +0.38, +1.18 pp) but with non-monotone magnitude. Replication at n ≥ 10 is required for strict-significance claims; the 80k+ties regime warrants targeted re-test.

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
- **Neutral / Zero-Drive Search (ε = 0):** accept ΔU ≥ 0. Established under "neutral drift on neutral networks" (van Nimwegen, Crutchfield, Huynen 1999, *PNAS*; Wagner 2005, *Robustness and Evolvability in Living Systems*, Princeton UP).
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
- **Ma et al. (2024), GECCO 2024, arXiv:2404.08239 (GLEET)** — deep-RL meta-learned exploration-exploitation policy for evolutionary computation. Cited as evidence that **landscape-adaptive policy** (a generalization of static ε) is an active frontier; the GLEET paper does not directly study threshold-accepting or static ε per se, so its quantitative improvement figures should not be transferred to our setting without dedicated re-test.
- **Ren et al. (2023), AISTATS 2024, arXiv:2311.13159** — Wasserstein–Fisher–Rao gradient flows decompose mutation-selection into Wasserstein transport + Fisher–Rao birth/death; ε plays the role of the Fisher–Rao reweighting temperature in this framing.
- **Discrete NES (2024), arXiv:2404.00208** — extends the natural-gradient view of evolution strategies to discrete binary domains, the closest existing match to our binary-spike substrate.
- **Liao et al. (2024), arXiv:2407.20724** — spin-glass / replica-symmetry-breaking analogy for DNN loss landscapes; provides background for the rugged-landscape interpretation of our H=384 bimodality but does not prove it for our system.
- **Angelini & Ricci-Tersenghi (2022), arXiv:2206.04760** — limits of simulated annealing on sparse hard inference problems (e.g. planted CSPs) where SA is trapped by glassy states. Background for "hard landscape + annealing limits"; we do not claim a closed-form ε* from this work.

We claim *not* that any of these results have been validated on our substrate; we claim that the framework they jointly form is the right reading-list for Phase D analysis.

**Phrasing of the null model** (per GPT review):

> *"We use A_π(ε) as a Gaussian acceptance-volume null model, not as an assumed law. The model is accepted only if empirical ΔU distributions and accept-rate curves fit it. If not, we replace A_π with the empirical CDF A_emp(ε)."*

**Empirical status (Phase D0, 2026-04-25)**: the Gaussian A_π null is **rejected** for this substrate. Lilliefors-corrected KS tests on the per-arm ΔU distributions from the 12.6M-row B.1 candidate logs gave KS statistics ≈ 0.49 (substantial deviation from Gaussian), driven by a **zero-dominated point-mass regime**: 82–92% of best-of-K candidates have ΔU = 0 exactly, 0.06% are negative, and 8–18% are positive. The substrate is therefore *non-isotropic locally*; the empirical A_emp(ε) curve is used in further analyses, and the π-formula has no empirical legitimacy in this regime.

A second, related D0 finding: under the best-of-K jackpot selector, **moving from ε = 0 to ε > 0 does not open new selectable moves** — the only changed cases are the 0.06% all-negative best-of-K instances. The full discontinuity in acceptance behaviour is concentrated at the strict → neutral boundary, parameterised by the probability p of accepting ΔU = 0 (`zero_p`). Phase D1 therefore tests a `zero_p` axis (probabilistic neutral) rather than a `ε > 0` axis (tolerant). See `docs/PHASE_D_PRE_REG.md` v2.1 and `docs/research/PHASE_D0_ACCEPTANCE_APERTURE.md`.

### Acceptance Aperture as a Three-Parameter Search Activation Function

Phase D0.5 (offline K-resampling on the same B.1 candidate logs, see `tools/diag_phase_d0_5_jackpot_aperture.py` and `docs/research/PHASE_D0_5_JACKPOT_APERTURE.md`) revealed that the acceptance aperture is structurally a **three-parameter parameterised activation function**: a sampling pool, a utility cutoff, and a boundary softness.

$$A(K,\tau,s)$$

| Parameter | Meaning | Range so far |
|---|---|---|
| **K** | jackpot pool size — how many candidates per step | tested K ∈ {1, 2, 3, 5, 9} (D0.5 offline; B.1 ran K=9) |
| **τ** | utility cutoff: τ < 0 tolerant, τ = 0 neutral/strict boundary, τ > 0 minimum-useful | only τ = 0 tested (B.1, D1 plan); τ > 0 is the open `+δ` axis |
| **s** | boundary softness at τ — hard step (s = 0) vs probabilistic / smooth (s ∈ (0, 1]) | tested s = 0 (strict) and s = 1 (full ties) on B.1; D1 plan covers s ∈ {0, 0.3, 1.0} at τ = 0 |

The pipeline:

```
1. K candidates sampled from parent       — sampling aperture (upstream)
2. best-of-K ΔU selected                  — max-pool operation
3. acceptance criterion at τ with softness s — soft-threshold valve (downstream)
```

This is mathematically the same shape as a CNN max-pool followed by an activation function. The mapping:

| Search component | Neural network analog |
|---|---|
| K (jackpot pool size) | max-pool kernel size (Boureau, Ponce, LeCun 2010) |
| best-of-K operation | max-pool over the candidate set |
| τ (utility cutoff) | activation **bias** term |
| `strict` (τ = 0, s = 0) | ReLU `max(0, x)` |
| `zero_p = p` (τ = 0, s = p) | Leaky-ReLU-like soft slope at zero (Maas et al. 2013; PReLU, He et al. 2015; ELU, Clevert et al. 2015) |
| `tolerant ε` (τ = −ε, s = 0) | activation with shifted threshold left |
| `minimum-useful δ` (τ = +δ, s = 0) | activation with shifted threshold right |

K-dependence is governed by extreme-value statistics on the candidate ΔU distribution (Fisher–Tippett–Gnedenko 1928): if the per-candidate positive-rate is p_pos, then under independent sampling the strict accept rate at jackpot K is `1 − (1 − p_pos)^K`. Empirically this prediction tracks observed strict accept across `K ∈ {1, 2, 3, 5, 9}` to within ~10–15% (slight deviation indicating per-step candidate correlation, likely from shared parent state).

Three empirical consequences for the framework:

1. **Ties acceptance saturates at small K** — by K=2, ties accept rate ≈ 96%; by K=3, ≈ 99%. The K=9 setting in our experiments has been operating in the **fully saturated ties regime**. Anything we previously interpreted as "ties is universally good" is instead "K=9 + ties is essentially a free-walk on the iso-fitness manifold". The `s` axis (zero_p) only differentiates outcomes in the strict-to-saturated range.
2. **Strict discovery scales monotonically with K** but **C_K per-cost declines mildly**. The optimal (K, s) pair is therefore a **2D trade-off** — at fixed τ = 0.
3. **The τ-axis (utility cutoff) is the third independent parameter**, and is **not yet empirically tested**. The `tolerant ε` (τ < 0) was rejected as empirically irrelevant under K=9 jackpot (D0). The `minimum-useful δ` (τ > 0) is the open direction — accepting only mutations with margin above zero, filtering out trivially-small positive ΔU. Phase D0.6 (offline calibration only) and Phase E (live test) address it.

The "Acceptance Aperture" is therefore not a single substrate-specific parameter but a **3D parameterised search activation function** A(K, τ, s). The optimal `(K*, τ*, s*)` for a given substrate is the activation that the substrate's gradient-free training requires — analogous to learning-rate × momentum × weight-decay tuning in gradient-based methods.

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
