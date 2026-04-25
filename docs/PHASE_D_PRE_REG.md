# Phase D Pre-Registration: Acceptance Aperture (D0 → D1 → D2 phasing)

**Document status.** Pre-registered experimental protocol. Written before Phase D data collection. Hypotheses, statistical tests, and decision rules are fixed here and must not be revised after seeing Phase D results, except as clearly noted amendments with a separate timestamp.

**Predecessors.** Phase A (30-cell baseline, `output/dimensionality_sweep/20260424_091217/`), Phase B (25-cell horizon-confound test on H=384 `mutual_inhibition`, `output/phase_b_full_20260424/`), Phase B.1 (30-cell horizon × tie-policy ablation, `output/phase_b1_horizon_ties_20260425/`).

**Phasing.** Per GPT review (2026-04-25), Phase D is split into three sub-phases of escalating compute cost. Each gate must pass before the next launches.

---

## 1. Background and motivation

The `Mutation-Selection Dynamics` page introduces an "Acceptance Aperture" parameter ε that defines the topology of the directed reachability graph:

- ε < 0 (strict): accept ΔU > 0 only.
- ε = 0 (neutral / Zero-Drive Search): accept ΔU ≥ 0.
- ε > 0 (tolerant / Threshold-Drive Search): accept ΔU ≥ −ε.

Phase B.1 demonstrated (n=5, directional, not Bonferroni-significant) that:

- **Strict acceptance plateaus within the tested 20k–80k horizon range** (5.50% at 40k, 5.60% at 80k, p=0.92).
- **80k + neutral acceptance (6.78%) exceeds the Phase A H=256 reference (5.28%)** — H=384 is not inherently inferior under combined longer-horizon + Zero-Drive conditions.
- **The largest discontinuity is strict → neutral**, not ε=0 → ε>0 (under neutral, accept rate already saturates at ~99.6%).

Phase D measures the *continuous* ε spectrum, including the tolerant regime (ε > 0). It tests whether a non-zero ε* exists at which (reachable-set expansion) outweighs (lost selection pressure) for our specific landscape, and whether the discontinuity is concentrated at strict→neutral or distributed across the full ε axis.

---

## 2. Phase D0 — Offline aperture analysis (no new compute)

**Purpose.** Use the existing 12.6M-row Phase B.1 candidate logs to characterise the empirical Δ_U distribution per arm × seed × operator, then propose ε_small and ε_large for D1 from data, not theory.

**Analyses (executed by `tools/diag_phase_d0_aperture.py`):**

1. **Δ_U and `best_delta_per_step` histograms** per arm, raw and log-scale.
2. **Mass at Δ_U = 0**: fraction within {1e-9, 1e-6, 1e-4} of zero — distinguishes a continuous tail from a true point-mass plateau.
3. **Empirical accept_rate(ε)** on a fine ε grid {strict, 0, 1e-6 … 1e-1}, per arm. This is the empirical equivalent of A_π(ε) without any Gaussian assumption.
4. **Gaussian null fit per arm and per operator**:
   - Lilliefors-corrected KS (parametric bootstrap; raw KS with fitted parameters is biased upward in p).
   - Anderson–Darling normality test (tail-emphasis).
   - QQ plot data dump for visual inspection.
   - Per-operator decomposition: a mixed-operator marginal can appear non-Gaussian even when individual operators are. Report goodness-of-fit per operator.
5. **ε_small and ε_large recommendation** for D1: choose ε_small at accept_rate-midpoint between strict and neutral; ε_large where accept_rate ≈ 0.5 (NOT the Li 2024 0.234 anchor — see §6 below).

**D0 outputs**: `output/phase_d0_aperture/{d0_summary.json, d0_accept_rate_curve.csv, d0_per_operator.csv, d0_recommendation.md}`.

**D0 decision rules**:

- If Anderson–Darling rejects Gaussian for ≥ 50% of arm-cells → A_π is not the right null; D1 ε grid set from empirical accept_rate curve, not Gaussian formula.
- If most operators individually fit Gaussian but the marginal does not → mixed-marginal artefact; per-operator A_π fits independently but the global formula still uses A_emp.
- If the recommended ε_small ≈ ε_large (i.e. accept rate jumps abruptly between strict and tolerant) → D1 needs a finer ε grid in the discontinuity region.

---

## 3. Phase D1 — Small policy sweep at H=384 (30 cells)

**Launches only after D0 reports a coherent ε recommendation.**

### 3.1 Design (v2.2 — adds K axis)

D0.5 (`tools/diag_phase_d0_5_jackpot_aperture.py` + `docs/research/PHASE_D0_5_JACKPOT_APERTURE.md`) showed that the K=9 jackpot **already saturates the ties acceptance regime** (ties accept ≈ 96% at K=2, ≈99% at K=3). Therefore, sweeping `zero_p` at fixed K=9 only differentiates outcomes within the saturated range. K is upstream and equally informative; D1 v2.2 adds a small K axis.

| Factor | Value | Total |
|---|---|---|
| Jackpot size K | {1, 3, 9} | 3 |
| Acceptance policy | {strict, zero_p=0.3, zero_p=1.0} | 3 |
| Training horizon | 40k | 1 |
| Seed | {42, 1042, 2042, 3042, 4042} | 5 |
| Fixture | `mutual_inhibition` only | 1 |
| H | 384 only | 1 |

Total: **45 cells.** The factorial structure tests whether the best result comes from:
- high discovery with saturated neutral aperture (K=9, zero_p=1.0 — the existing "ties at K=9" baseline),
- lower-cost partial aperture (K=3, various zero_p),
- minimal pooling with high selection pressure (K=1, various zero_p) — also informative as the "no max-pool" reference (analogous to activation function with no spatial pooling in CNN literature).

Lean alternative if compute budget is tight: K ∈ {3, 9} × policy ∈ {strict, zero_p=0.1, zero_p=0.3, zero_p=1.0} = 40 cells. Slightly more zero_p resolution, drops K=1.

### 3.2 D0 finding that changes the design (v2.1)

D0 (offline analysis on the existing 12.6M-row B.1 candidate logs, see `tools/analyze_acceptance_aperture.py` and `docs/research/PHASE_D0_ACCEPTANCE_APERTURE.md`) revealed that the **best-of-K jackpot selector saturates the acceptance aperture at ε = 0**:

- `best_negative_rate ≈ 0.0006` (only ~0.06% of best-of-9 candidates have ΔU < 0)
- `best_exact_zero_rate ≈ 0.82–0.92` (the best-of-9 candidate is exactly 0 in 82–92% of steps — *zero-dominated regime*)
- `best_positive_rate ≈ 0.08–0.18`

Therefore, **moving from ε = 0 to ε > 0 does not open new selectable moves** under this selector — it only changes which of the ~0.06% all-negative cases are accepted, which is negligible. The full discontinuity is concentrated at the strict → neutral boundary, parameterised by the **probabilistic zero-acceptance probability `zero_p`** (accept ΔU = 0 with probability p, otherwise reject).

D1 v2.1 therefore tests the `zero_p` axis only, abandoning the explicit ε > 0 sweep:

- **strict** (zero_p = 0): accept iff ΔU > 0
- **zero_p = 0.1, 0.3, 0.6, 1.0**: accept iff ΔU > 0; otherwise iff ΔU = 0 with probability p
- **legacy_ties**: existing `accept_ties=true` policy, included as exact reproducer of B.1 80k ties cell

A Gaussian null A_π(ε) was Lilliefors-corrected-KS-tested per arm and per operator in D0; the test rejects Gaussianity (KS stat ≈ 0.49 across arms; H4 status: **Gaussian null rejected — substrate is non-isotropic locally, dominated by zero point-mass**). The π-formula does not apply for this substrate; the empirical CDF A_emp(ε) is used in any further analyses where an aperture model is needed.

### 3.3 Required Rust patch

A new acceptance-rule abstraction on `evolve_mutual_inhibition.rs`:

```
strict:        accept iff ΔU > 0
zero_p:        accept iff ΔU > 0; OR (ΔU == 0 AND uniform(0,1) < p)
epsilon:       accept iff ΔU >= -ε         (kept in CLI for future use, not in D1)
```

Implemented in `--accept-policy {strict,ties,zero-p,epsilon}` with `--neutral-p` and `--accept-epsilon` sub-arguments. A parity smoke test must show:
- `--accept-policy strict` ≡ no-flag (current strict default).
- `--accept-policy zero-p --neutral-p 1.0` ≡ `--accept-ties true`.

GPT confirmed PASS on smoke (commit 383d9f6 on `codex/phase-b-logging-smoke`): strict accept rate 10%, zero-p=0.1 accept rate 22%, zero-p=1.0 accept rate 100%.

### 3.4 Hypotheses (pre-registered, v2.1)

**H1 (existence of an intermediate zero_p winner)**: There exists at least one zero_p ∈ {0.1, 0.3, 0.6} for which mean peak_acc exceeds both `strict` and `zero_p=1.0` (= legacy ties) by ≥ 0.5pp. If true, this means probabilistic zero-acceptance is a tunable knob, not a binary on/off.

**H2 (Li 2024 literature anchor — NOT a direct prediction)**: as v2.0 — report empirical accept_rate(p) alongside the 0.234 anchor as a literature reference point only.

**H3 (Bouchaud-style trap rescue)**: deep-trap seeds (e.g. analogues of the seed=1042 H=384 0.0% trap in Phase A bytepair_proj) escape under intermediate zero_p more than under strict and equally well as under zero_p=1.0. If higher trap-escape rates are observed at intermediate p, probabilistic neutral search has structural value beyond binary acceptance.

**H4 (Gaussian null status, REJECTED in D0)**: per-arm and per-operator Lilliefors-corrected KS tests on B.1 ΔU distributions reject Gaussianity (KS ≈ 0.49). The A_π(ε) formula does not apply; the framework's π-paragraph is downgraded from "candidate null" to "rejected null in this substrate".

**H5 (Chen 2023 impossibility class)**: as v2.0 — if no zero_p produces peak gain over strict, the substrate may sit in the Chen impossibility regime within the tested compute budget.

### 3.5 Statistical plan

Welch t-test of each non-baseline arm vs the strict baseline (n=5, df ≈ 8). Bonferroni-correct over 5 non-baseline comparisons (α/5 ≈ 0.01). Effect size threshold ≥ 1.5 pp for "Bonferroni-significant" (we expect to fall short and rely on directional + future replication).

### 3.6 Decision rules

- D1 winner = arm with highest mean peak_acc AND consistent direction across seeds (no seed reverses the gap from baseline).
- If D1 winner is `neutral_p=1.0` → tolerant ε > 0 does NOT add to neutral; D2 not warranted.
- If D1 winner is `ε_small` or `ε_large` → tolerant ε > 0 adds value; D2 H-axis sweep warranted.
- If D1 winner is intermediate `neutral_p ∈ {0.1, 0.3}` → probabilistic interpolation matters; new pre-reg may be needed before D2.

---

## 4. Phase D2 — H-axis sweep (15 cells, conditional on D1 winner)

**Launches only if D1 winner is non-trivial (not strict, not exactly neutral_p=1.0).**

### 4.1 Design

| Factor | Value | Total |
|---|---|---|
| Acceptance policy | D1 winner | 1 |
| Training horizon | 40k | 1 |
| Seed | {42, 1042, 2042, 3042, 4042} | 5 |
| Fixture | `mutual_inhibition` | 1 |
| H | {128, 256, 384} | 3 |

Total: **15 cells.**

### 4.2 D2 hypothesis (high-D concentration of measure — exploratory, v2.1)

**H6 (geometric, reformulated for `zero_p` not ε)**: optimal `zero_p*` shrinks with H. Concretely: at higher H the substrate has more zero-plateau states (per-step `best_exact_zero_rate` increases), and the optimal probability of accepting them might decrease — preserving the rare positive moves more strictly. Or it might INCREASE — rewarding plateau exploration to find the rare positive-direction stepping stones.

The high-D unit-ball volume formula V_n = π^(n/2)/Γ(n/2+1) (peak n=5, decay to 0 thereafter) is invoked as a *geometric analogy*, not a deduction — H is not directly an Euclidean dimension. The reformulation from ε* to p* reflects the D0 finding that ε > 0 is empirically irrelevant under this selector.

Falsifying outcome: `zero_p*(H)` is flat across {128, 256, 384} — supports the "permutation symmetry compensates for measure shrinkage" interpretation.

This hypothesis is *exploratory* and not Bonferroni-corrected; it is a guidance for paper interpretation, not a strict-significance claim.

---

## 4.5 Phase E — minimum-useful threshold (+δ) sweep (placeholder)

**Status: pre-reg placeholder, not yet active.** Design follows once D1 winner is known.

The third axis of the search activation function A(K, τ, s) — the utility cutoff τ — has been tested only at τ = 0 (B.1, D1 plan) and at τ < 0 (D0; rejected as empirically irrelevant under K=9 jackpot). The τ > 0 direction (`minimum-useful δ`, `accept iff ΔU > δ`) is unvested.

D0.6 (`tools/diag_phase_d0_6_minimum_useful.py`) provides offline **calibration-only** quantiles of the positive best-of-K ΔU distribution. These quantiles inform δ_small / δ_med / δ_large grid placement for Phase E. **D0.6 explicitly does not predict outcomes** (peak_acc, final_acc): once a move is rejected, the trajectory diverges, and the existing logs do not preserve counter-factual trajectories.

Phase E sketch (to be pre-registered before launch, after D1 winner is known):

| Factor | Range |
|---|---|
| τ (minimum-useful) | {0 (strict baseline), δ_small, δ_med, δ_large} |
| K | D1 winner |
| s | D1 winner |
| Seed | 5 |
| Fixture, H, horizon | mutual_inhibition, H=384, 40k |

Total: ~20 cells if 4 τ points × 5 seeds.

Phase E hypothesis (placeholder):

**H7 (minimum-useful trade-off)**: there exists δ* > 0 such that the substrate trained with `accept iff ΔU > δ*` reaches better peak_acc than strict (δ = 0). Mechanism candidate: filtering trivially-small positive ΔU prevents drift on noise. Falsifying outcome: every δ > 0 produces lower or equal peak_acc — substrate benefits from accumulating small positive moves; small-step accumulation is real.

---

## 5. Cross-phase post-hoc analyses (zero new compute on B.1 data)

These run on the Phase B.1 candidate logs alongside D0 and report independently:

1. **Avalanche size distribution** from the candidate logs. Avalanche = consecutive accepted-mutation runs. Fit P(s) ∝ s^{−τ} (Beggs–Plenz 2003). Critical signature: τ ≈ 1.5 with branching ratio σ ≈ 1. We do *not* pre-register criticality as a claim; we report the fit quality and let the paper position it.
2. **Two-time fitness correlation** C(t_w, t) from training time-series logs. If ratio-dependent (aging signature, Bouchaud 1992), the system is glassy in a specific technical sense; if stationary, it is not.

---

## 6. Literature anchors and their (limited) bearing on D-phase claims

The following 2022+ papers inform D-phase design but **none provides a direct prediction for our deterministic threshold-accepting grower**:

- **Li, Wang, Dou, Rosenthal (2024), arXiv:2408.06894** — 0.234 acceptance rate optimal for *random-walk Metropolis* and parallel-tempering kernels (probabilistic acceptance, ESJD-optimised). Our acceptance is deterministic threshold; we report empirical ε* accept_rate alongside the 0.234 anchor only as a literature reference, NOT as a prediction.
- **Chen, Mikulincer, Reichman, Wein (2023), arXiv:2312.13554** — time lower bounds for SA establish that no ε schedule can rescue every regime. Background for the "some landscapes are intrinsically hard" boundary statement.
- **Ma et al. (2024) GECCO 2024, arXiv:2404.08239 (GLEET)** — meta-learned adaptive ε schedules outperform static ε in evolutionary algorithms. Frontier reference; D1/D2 use static ε for simplicity.
- **Ren et al. (2023) AISTATS 2024, arXiv:2311.13159** — Wasserstein–Fisher–Rao gradient flows decompose mutation-selection into Wasserstein transport + Fisher–Rao birth/death; ε plays the role of the Fisher–Rao reweighting temperature in this framing.
- **Discrete NES (2024), arXiv:2404.00208** — natural-gradient view of evolution strategies extended to discrete binary domains.
- **Liao et al. (2024), arXiv:2407.20724** — RSB / DNN landscape analogy. Background for the rugged-landscape interpretation.
- **Angelini & Ricci-Tersenghi (2022), arXiv:2206.04760** — limits of SA on glassy planted-CSP regimes. Background for the "annealing limits in glassy regimes" boundary.

---

## 7. Compute budget

| Phase | Cells | Wall (parallel `--jobs 12`) | Wall (serial) |
|---|---|---|---|
| D0 (no new compute, analysis only) | 0 | ~5 min | ~5 min |
| D1 (after D0) | 30 | ~2.5 h | ~30 h |
| D2 (after D1, conditional) | 15 | ~1.0 h | ~12 h |

D1 and D2 disk: ~3 GB additional candidate-log artefacts each.

---

## 8. Versioning

- Phase D pre-reg v2.0 — restructured into D0/D1/D2 phasing per GPT review (2026-04-25).
- v1.0 was a 70-cell single-phase ε-sweep, since superseded by the more efficient phasing.
- No amendment after each sub-phase begins, except by separately timestamped addendum.

---

## 9. Boundaries

- D1 / D2 test `mutual_inhibition` H=384 only (D2 extends H-axis). Generalisation to other fixtures, other tasks is future work.
- The high-D V_n hypothesis (H4 in D2) is exploratory; strict-significance claims about it require a dedicated dedicated experiment beyond Phase D.
- All claims are conditional on the substrate, fitness function, and mutation operator schedule used in Phase A / B / B.1. The framework's external generality is unaddressed by Phase D.

---

*End of Phase D pre-registration.*
