# Phase D Pre-Registration: Acceptance Aperture (ε) Sweep

**Document status.** Pre-registered experimental protocol. Written before Phase D data collection. Hypotheses, statistical tests, and decision rules are fixed here and must not be revised after seeing Phase D results, except as clearly noted amendments with a separate timestamp.

**Predecessors.** Phase A (30-cell baseline, `output/dimensionality_sweep/20260424_091217/`) and Phase B (25-cell horizon-confound test on H=384 `mutual_inhibition`, `output/phase_b_full_20260424/`). Phase B.1 (binary `accept_ties` ablation across horizons) is in progress at the time of writing and may complete before or during Phase D; Phase D *generalises* Phase B.1 from a binary policy to a continuous tolerance sweep.

---

## 1. Background and motivation

The `Mutation-Selection Dynamics` page introduces an "Acceptance Aperture" parameter ε that defines the topology of the directed reachability graph:

- ε < 0 (strict): accept ΔU > 0 only.
- ε = 0 (neutral / Zero-Drive Search): accept ΔU ≥ 0.
- ε > 0 (tolerant / Threshold-Drive Search): accept ΔU ≥ −ε.

Phase B.1's preliminary 20-cell snapshot suggested neutral acceptance directionally improves peak (~+1pp at 20k, ~+0.4pp at 40k), with accept-rate jumping from ~17% to ~99.8%. This indicates a large iso-fitness plateau is being traversed under neutral acceptance.

Phase D measures the *continuous* ε spectrum, including the tolerant regime (ε > 0). It tests whether a non-zero ε* exists at which (reachable-set expansion) outweighs (lost selection pressure) for our specific landscape.

---

## 2. Hypotheses (pre-registered)

**H1 (existence)**: There exists ε* in (0, 0.005] for which mean peak_acc exceeds the B0 baseline (3.52% at H=384, 20k steps) by ≥ 1.5pp, *while* arm-mean R_neg(accepted) remains ≤ 1.5× B0 baseline.

**H2 (Li 2024 literature anchor — NOT a direct prediction)**: Li, Wang, Dou, Rosenthal (2024, arXiv:2408.06894) report that for *random-walk Metropolis* and parallel-tempering kernels (probabilistic acceptance, ESJD-optimized), the asymptotic optimal acceptance rate is robustly near 0.234. Our acceptance rule is **deterministic threshold accepting**, not Metropolis, so we expect a different optimum. We report the empirical ε* and the corresponding accept_rate(ε*) and *contrast* it against the 0.234 anchor only as a literature reference point. We do NOT pre-register 0.234 as a target or as a falsifying threshold for our system.

**H3 (Bouchaud trap rescue)**: The H=384 `bytepair_proj` seed=1042 0.0% trap (Phase A) escapes for ε ≥ ε_trap. If observed, ε_trap gives a lower bound on the local barrier height (Bouchaud 1992 trap model). If no ε ≤ 0.01 escapes, the trap is too deep for tolerance-only rescue. This is a single-seed test; not part of the Bonferroni family below.

**H4 (Gaussian null fit per arm and per operator)**: Per-arm AND per-operator goodness-of-fit test of empirical ΔU histogram against fitted Normal(μ, σ²).

Methodological note: a naïve KS-test against a Normal(μ, σ²) where μ and σ are estimated from the same sample produces an inflated p-value. We use:
1. **Lilliefors-corrected KS** (or equivalent: parametric bootstrap calibration) for the parameter-fitted-from-data case;
2. **QQ plot inspection** as a non-test sanity diagnostic (visual tail behaviour);
3. **Tail-deviation metric** (e.g. Anderson–Darling A² with simulated p-value) emphasising tail behaviour, where the heavy-tail signature of a rugged landscape is most visible;
4. **Per-operator decomposition** — the mixed-operator marginal can appear non-Gaussian even when each operator's ΔU is approximately Gaussian (and vice-versa). We report goodness-of-fit per operator AND for the marginal.

Predicted regime split (after methodological adjustments): Gaussian fit holds in `mutual_inhibition` H=128 / H=256 cells (easy regime) and breaks in H=384 / `bytepair_proj` cells (rugged regime). The break itself is a positive signature of the CSP-clustering / RSB landscape interpretation (Mézard, Parisi, Zecchina 2002; Liao et al. 2024 arXiv:2407.20724 for DNN landscape RSB analogy).

**H5 (Chen 2023 impossibility class)**: If no ε in the tested range produces peak_acc improvement and accept_rate(ε) does not track the Gaussian null, the landscape is consistent with Chen, Mikulincer, Reichman, Wein (2023, arXiv:2312.13554) impossibility-class regimes (no ε schedule rescues). This is a falsifying-the-existence-of-ε* outcome and is itself a defensible finding.

---

## 3. Design

### 3.1 Sweepelt dimensions

| Factor | Values | Total |
|---|---|---|
| Acceptance tolerance ε | {strict (Δ>0), 0 (neutral), 1e-5, 1e-4, 1e-3, 5e-3, 1e-2} | 7 |
| Training horizon | {20 000, 40 000} | 2 |
| Seed | {42, 1042, 2042, 3042, 4042} | 5 |
| Fixture | `mutual_inhibition` only | 1 |
| H | 384 only | 1 |

Total: **70 cells**, all on `mutual_inhibition` H=384 to keep the comparison clean.

### 3.2 Why `mutual_inhibition` only

`bytepair_proj` H=384 has two distinct failure modes (knife-edge bimodality + heavy crystallise-prune dynamics) that confound an ε-sweep interpretation. Mixing fixtures in Phase D would not yield a clean ε* per fixture without a much larger budget. A separate `bytepair_proj` collapse-ablation experiment is left as future work.

### 3.3 Required Rust patch

A new flag `--accept-tolerance ε` on `evolve_mutual_inhibition.rs` generalises `accept_ties=true` to a continuous parameter:

- Acceptance criterion: `Δ ≥ −ε`.
- ε < 0: strict (current default).
- ε = 0: neutral (matches `accept_ties=true`).
- ε > 0: tolerant.

A parity smoke test (no-flag run vs `--accept-tolerance -1` run) must confirm bit-identical results before Phase D launches. Same as Gate 1 logic in Phase B.

### 3.4 Per-cell artefacts

All Phase B logging infrastructure carries forward:

- per-candidate CSV (`run_id, arm, seed, step, candidate_id, operator_id, before_U, after_U, delta_U, accepted, eval_ms`)
- panel summary JSON (final-state metrics)
- panel time-series CSV (every 2000 steps)
- final checkpoint

---

## 4. Statistical plan

### 4.1 Primary outcome

Peak accuracy across the ε grid at each horizon. Evaluation per (ε, horizon) cell: mean ± std over 5 seeds.

### 4.2 H1 test

For each (ε, horizon) cell, Welch t-test vs (ε=strict, horizon=20k) baseline. Bonferroni correction over 6 ε values × 2 horizons − 1 baseline = 11 comparisons; α/11 ≈ 0.0045. Effect size threshold ≥ 1.5pp peak gain. R_neg(accepted) sanity ≤ 1.5× baseline.

### 4.3 H2 test (Li 2024 calibration)

Compute accept_rate(ε) for each cell. The ε*_argmax-peak cell's accept_rate should be in [0.15, 0.30] (loose interval around 0.234) for the Gaussian null to be consistent. Falsification: ε* exists *and* its accept_rate is far from 0.234 (e.g., < 0.05 or > 0.75).

### 4.4 H4 test (Gaussian null fit per arm)

For each (ε, horizon) cell, run KS-test on empirical ΔU histogram (from candidate log) against fitted Normal(μ, σ²). Report KS-statistic and p-value. Pass threshold: p > 0.05 for Gaussian fit.

### 4.5 Post-hoc analyses on existing data (zero new compute)

Computed from the existing Phase B candidate logs and final checkpoints:

1. **Avalanche size distribution**. From Phase B candidate logs, extract avalanche = consecutive accepted-mutation runs. Fit P(s) ∝ s^{−τ} (Beggs–Plenz 2003). Critical signature: τ ≈ 1.5 with branching ratio σ ≈ 1 specifically at the H=256 cell (Phase A). Computable today.

2. **Two-time fitness correlation** C(t_w, t) from training logs of Phase B. If t/t_w-dependent (aging), the system is glassy in the Bouchaud sense. Computable today.

These two analyses do not require Phase D launch and can be performed concurrently with the Phase D parity smoke + first runs.

---

## 5. Decision rules

- **H1 confirmed**: ≥ 1 ε > 0 cell with peak gain ≥ 1.5pp at p < 0.0045, and R_neg(accepted) ≤ 1.5× baseline. → ε* established for this substrate.
- **H1 falsified**: no ε > 0 cell satisfies both criteria. → strict / neutral remain optimal in tested range; document as null result.
- **H2 (literature anchor only)**: report empirical ε* accept_rate alongside the 0.234 Li et al. 2024 reference point. We do not "support" or "falsify" Li 2024 with this comparison, since their result is for a different acceptance class (probabilistic Metropolis, not deterministic threshold).
- **H4 split (predicted)**: Gaussian KS fit in MI H=128 / 256, fails in H=384 cells. → CSP-clustering interpretation of bytepair-style failures gains structural support.
- **H5 (impossibility class)**: H1 falsified *and* H4 fails everywhere. → escalate to Chen 2023 framing in `Constructed Computation` / future-work.

---

## 6. Compute budget

Per-cell wall-clock at H=384 ranges from ~25 min (20k steps) to ~50 min (40k steps). Parallel run on a 12-core machine (`--jobs 12`):

- 35 cells × 25 min × 0.6 (parallel efficiency at 12 jobs) ≈ 73 min
- 35 cells × 50 min × 0.6 ≈ 146 min
- **Total ≈ 4 h parallel.** Serial: ~ 30 h.

Disk: ~3 GB additional candidate-log artefacts.

---

## 7. Versioning

- Phase D pre-reg v1.0 — pre-data commit.
- No amendment after Phase D run begins, except by separately timestamped addendum.

---

## 8. Boundaries

- Phase D tests `mutual_inhibition` H=384 only. Generalisation to other fixtures, other H values, and other tasks is future work.
- The ε-sweep grid is geometrically spaced from 1e-5 to 1e-2; finer or wider grids may be needed if results sit near grid edges.
- All claims are conditional on the substrate, fitness function, and mutation operator schedule used in Phase A / Phase B. The framework's external generality is unaddressed by Phase D.

---

*End of Phase D pre-registration.*
