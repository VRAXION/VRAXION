# Phase B Pre-Registration: Confound vs. Intrinsic Test of the H=256 Inverted-U

**Document status.** Pre-registered experimental protocol. Written before Phase B data collection. Hypotheses, statistical tests, and success/failure criteria are fixed here and must not be revised after seeing Phase B results, except as clearly noted amendments with a separate timestamp.

**Predecessor.** Phase A: 30-cell baseline sweep, `output/dimensionality_sweep/20260424_091217/`, `seeds=5`, `H ∈ {128, 256, 384}`, fixtures `mutual_inhibition` and `bytepair_proj`, `steps=20000`.

---

## 1. Background and motivation

The Structured Chaos Theory (`docs/wiki/Structured-Chaos-Theory.md`) claims `L = Ψ · σ_μ / D`, implying monotonic degradation of learning rate with search dimensionality. A single-seed observation suggested the opposite (H↑ improves). Phase A disconfirmed both single-direction claims:

| H | n | mean peak_acc | ±std | mean accept% | mean alive_frac | mean edges / cap |
|---|---|---------------|------|-------------|-----------------|------------------|
| 128 | 5 | 3.76% | 0.91 | 78% | 0.72 | 1145 / 1147 |
| 256 | 5 | 5.28% | 1.79 | 42% | 0.44 | 4587 / 4587 |
| 384 | 3 (partial) | 3.63% | 0.85 | 13% | 0.48 | 8645 / 10322 |

(`bytepair_proj` fixture pending at time of writing.)

The result is an **inverted-U** with peak at H=256. Two mutually exclusive hypotheses:

- **H_intrinsic.** The H=384 decline reflects a true architectural ceiling, independent of search policy. Scaling the non-scaling hyperparameters will not rescue it.
- **H_confound.** The H=384 decline reflects that non-scaling hyperparameters (propagation ticks, input-embedding width, jackpot size, step budget) become the binding constraint as H grows. Scaling any of them will restore H=384 performance to or beyond H=256.

Mechanistic signals from Phase A already favor H_confound (monotonic accept-rate collapse from 78→42→13%, edge-cap not reached at H=384), but the signals are suggestive, not decisive. Phase B resolves the question.

---

## 2. Design

### 2.1 Primary measured quantity

**C_K**, the per-step normalized expected useful progress (consensus convention, this codebase, v1):

$$C_K(g) \;=\; \frac{\mathbb{E}\bigl[\max\bigl(0,\; \max_{i \le K} \Delta U_i \;-\; \varepsilon\bigr)\bigr]}{\mathbb{E}[\mathrm{cost}_K]}$$

- `g` is a network state; `K` is the jackpot size (currently 9); `ΔU_i` is the fitness delta of the `i`-th candidate mutation; `ε` is the detection threshold.
- `ε = 1e-4` (fitness is cosine similarity in `[-1, 1]`; empirical sub-threshold drift is below this).
- The expectation is over mutations under the proposal distribution fixed by the mutation schedule.
- `cost_K` is the wall-clock time to evaluate all K candidates.
- Phase B reports C_K computed on a rolling window (e.g. 2000 steps) across training.

### 2.2 Explanatory decomposition (hypothesis, not definition)

Following consensus with the swarm and GPT review:

$$C_K \;\stackrel{?}{\approx}\; V_\text{raw} \cdot M_\text{pos} \cdot A \cdot I_\text{proxy} \;/\; (D_\text{eff} \cdot \mathrm{cost}_\text{eval} \cdot R_\text{neg})$$

where each factor is a separately measured empirical quantity (operationalization in §3). Phase B tests whether the log-linear regression of `log C_K` onto the log-components achieves `R² > 0.8` across all arms and seeds; a poor fit falsifies the decomposition as an adequate explanation of C_K.

### 2.3 Fixtures

Only `evolve_mutual_inhibition` (`instnct-core/examples/evolve_mutual_inhibition.rs`), using the champion recipe from Phase A (Law I + II, smooth cosine × `(1 + 0.1·alive_frac)`, 1+9 jackpot). `bytepair_proj` is omitted because it mixes a grow-prune cycle into the training schedule, confounding arm interpretation.

### 2.4 Arms

All arms fix `H = 384`, `seeds ∈ {42, 1042, 2042, 3042, 4042}` (5 seeds), and keep every other Phase A parameter at its current default unless explicitly altered. The per-arm deviation is the minimal single-axis change.

| Arm | Change vs Phase A H=384 baseline | Specifically tests |
|-----|----------------------------------|--------------------|
| **B0** | none (re-run baseline for cost control) | reproducibility sanity check |
| **B1** | `--steps 40000` (2× total mutations) | **horizon / budget**: does H=384 need more search iterations to reach the edge-cap and saturate? |
| **B2** | `--jackpot 18` (2× candidates per step) | **selection order-statistic**: does C_K rise if the best-of-K tail is fed more samples? Cost per step doubles; `C_K` formula self-normalizes for this. |
| **B3** | `--ticks 12` (2× propagation depth) | **information propagation**: does a deeper signal pass through the larger network rescue representation retention? Conflated with eval cost; `C_K` still self-normalizes. |
| **B4** | `--input-scatter` (32-dim input replicated/scattered across full `phi_dim ≈ 237`) | **input-representation bottleneck**: does saturating the entire input zone with signal rescue learning? |

B0 must be re-run because Phase A had `n=5` for H=384 only partially at time of pre-reg; B0 guarantees a same-environment control matched to the other arms.

Total: **5 arms × 5 seeds = 25 new runs**, plus the existing 5 Phase A runs at H=384 serve as an independent replication check against B0.

### 2.5 Arm semantics — explicit cautions

Following the GPT critique, each arm must be interpreted with its native semantics; none is a clean "single C-component probe":

- **B1** does not reduce `cost_eval` (per-candidate cost is unchanged). It increases *total* mutations available; `C_K` per step is unchanged in expectation unless search has plateaued.
- **B2** increases `K` in `best-of-K`, directly changing the numerator of `C_K` via extreme-value statistics. Denominator (cost_K) also doubles. Net effect depends on the tail of the ΔU distribution.
- **B3** changes both `I_proxy` (via deeper signal traversal) *and* `cost_eval` (each forward pass is 2× longer). `C_K` nets these; component decomposition disentangles.
- **B4** changes the input channel utilization, not the parameter count — it modifies `D_eff` only if the embedding currently bottlenecks information flow. Otherwise it is a no-op.

### 2.6 Non-scaling controls kept fixed

Chain count, density_pct, threshold_max, channel_max, inhibitory_pct, edge_cap_pct, λ (anti-monopoly), and evaluation corpus remain at their Phase A defaults.

---

## 3. Measurement panel (per run)

All arms log the same diagnostic panel. Each field is logged either per candidate, per progress interval, or at run end.

### 3.1 Per-candidate log (CSV, flag-gated via `--candidate-log <path>`)

One row per evaluated candidate (10⁵–10⁶ rows/run at 20k–40k steps × 9–18 jackpot). Fields:

```
run_id, arm, seed, H, step, candidate_id, operator_id,
before_U, after_U, delta_U, accepted, eval_ms
```

### 3.2 Periodic panel (every `PROGRESS_INTERVAL = 2000` steps)

Computed on a fixed probe set of 32 corpus pairs drawn deterministically at startup (seed-independent):

| Metric | Formula | Literature anchor |
|---|---|---|
| `accept_rate_window` | fraction of last 2000 steps with ≥1 accepted candidate | (internal) |
| `f_active` | fraction of output neurons that are non-constant across probe set | Kauffman 1969 frozen component |
| `H_output_mean` | `mean_i H[P(class | x_i)]` in nats | Standard output entropy |
| `H_output_var` | `Var_i H[P(class | x_i)]` | Derived disambiguator |
| `stable_rank` | `‖Y‖_F² / ‖Y‖₂²` on charge matrix `Y ∈ ℝ^{32×phi_dim}` | Dong et al. 2021; Gao et al. 2017 |
| `kernel_rank` | `numerical_rank(Y, tol=1e-6)` | Maass 2002 LSM |
| `separation_SP` | `mean_{u,v} ‖x_u−x_v‖ / (‖u−v‖+ε)` on probe pairs | Maass 2002 separation property |
| `D_eff_sensitivity` | `E[‖Δoutput‖₂ : 1 random candidate mutation]` over 64 samples | Derrida & Pomeau 1986 |

### 3.3 Run-end panel

Final single-pair metrics computed on full evaluation set (1000 pairs):

| Metric | Formula | Anchor |
|---|---|---|
| `peak_acc`, `final_acc` | as Phase A | — |
| `dCor_io` | `dCor(input_embed, output_charge)` over probe set | Székely & Rizzo 2007 |
| `collision_rate` | `|unique(argmax(Y))| / |unique(X)|` | (internal) |
| `CKA_linear` | `‖Y^T X‖_F² / (‖X^T X‖_F · ‖Y^T Y‖_F)` after centering | Kornblith et al. 2019 |
| `motif_z3` | Z-score of signed feed-forward triangle count vs degree-preserving null, 200 permutations | Milo et al. 2002, 2004 |
| `branching_σ` | mean ratio of active-neuron count at tick t+1 to tick t, over all inputs and ticks | Beggs & Plenz 2003 |
| Edge-count, wall_clock_s | as Phase A | — |

### 3.4 Post-hoc per-arm aggregates

Computed in `tools/diag_constructability_analysis.py` from §3.1 log:

- `V_raw = P(ΔU > ε)` over all candidates
- `V_sel = P(best-of-K ΔU > ε)` over steps
- `M_pos = E[ΔU | ΔU > ε]`
- `R_neg = E[|ΔU| | ΔU < -ε]`
- `cost_eval = mean(eval_ms)`
- `cost_progress = total_wall_clock / n_accepted`
- Per-operator breakdowns of all the above (11 operators)
- `C_K_rolling(step)` — rolling window of 2000-step expected-useful-progress

---

## 4. Statistical plan

### 4.1 Primary comparison

For each arm `B1..B4` versus `B0`:

- **Outcome**: `peak_acc` (primary) and `C_K_mean_over_run` (secondary).
- **Test**: Welch's t-test on per-seed means, two-sided, α = 0.05 Bonferroni-corrected for 4 arms (effective α = 0.0125).
- **Minimum detectable effect size**: Δmean ≥ 1.5pp on `peak_acc` (comparable to Phase A H=128→256 gap of 1.52pp). Power analysis: with n=5, σ ≈ 1.8pp, detectable Δ ≈ 2.3pp at α=0.0125, power=0.8 — borderline. The Phase B design is powered to detect LARGE arm effects only. Small effects will produce inconclusive p-values; this is acknowledged and not post-hoc salvaged.

### 4.2 Secondary analyses

**Decomposition regression.** For each (arm, seed) data point:

$$\log C_K \;=\; \beta_0 + \beta_1 \log V_\text{raw} + \beta_2 \log M_\text{pos} + \beta_3 \log A + \beta_4 \log I_\text{proxy} - \beta_5 \log D_\text{eff} - \beta_6 \log \mathrm{cost}_\text{eval} - \beta_7 \log R_\text{neg} + \varepsilon$$

With n=25 data points and 7 regressors this is underdetermined for stable coefficient estimation; report `R²` as the primary readout. If `R² > 0.8`, the product decomposition is an adequate model. If `R² < 0.6`, the decomposition is inadequate and the paper will report this as a finding, not suppress it.

**Criticality diagnostic (optional secondary).** If any arm reaches a peak_acc ≥ 6%, compute the **Derrida/Hamming divergence slope** on that network: perturb the input by 1-bit flips, measure mean Hamming distance between the perturbed and unperturbed output at each of 6 ticks, fit slope `log(d_t/d_0)/t`. A slope ≈ 1 at the winning arm *only* constitutes evidence for the edge-of-chaos framing; otherwise the criticality narrative will not be used in the paper.

**Accept-rate-matched control.** If the decomposition regression reveals an overwhelming coefficient on `V_raw`, a follow-up experiment (Phase C) will hold accept-rate constant across H by adaptively tuning the mutation size; Phase B does not conduct this.

### 4.3 Pre-specified decision rules

- **Rule 1**: If any of B1..B4 shows `peak_acc ≥ H=256 Phase A mean − 0.5pp` (i.e. ≥ 4.78%), and Welch p < 0.0125 vs B0, **H_confound is confirmed** for that knob.
- **Rule 2**: If none of B1..B4 improves over B0 at any `α ≤ 0.05`, and accept_rate remains below 20% across all arms, **H_intrinsic is confirmed** within the tested knob range.
- **Rule 3**: If the decomposition regression `R² < 0.6`, a revised formula will be proposed (e.g. dropping R_neg if strict-improvement policy makes it non-identifiable) but only as a Phase C hypothesis, not adopted post hoc.
- **Rule 4**: Any "weird outlier seed" will be kept in the analysis. No seed exclusion post hoc. If a seed produces a crash or NaN, the full run is redone with the next-available seed `5042, 6042,…` and that fact is reported.

### 4.4 Protected claims (paper language)

**Claims we permit ourselves to make** after Phase B (conditional on outcomes):

- ✅ "Phase A shows peak accuracy in byte-pair prediction on the INSTNCT architecture is not monotonic in neuron count H under fixed training budget; we find an inverted-U with peak at H=256 (mean 5.28% ± 1.79pp across 5 seeds). The prior 7.50% champion accuracy corresponds to the upper tail (seed=42), not the distribution median of 4.70%."

- ✅ "Phase B identifies that [specific knob] is the binding constraint at H=384 under our training schedule." (conditional on Rule 1)

- ✅ "We operationalize a progress metric C_K for mutation-selection search that normalizes useful improvement by evaluation cost; decomposition into established factors (DFE tail, novelty, representational retention, landscape sensitivity) explains [R² value] of C_K variation across our conditions."

**Claims we will NOT make without further dedicated experiments:**

- ❌ "The three laws of structured chaos theory are necessary and sufficient."
- ❌ "The network operates at the edge of chaos." (requires explicit Derrida slope = 1 at the winning configuration, AND accept-matched control)
- ❌ "Constructability is a new fundamental law of gradient-free learning." (this is evolvability adapted to this architecture)
- ❌ Any claim about scaling to unseen tasks or architectures.

---

## 5. Implementation changes required before launch

1. **`instnct-core/examples/evolve_mutual_inhibition.rs`**
   - Add CLI flags: `--jackpot N` (currently hardcoded 9), `--ticks N` (currently hardcoded via `InitConfig::phi`), `--input-scatter` (boolean), `--candidate-log <path>` (optional).
   - Extend the `SUMMARY` JSON output with: `dCor_io, collision_rate, CKA_linear, motif_z3, branching_σ, f_active, H_output_mean, H_output_var, stable_rank, kernel_rank, separation_SP, D_eff_sensitivity`.
   - Emit per-candidate CSV rows if `--candidate-log` path set.

2. **`tools/diag_dimensionality_sweep.py`**
   - Add `--arm <name>` parameter forwarding to the per-candidate log filename.
   - Pass-through for new Rust flags.

3. **`tools/diag_constructability_analysis.py`** (new)
   - Load all per-candidate CSVs across arms.
   - Compute all aggregates in §3.4.
   - Run the §4.2 regression.
   - Emit aggregate tables and a pre-specified plot set.

4. **Branch discipline**
   - All Phase B implementation lives on `claude/review-repo-access-Ug8Si` (current working branch).
   - The logging and new-flag patches do NOT rebuild the existing binaries while Phase A is in flight: the driver invokes `cargo run --release --example ...`, and cargo rebuilds on source change. To avoid mid-sweep rebuild, the implementation waits for Phase A to emit its final aggregate line, confirmed by a sentinel in `results.json` (`len(results) == 30` or `Sweep total wall` line in driver.log).
   - Logging is flag-gated (default off) so re-running Phase A for any reason reproduces Phase A identically.

---

## 6. Compute budget

Per-arm expected wall-clock (estimated from Phase A H=384 ≈ 1800s per run):

| Arm | Per-run factor | Per-run wall | 5 seeds wall | Total |
|-----|---------------|--------------|-------------|-------|
| B0 | 1.0 | 30 min | 150 min | 2.5 h |
| B1 | 2.0 (2× steps) | 60 min | 300 min | 5.0 h |
| B2 | 2.0 (2× candidates) | 60 min | 300 min | 5.0 h |
| B3 | 2.0 (2× ticks) | 60 min | 300 min | 5.0 h |
| B4 | ≈1.05 (scatter is cheap) | 32 min | 160 min | 2.7 h |
| — | — | — | — | **~20 h** |

Consistent with the sandbox's throughput (~1 CPU core at 100% continuous). If Phase A takes the projected ~10 hrs total (currently running), Phase B starts roughly 20–24 hours from Phase A launch; Phase B completes roughly 40–48 hours from Phase A launch.

If the user's local machine has more cores, Phase B arms are trivially parallelizable (each arm is a separate cargo invocation), cutting wall-clock to ~4–5 h on an 8-core machine.

---

## 7. Versioning

- Phase B pre-reg v1.0 — pre-data commit SHA to be stamped on first commit.
- Amendments: timestamped, appended only. No silent edits of decision rules or claims after seeing data.

---

*End of pre-registration document.*
