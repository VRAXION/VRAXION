# Overnight SCT Empirical Research — Consolidated Findings

**Window**: 2026-04-24 01:42 → 02:16 CEDT (7 iterations)
**Branch**: `research/overnight-sct-empirical-20260423`
**Per-iteration narrative**: `docs/research/overnight_2026-04-23_progress.md` (same directory)

---

## Executive summary (read this first)

1. **The strongest positive finding**: in the grower-regression lane (3 seeds × 6 symbolic tasks), **task-level final val_acc collapses tightly across seeds for 5 of 6 tasks** (stdev ≤ 3.5pp). This is the only observable tested tonight that has both a clean cross-seed signal AND is inside SCT's stated scope (gradient-free, mutation-driven learning). It's a candidate LHS for any future empirical scaling law, but at (system × task) granularity — not as a single system-level scalar.

2. **The most important correction**: iterations 1-3 declared the SCT formula `L ∝ 1/D` "refuted" based on the byte-unit latent-dim sweep. That sweep uses Adam + LBFGS (gradient-based training). SCT is explicitly scoped to "gradient-free, mutation-driven" systems. **The original refutation claim is retracted as a scope mismatch.** The byte-unit findings remain valid as INSTNCT activation facts but do not test SCT.

3. **The unresolved question for morning**: Is the 5/6 task-level val_acc collapse "architecture matters" signal, or "easy tasks hit their ceiling regardless of seed" artifact? Existing grower task list does not include any task where no seed reaches ceiling, so this cannot be answered with overnight-local compute.

---

## 1 · Belief diff (iteration 1 vs now)

| At iteration 1 | After iteration 6 |
|---|---|
| SCT naive-D formula contradicted | Retracted — wrong domain applied |
| Byte-unit sweep H=16/LD=24/c19 is "the architecture" | Rank #1 holds across 3 seeds (80.86 mean), BUT specific-cell claims have stdev 10pp — only axis-trends (c19, bigger H, bigger LD) are seed-stable |
| Per-step delta can be treated as σ_μ | Per-step delta varies 53× across tasks at same seed (38.5pp diagonal_xor vs 0.71pp digit_parity) — "learning rate" is not a system-level scalar |
| No multi-seed evidence existed | 3-seed data on 2 independent systems (byte-unit and grower-regression), ~10 min wall-clock total |

---

## 2 · Standing claims (with evidence)

### 2.1 Task-level val_acc seed-stability (grower, 3 seeds)

| task | s42 | s123 | s777 | mean | stdev |
|---|---:|---:|---:|---:|---:|
| four_parity | 100.0 | 100.0 | 100.0 | 100.00 | 0.00 |
| four_popcount_2 | 100.0 | 100.0 | 100.0 | 100.00 | 0.00 |
| is_digit_gt_4 | 71.5 | 70.0 | 67.0 | 69.50 | 2.29 |
| diagonal_xor | 88.5 | 88.5 | 82.5 | 86.50 | 3.46 |
| digit_parity | 90.0 | 85.5 | 87.5 | 87.67 | 2.25 |
| full_parity_4 | 80.5 | 78.0 | 98.0 | 85.50 | **10.90** (outlier) |

**Claim**: 5 of 6 tasks have stdev ≤ 3.5pp. Final val_acc per task IS a seed-stable observable. Confidence: high for these 6 tasks, moderate in generalization (only 3 seeds, only 1 task family — boolean classification).

### 2.2 Structural non-stability

| task | neurons @ s42 | s123 | s777 | stdev |
|---|---:|---:|---:|---:|
| is_digit_gt_4 | 7 | 3 | 1 | **3.06** |
| full_parity_4 | 1 | 1 | 5 | **2.31** |
| digit_parity | 12 | 8 | 12 | 2.31 |

**Claim**: the same final val_acc can emerge via very different-sized solutions. There is no unique "architectural complexity" for a task. Confidence: high.

### 2.3 c19 activation advantage in gradient training (byte-unit, 3 seeds)

Mean final_lossless across 60-config sweep, averaged over 3 seeds:
- identity 30.40, tanh 33.03, relu 33.44, **c19 48.52**

**Claim**: c19 activation gives +15-18pp mean advantage in the gradient-trained byte-unit auto-encoder. Confidence: high.

**Caveat**: this is not an SCT test — wrong regime. It is an INSTNCT activation validation. Also: c19 wins via higher ceilings (max 80.86), not lower floors (min 18.23 ≈ others' mins). The advantage is "access to better maxima", not "safer distribution".

### 2.4 full_parity_4 stepping-stones divergence (indirect Law II evidence)

Seed 777 found a 5-neuron 98% solution where seeds 42 and 123 stopped at 1-neuron 80/78%. Same architecture, same budget, same stall limit. The grower's `non-strict accept gate` (which accepts compositional bets with no immediate fitness gain) enabled this path for one seed.

**Claim**: exploration-preserving mechanisms (stall tolerance + scout breadth + non-strict accept) enable basin discovery that single-seed runs miss. Confidence: moderate (n=1 divergence across 3 seeds).

**Significance for SCT**: this aligns with Law II's *spirit* (anti-monopoly = prevent premature convergence), though not its specific mechanism (mutual inhibition). The grower has a different anti-monopoly mechanism — the non-strict gate — that serves the same function.

---

## 3 · Retracted or significantly weakened

### 3.1 "SCT naive `L ∝ 1/D` refuted" (iter 1-3) — **RETRACTED**

Correct version: "In gradient-trained autoencoder settings (outside SCT's stated 'mutation-driven' scope), bigger parameter count monotonically helps. This does not test SCT." The `byte-unit_latent_dim_sweep_gpu_probe` data remains a valid INSTNCT finding, just mislabeled.

### 3.2 "H=16/LD=24/c19 is THE architecture" (iter 1) — **weakened**

It's the top-mean cell, yes, but Kendall rank stability across seeds is only 0.25-0.47. Individual cell rankings are unstable; only axis-level trends (c19 > others, bigger H > smaller, bigger LD > smaller) are robust.

---

## 4 · Orthogonal insights (not SCT-direct but important)

### 4.1 Per-step delta is not a system scalar

For seed=42 in the grower, per-step val_acc delta mean ranged from 38.5pp (diagonal_xor, 1 neuron) to 0.71pp (digit_parity, 12 neurons). The formula `L = Ψ · σ_μ / D` implicitly treats L and σ_μ as system-level constants. That framing cannot express this task-dependence.

### 4.2 Non-strict accept gate is active in practice

In `digit_parity`'s trajectory, 6 of 12 accepted neurons had **negative** val_acc delta (the neuron made things worse at that step). The grower accepted them anyway as compositional stepping stones. This means σ_μ (raw fitness delta) misreads 50% of the selection signal in that task as "noise" when it is structured exploration.

### 4.3 Wall-clock time is highly reproducible across seeds

Per-task wall-clock stdev: 0-5s on runs of 0.2-33s. The search budget (stall limit) is deterministic; the search *path* varies. This decouples compute-cost from solution-quality in a way the formula does not address.

---

## 5 · Open adversarial questions (morning handoff)

**Q1 (highest priority)**: Is the 5/6 task-level val_acc collapse architecture-signal or easy-task-ceiling artifact?
- To distinguish: run on a task where no seed reaches ceiling. Existing task list (boolean classification) does not include such a task.
- Requires: authoring a harder grower task OR applying grower to a completely new domain.

**Q2**: Does Law II (anti-monopoly) hold causally, or is the `full_parity_4` divergence coincidence?
- To test: toggle non-strict accept gate off (adding `--strict-gate` flag to grower) and re-run 3 seeds on full_parity_4.
- Requires: code modification to `neuron_grower.rs` (add flag + behavior switch).

**Q3**: Can Ψ be operationalized from already-available data?
- **Preview attempted in iteration 8** (see progress.md §8). Defined two candidate Ψ values per activation using existing 3-seed byte-unit sweep data:
  - Ψ_ratio@70 (fraction of cells reaching ≥70% final_lossless): identity 0.036, tanh 0.091, relu 0.055, c19 0.291
  - Ψ_mean (mean final_lossless / 100): identity 0.337, tanh 0.366, relu 0.394, c19 0.530
- **Finding**: the two definitions produce DIFFERENT orderings — Ψ_ratio says `tanh > relu`, Ψ_mean says `relu > tanh`. This is a failure mode: an acceptable Ψ should be stable across reasonable measurement choices.
- **Conclusion**: Ψ operationalization remains unresolved. The preview demonstrates the problem — without a pre-registered measurement procedure, Ψ is definition-sensitive.
- **Note**: this preview is on the byte-unit (gradient-trained) lane, which is outside SCT's stated scope. A Ψ-analog for the grower lane (accept-fraction per step, scout pair-lift rate) was NOT tested tonight and is the real open question.

---

## 6 · Ranked next-steps (for morning discussion)

1. **Resolve Q1 (ceiling-artifact question)** — 70% of the night's value hinges on this. If the collapse is architecture, SCT has a real empirical foothold. If ceiling artifact, we need harder tasks before any formula claim.
2. **Implement Q2 (Law II ablation on grower)** — directly probes the SCT hypothesis, not just correlates with it. Requires small code change to the grower.
3. **Operationalize Ψ concretely** — iteration 8's preview is a start. A proper definition needs: a measurement procedure, an error-bar protocol, and a sanity check that different Ψ values actually predict different outcomes.
4. **Multi-seed expandability probe** (GPT's candidate) — never attempted tonight. Would give an alternative LHS for comparison with task-level val_acc.

---

## 7 · Methodological suggestions for SCT doc

The SCT doc at `docs/wiki/Structured-Chaos-Theory.md` already has a useful §6 disclaimer, but two additions would make claims harder to misread:

- **Explicit scope labelling on each law** — "Law I is tested on gradient-free mutation in X context; does not claim to hold in gradient-based training." Would have saved me iterations 1-3 of scope-mismatch work.
- **Per-step vs task-level distinction for L** — the formula implicitly treats L as a system-level scalar. The grower data shows this is wrong at per-step granularity but possibly right at task-outcome granularity. Two symbols (L_step, L_task) may be needed.

These are optional — the user already decided to add the experimental disclaimer earlier, which is the most important safeguard.

---

## 8 · Reproducibility

All commands, seeds, and file paths are in `docs/research/overnight_2026-04-23_progress.md`. Key reruns:

```bash
# Byte-unit multi-seed sweep (GPU, ~3 min per seed):
python tools/diag_byte_unit_latent_dim_sweep.py --seed 456 --float-epochs 80 --qat-epochs 80 --out docs/research/data/latent_sweep_seed456
python tools/diag_byte_unit_latent_dim_sweep.py --seed 789 --float-epochs 80 --qat-epochs 80 --out docs/research/data/latent_sweep_seed789

# Grower regression multi-seed (CPU, ~90s per seed):
python tools/run_grower_regression.py --data-seed 123 --search-seed 123 --report-dir target/grower-regression-multiseed/seed123 --golden /tmp/no_golden_file_here
python tools/run_grower_regression.py --data-seed 777 --search-seed 777 --report-dir target/grower-regression-multiseed/seed777 --golden /tmp/no_golden_file_here
```

Raw JSON outputs are local-only (`docs/research/data/` is in the global `data/` gitignore). Concrete numbers are embedded in `progress.md` for audit.

---

**Bottom line for the user**: we found one seed-stable observable (task-level val_acc, 5/6 tasks), we retracted one overclaim (scope mismatch on byte-unit sweep), and we located one pressing open question (ceiling artifact vs architecture signal) that needs one harder task or one design change to resolve. The formula `L = Ψ · σ_μ / D` is neither validated nor refuted on its own terms — it remains poetry until Ψ gets a measurement procedure.
