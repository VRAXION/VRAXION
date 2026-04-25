# Local Constructability Framework

> *Finite-budget mutation-selection in sparse neural substrates.*

**Status: Empirical framework.** Not a theory. The framework names a measurable quantity (C_K, local constructability) and proposes an explanatory decomposition. The core claims are supported by two preregistered experimental phases (A and B); generalisation across architectures, tasks, and longer horizons is open work.

---

## Core thesis

Useful computation in a gradient-free mutation-selection system arises from **two coupled mechanisms**, neither sufficient alone:

1. **Interference dynamics** at the signal level — what survives destructive cancellation in a recurrent substrate is what gets read out as inference.
2. **Mutation-selection dynamics** at the structural level — the substrate that supports productive interference is **built**, not searched, by local mutation under three organising laws.

The framework is **local** because it characterises behaviour around the current state of a single network, under a finite training budget. It does not claim universal architectural conclusions.

---

## The three-level structure

| Level | Sub-document | What it explains |
|---|---|---|
| Signal | [Interference Dynamics](Interference-Dynamics) | how signal becomes inference via cancellation in a recurrent substrate |
| Structure | [Mutation-Selection Dynamics](Mutation-Selection-Dynamics) | how the substrate is constructed by local mutation under three laws, measured by C_K |
| Emergence | [Constructed Computation](Constructed-Computation) | which computational behaviours emerge when both mechanisms operate within their effective regimes |

A speculative extension is kept separate, deliberately:

- [Speculative Extension — Cognitive Emergence](Cognitive-Emergence-Speculative)

---

## Primary measured quantity

$$C_K(g)\;=\;\frac{\mathbb{E}\bigl[\max\bigl(0,\;\max_{i \le K} \Delta U_i \;-\; \varepsilon\bigr)\bigr]}{\mathbb{E}[\mathrm{cost}_K]}$$

Per-step expected useful improvement, normalised by the cost of evaluating *K* candidate mutations. Operationally, *C_K* is computed from a per-candidate log of (operator, before, after, accepted, eval-time) emitted by the training run.

The framework also proposes (as a tested hypothesis, not an axiom) a multiplicative decomposition

$$C_K \;\stackrel{?}{\approx}\; \frac{V_{\mathrm{raw}}\,\cdot\,M_{\mathrm{pos}}\,\cdot\,A\,\cdot\,I_{\mathrm{proxy}}}{D_{\mathrm{eff}}\,\cdot\,\mathrm{cost}_{\mathrm{eval}}\,\cdot\,R_{\mathrm{neg}}}$$

— see [Mutation-Selection Dynamics](Mutation-Selection-Dynamics) for component definitions and the regression test we use to evaluate the decomposition.

---

## Empirical anchors

- **Phase A** — 30 runs, 2 fixtures × 3 H values × 5 seeds. Establishes a multi-seed baseline and shows that the H-profile is **recipe-dependent**: `mutual_inhibition` fixture exhibits an inverted-U with peak at H=256 (mean 5.28% peak); `bytepair_proj` fixture monotonically declines with H, with high variance and near-collapse at H=384.
- **Phase B** — 25 runs, 5 ablation arms × 5 seeds at H=384 with full per-candidate logging. The B1 arm (2× horizon) recovers H=384 to the H=256 reference band (5.50% mean peak), supporting a training-horizon confound interpretation directionally (Welch p=0.047, not formally Bonferroni-significant). Other arms (extra ticks, larger jackpot, scattered input) do not help and several actively hurt.
- **Cross-replication** — Phase B's B0 arm reproduces Phase A's `mutual_inhibition` H=384 cell bit-for-bit on independent compute, across operating systems.

Per-run artifacts: `output/dimensionality_sweep/20260424_091217/` (Phase A) and `output/phase_b_full_20260424/` (Phase B).

---

## What the framework claims

- Search dimensionality interacts with training horizon, mutation schedule, and fixture-specific pruning policy. A fixed-H sweep alone produces misleading architectural conclusions.
- Per-step constructability *C_K* is a workable empirical proxy for "how much useful structure can be built from the current state under a single jackpot step".
- The mechanism-level interpretation that should accompany any architectural claim has at least three components (signal, structure, emergence), and they should be tested separately rather than collapsed into a single scalar.

## What the framework does not claim

- That *L = Ψ · σ_μ / D* (the earlier Structured Chaos learning equation) is a quantitative law. Phase B's B1 arm shows that doubling search horizon recovers performance at "high D" without changing D, contradicting any interpretation of the equation as monotone in D under fixed budget.
- Edge-of-chaos / criticality. Variance peaks at H=256 are consistent with multiple mechanisms (selection-rate bimodality, exploration regimes); a separate Derrida-style perturbation test with accept-rate-matched controls is required before that language is used.
- That the framework generalises to non-spiking, gradient-trained, or larger architectures. Such claims require dedicated cross-architecture replication.
- Anything about consciousness. Cognition is measurable here through task performance and capacity proxies. Consciousness is not. Speculative extensions are sequestered to the dedicated appendix and labelled as such.

---

## Read next

- [Interference Dynamics](Interference-Dynamics) — signal-level mechanism, residual computation, destructive cancellation, R_neg readings.
- [Mutation-Selection Dynamics](Mutation-Selection-Dynamics) — structure-level mechanism, three laws, C_K measure, per-operator findings.
- [Constructed Computation](Constructed-Computation) — what emerges when both mechanisms operate within regime, empirical capacity findings.
- [Speculative Extension — Cognitive Emergence](Cognitive-Emergence-Speculative) — research-direction notes; not paper claims.
- [INSTNCT Architecture](INSTNCT-Architecture) — the substrate the framework is studied on.
- [Research Process & Archive](Timeline-Archive) — full chronology and prior-name history (Theory of Thought, Structured Chaos Theory).

---

*Local Constructability Framework v0.1 (April 2026). The framework will be promoted to "Local Constructability Theory" only after replication at n ≥ 10 strict-significance and a successful operator-schedule retuning experiment (Phase C).*
