<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Project Timeline

## What This Page Is

This page is the single primary timeline and lookup surface for VRAXION. Use it to see what is current, what changed, what got retired, and what still blocks promotion.

## How To Read It

- Start with `Current Snapshot` if the question is what is live right now.
- Use `Project Timeline` if the question is what changed and why it mattered.
- Use `Open Questions and Promotion Gates` for the unresolved lines that still need proof before promotion.
- Use [Validated Findings](Validated-Findings) for evidence and [INSTNCT Architecture](INSTNCT-Architecture) for the active technical line.

## Current Snapshot

- Version source of truth: [`docs/VERSION.json`](https://github.com/VRAXION/VRAXION/blob/main/docs/VERSION.json)
- Current canonical public release: [`v4.2.0`](https://github.com/VRAXION/VRAXION/releases/tag/v4.2.0) (stable, published `2026-03-16`)
- Internal code path remains [`instnct/`](https://github.com/VRAXION/VRAXION/tree/main/instnct); that repo path is not the public release label.
- Current mainline code path: [`instnct/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/instnct/model/graph.py)
- Current strongest schedule result in the public evidence layer: voltage medium leak at `22.11%` peak / `21.46%` plateau
- Current strongest compact learnable schedule result: the 3-angle decision-tree schedule at `20.05%`
- Current English recipe candidate on `main`: [`instnct/recipes/train_english_1024n_18w.py`](https://github.com/VRAXION/VRAXION/blob/main/instnct/recipes/train_english_1024n_18w.py) with an `8`-tick triangle-derived `2 add / 1 flip / 5 decay` schedule; it still uses the existing float signed edge representation.
- Current strongest edge-representation quality result: sign+mag + magnitude resample reached `18.69%` at `155` edges (`q=0.121`), but it remains validated evidence rather than the current recipe candidate.
- Current I/O architecture finding: tentacle I/O (4.7%) beats holographic projection (1.2%) by 3.9x — not yet promoted to mainline. 8-bit binary I/O (0.2%) rejected; high-dimensional spreading is load-bearing.
- Current best input encoding: SDR_64 (sparse 20%, peak 7.3%) — baked into `graph.py` as `input_mode='sdr'` option.
- Current best eval result: `20.0%` with out_dim=160, learnable theta, SDR_64 input, charge readout (H=256 test scale).
- Current next public build target: context-dependent task learning, with input-window injection, evaluation-path changes, and associative-memory probes under active evaluation

## What Matters Now

- The active public architecture line is [INSTNCT Architecture](INSTNCT-Architecture); earlier architecture lines are historical context only.
- The Diamond Code era is preserved on archive branch `archive/diamond-code-era-20260322`; it is not part of active `main`.
- The pre-beta surface freeze is preserved on archive branch `archive/instnct-surface-freeze-20260322`; raw result dumps, archived sweeps, and retired exploratory probes no longer live on active `main`.
- The strongest schedule and mutation results live in [Validated Findings](Validated-Findings) until code on `main` actually adopts them.
- The main unresolved public targets are context-dependent task learning, schedule-policy promotion, edge-representation promotion, and re-checking low-theta / low-scale against today’s stronger recipe stack.
- History, terminology, and page retirements now live here instead of being spread across separate glossary, roadmap, theory, and archive leaves.

## Preparing for v5.0.0 Public Beta

### Already in place

- [`v4.2.0`](https://github.com/VRAXION/VRAXION/releases/tag/v4.2.0) exists as the current latest stable GitHub release on `main`.
- [`docs/VERSION.json`](https://github.com/VRAXION/VRAXION/blob/main/docs/VERSION.json) now anchors the public release identity in the repo-tracked source layer.
- The README, Pages front door, and wiki Home surface now use the same explicit release framing for the current canon, the next milestone, and the internal code path.
- CI already runs compile sanity, the reference stress test, public-surface checks, wiki mirror dry-run, and a tiny cyclic smoke test.
- The active tracked surface is being narrowed so English remains the single first-class public lane while task-memory and GPU stay as secondary validation surfaces.
- Contributor and safety posture already exist through [`CONTRIBUTING.md`](https://github.com/VRAXION/VRAXION/blob/main/CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](https://github.com/VRAXION/VRAXION/blob/main/CODE_OF_CONDUCT.md), [`LICENSE`](https://github.com/VRAXION/VRAXION/blob/main/LICENSE), [`COMMERCIAL_LICENSE.md`](https://github.com/VRAXION/VRAXION/blob/main/COMMERCIAL_LICENSE.md), [`SECURITY.md`](https://github.com/VRAXION/VRAXION/blob/main/SECURITY.md), [Discussions](https://github.com/VRAXION/VRAXION/discussions), and private security reporting.

### Still planned before Public Beta

- Tighten the newcomer path with a beta-grade quickstart and explicit known-limitations language.
- Improve public intake so bug reports, usage questions, and security reports route cleanly under higher traffic.
- Keep canonical code, validated evidence, and experimental work visibly separated as outside traffic increases.

## Project Timeline

---

### Early 2026 — Diamond Code Era

- **Diamond Code / LCX** dominated the public architecture story. External memory, dreaming, observability, and Goldilocks-style probes before INSTNCT became the active center. See [INSTNCT Architecture](INSTNCT-Architecture).

---

### 2026-02-17 to 2026-02-22 — Foundation & Evidence Discipline

- **Engineering doctrine** and contributor policy split into stable homes. See [Engineering Protocol](Engineering).
- **Terminology cleanup** — readers no longer need legacy page chains.
- **Evidence discipline** hardened — schedule, depth, and mutation results moved into explicit evidence layer. See [Validated Findings](Validated-Findings).

---

### 2026-03-21 — Canonical Docs & Schedule Research

- **Repo-tracked docs became canonical**, GitHub wiki became mirror.
- **Schedule-control work** became the main live research frontier: 8 ticks, decay-aware scheduling, voltage/leak control.
- **Roadmap, theory, archive, glossary** collapsed into one timeline surface.

---

### 2026-03-22 — Recipe Consolidation & Surface Freeze

- **Triangle convergence** distilled into current fixed English recipe: `2 add / 1 flip / 5 decay`.
- **Sign+mag edge representation** — best quality-per-edge: `18.69%` at 155 edges (`q=0.121`).
- **Task-learning experiments** displaced older swarm line: window=2 superposition (`21.8%`), word-pair memory (`23.8%`).
- **v5.0.0 Public Beta** preparation started.
- **Diamond Code era** extracted to `archive/diamond-code-era-20260322`.
- **Pre-beta surface freeze** — English only first-class lane, bulky dumps moved to `archive/instnct-surface-freeze-20260322`.

---

### 2026-03-25 — Resonator Theory

- **Resonator Chamber theory** formalized with FlyWire validation (139K neurons, 16.8M connections). 6 empirical findings: 10% hub-inhibitors, binary weights sufficient, ticks = diameter, log2(N) scaling. Core thesis: *Inference emerges as the fixed point of destructive interference.* See [Resonator Theory](Resonator-Theory).

---

### 2026-03-27 — I/O Architecture Overhaul (11 experiments)

- **V5 forward-pass mismatch FIXED** — recipe had hardcoded v4.2 forward pass (wrong decay, C19, no hard reset). Now delegates to canonical `rollout_token()`. A/B smoke test: bit-identical.
- **Tentacle I/O validated (A/B/C/D sweep)** — tentacle I/O `4.7%` beats holographic projection `1.2%` by 3.9x. Random init > structured resonator init. See [INSTNCT Architecture](INSTNCT-Architecture).
- **8-bit binary I/O REJECTED** — `0.2%` vs random 64-dim `4.4%`. Too little signal richness.
- **SDR input encoding VALIDATED** — SDR_64 `7.3%` > multiscale `7.1%` > random `4.4%` > Fourier `3.6%`. Sparse 20% activation wins. **Baked into `graph.py`** as `input_mode='sdr'`.
- **Output encoding: random CONFIRMED** — SDR on output `3.4%` worse than random `7.3%`. Asymmetric I/O optimal: sparse in, dense out.
- **Learnable theta: 14.1% NEW PEAK** — full resample [0,16] converges to ~6-7. Init from 1.0 recommended.
- **Charge readout > state readout** — charge `14.1%` vs spike state `10.3%`. Continuous signal richer for 256-class prediction.
- **8-bit binary I/O v2** — with learnable theta: 8-bit in + dense out = `9.1%`. Binary input viable but SDR superior.
- **Unified visualization** shipped — replaces 7 scattered generators with single self-contained HTML.

---

### 2026-03-28 — Cross-validation & Gemini Collaboration

- **Potential-aware fitness REJECTED** — Gemini's proposal: `score + w * target_logit`. Standard `14.1%` > potential w=0.05 `11.3%` > w=0.10 `8.3%`. False positives via 64-to-256 random projection. May work with direct 256 output.
- **Zero-theta trap CONFIRMED** — cross-validated with Gemini. `THETA_INIT=0.0` nullifies C19 Soft-Wave. Already fixed by learnable theta.
- **Claude vs Gemini graph.py A/B** — same params: Claude `14.1%` > Gemini `11.3%`. C19 clip and batch refractory are load-bearing. Gemini branch not merged.
- **Output dim sweep: 20.0% NEW PEAK** — out_dim=160 (hidden=32) beats out_dim=64 (hidden=128, 14.1%). Richer readout matters more than hidden neuron count. Non-monotonic: dip at 128, peak at 160.
- **Direct 256 output (H=384)** — `7.1%`, too slow and too few hidden. Random projection is more practical.
- **8-bit / repeated binary output** — confirmed dead. Need 48+ random dims to separate 256 classes.
- **Fine output dim sweep** — confirmed 160 as peak: 144=16.6%, 152=18.0%, **160=20.0%**, 168=19.6%, 176=18.0%.
- **Multi-seed confirmation** — od=160 across 3 seeds: mean=18.2% std=0.6% [17.4-18.8%]. Theta converges ~6.2 every time. `DEFAULT_OUTPUT_DIM=160` baked into `graph.py`.
- **Scale sweep confirms phi ratio** — 0.625 output ratio tested at H=128 (17.2%), H=192 (19.2%), H=256 (20.0%), H=384 (19.4%). Phi nested downshift (`H/phi` output, `remainder/phi` input) predicts optimum within 2-4 neurons. Theta converges ~5-7 at every scale. Architecture follows golden ratio proportioning.

## Retired Surfaces and Replacements

<details>
<summary>Open retired surface map</summary>

| Retired surface | What it was | Why it was retired | Replacement |
|---|---|---|---|
| `Glossary` | Standalone terminology page | The useful live terms were short enough to live inline with the timeline and status surface. | `Key Terms` on this page |
| `Legacy Vault` | Archive index / historical directory | The archive index role was more useful as a curated timeline than as a standalone vault page. | `Project Timeline` and this table |
| `Hypotheses` | Open-question tracker | Active public open questions were already coupled to current status and promotion gates. | `Open Questions and Promotion Gates` on this page |
| `Theory of Thought` | Legacy theory / hypothesis ledger | It no longer described the active public architecture or the live open-question surface. | [VRAXION Home](Home), [INSTNCT Architecture](INSTNCT-Architecture), and this page |
| `Chapter 11 - Roadmap` | Public roadmap/status page | The project needed one timeline page, not a separate roadmap surface. | This page |
| `Diamond Code v3 Architecture` | Earlier architecture hub | It no longer described the active VRAXION architecture line. | [INSTNCT Architecture](INSTNCT-Architecture) and this page |
| `Proven Findings` | Earlier evidence hub | The active evidence layer now lives in one explicit findings page for the INSTNCT-era stack. | [Validated Findings](Validated-Findings) |

</details>

## Open Questions and Promotion Gates

| Topic | What still must be proven | What promotion would mean | Current status |
|---|---|---|---|
| Context-dependent task learning | Show that word-pair memory, framed tasks, and windowed input gains hold under reruns and stronger evaluation without collapsing back to context-free behavior. | Promote the task-learning line from active frontier to validated finding, and make it the next serious architecture-update candidate. | Active |
| Input-window promotion | Show that `w=2` superposition keeps winning across reruns and task families without creating unstable overflow or masking effects. | Promote a windowed injection policy from evidence into the current recipe discussion. | Active |
| Voltage-aware schedule pressure | Show that a voltage-style schedule policy wins on plateau accuracy under confirmation reruns, not only on isolated peaks. | Promote the policy from interesting schedule evidence to a stronger recipe candidate. | Active |
| Compact learnable schedule control | Show that a low-parameter learnable controller, such as the 3-angle tree, can match or beat the best fixed schedules without unstable drift or overflow. | Promote the controller from exploratory mechanism to validated schedule candidate. | Active |
| Edge representation promotion | Show matched-budget reruns that sign+mag + magnitude resample keeps its quality-per-edge advantage and justifies changing the current English candidate. | Promote a new edge format / mutation policy into the current recipe line instead of leaving sign+mag as evidence only. | Active |
| Decay resample promotion | Show that single-neuron decay resample in `[0.01, 0.5]` keeps winning over local perturbation across reruns and budgets. | Promote the resample mutation policy into the current recipe line. | Active |
| Low-theta / low-scale generalization | Re-run `INJ_SCALE=1.0` with low theta against the stronger current English recipe stack instead of the older baseline only. | Promote the low-scale line from older validated evidence into the current recipe discussion. | Active |

## Key Terms

<details>
<summary>Open key terms</summary>

**Current mainline**
What is actually shipped in code on `main`. If a public page and the code disagree about current behavior, the code wins.

**Validated finding**
An experiment-backed result that has not yet been promoted into the canonical code path.

**Experimental branch**
An active build target or design direction that is still under evaluation, not a live default.

**INSTNCT**
The current public architecture line for VRAXION: a self-wiring, gradient-free system where the learnable object is the evolving directed graph and its co-evolved neuron parameters.

**Artifacts (run bundle)**
The minimum files required to treat a run as evidence: `run_cmd.txt`, `env.json`, `metrics.json`, and `summary.md`.

**Fail gates**
The hard invalidation conditions for a run, such as OOM, NaN/Inf, step-time explosion, heartbeat stall, or VRAM guard breach.

**Engineering Protocol**
The run and evidence contract for VRAXION experiments: how runs are executed, validated, and promoted into public truth. See [Engineering Protocol](Engineering).

### Legacy Terms

**LCX (Latent Context Exchange)**
An earlier external-memory system from the Diamond Code era, preserved in `archive/diamond-code-era-20260322`, not part of the current INSTNCT public line.

**`v4`, `v22_ternary`, `v23_instnct_lm`**
Older local/archive line names that may still appear in notes, screenshots, or local workspaces. Treat them as historical context, not as active tracked public lines on `main`.

**Zoom gate / bottleneck projection / C19 activation / score margin**
Historical Diamond Code-era terms that still appear in older issues, notes, or screenshots. Treat them as legacy context, not as current INSTNCT defaults.

</details>

## Published Releases

- Current canonical public release: [`v4.2.0`](https://github.com/VRAXION/VRAXION/releases/tag/v4.2.0)
- Next public milestone: preparation toward `v5.0.0 Public Beta`
- GitHub Releases: [VRAXION releases](https://github.com/VRAXION/VRAXION/releases)
- Public update issues: [public-update label](https://github.com/VRAXION/VRAXION/issues?q=label%3Apublic-update)

Older sprint bundles, prerelease notes, and retired page histories are now summarized through the timeline above instead of being maintained as separate public lookup pages.

## Read Next

- [VRAXION Home](Home)
- [INSTNCT Architecture](INSTNCT-Architecture)
- [Validated Findings](Validated-Findings)
- [Engineering Protocol](Engineering)
