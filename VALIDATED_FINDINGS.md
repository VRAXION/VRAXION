# VRAXION Validated Findings

This page is the **canonical public evidence summary** for the repo.

It exists so the strongest current findings do not live only in issue traffic.

Repo-tracked docs are canonical. The GitHub wiki is treated as a mirrored secondary surface.

## What Matters Most Right Now

- **Current mainline:** [`instnct/model/graph.py`](instnct/model/graph.py) ships per-neuron `theta`, `decay`, `polarity`, `freq`, `phase`, `rho` with nonnegative charge dynamics, C19 Soft-Wave gating, Dale's Law inhibitory fraction, and refractory period — all active in both the single-token and batch forward paths.
- **Current recipe candidate on `main`:** [`instnct/recipes/train_english_1024n_18w.py`](instnct/recipes/train_english_1024n_18w.py) uses `8` ticks with a triangle-derived `2 add / 1 flip / 5 decay` schedule.
- **Strongest schedule result so far:** voltage medium leak reached `22.11%` peak / `21.46%` plateau.
- **Best compact learnable control policy so far:** the 3-angle decision-tree schedule reached `20.05%` at `156` edges.
- **Best edge-representation quality result so far:** sign+mag + magnitude resample reached `18.69%` at `155` edges (`q=0.121`), but it is not promoted into the current recipe candidate or `graph.py` defaults.
- **Strongest current task-learning input result so far:** window=2 superposition reached `21.8%`, beating the `w=1` baseline by `+72%`.
- **Strongest current task-memory evaluation result so far:** word-pair log-likelihood eval reached `23.8%`, beating bigram cosine at `18.8%`.
- **Binary Mask Finding:** Ternary masks (`{-1, 0, +1}`) are redundant; pure binary masks (`{0, 1}`) achieve equal accuracy (`86.5%`) because inhibition is implicitly handled by the input projection's negative values and per-neuron decay. This enables a **multiply-free forward pass**, which is `1.5x` to `3.4x` faster on edge hardware.
- **Surface policy on `main`:** English remains the only first-class public lane; task-memory and GPU remain tracked only as secondary validation surfaces.

## How To Read This Page

- **Current mainline**: what is actually shipped in code on `main`.
- **Validated finding**: experimentally supported, but not yet promoted into the canonical code path.
- **Experimental branch**: active target or design direction, not yet validated as the default.

## Current Mainline

The canonical code path for `main` is [`instnct/model/graph.py`](instnct/model/graph.py).

Current mainline defaults in that file:

- `DEFAULT_THETA = 15.0`
- `DEFAULT_PROJECTION_SCALE = 3.0`
- `DEFAULT_EDGE_MAGNITUDE = 1.0`
- `DEFAULT_DECAY = 1.0`
- `DEFAULT_RHO = 0.3` (C19 Soft-Wave modulation depth)
- `DEFAULT_INHIBITORY_FRACTION = 0.20` (Dale's Law)
- per-neuron `theta`, `decay`, `polarity`, `freq`, `phase`, `rho` — all co-evolved
- charge uses nonnegative ReLU-style dynamics in the forward pass
- C19 Soft-Wave gating active on spike decision in both forward paths
- refractory period enforced in both `rollout_token()` and `rollout_token_batch()`

Anything that differs from those settings should be described as a **Validated finding** or **Experimental branch**, not as the live default.

The current English recipe candidate on `main` is [`instnct/recipes/train_english_1024n_18w.py`](instnct/recipes/train_english_1024n_18w.py). It currently uses the triangle-derived `2 add / 1 flip / 5 decay` schedule with the binary edge mask. It is useful evidence, but it is not the canonical architecture default.

The current secondary validation recipe on `main` is [`instnct/recipes/train_wordpairs_loglik.py`](instnct/recipes/train_wordpairs_loglik.py). It remains important for task-memory evaluation, but it is not a second front-door default.

## Evidence Table

| Topic | Status | Result | Code status |
|---|---|---|---|
| Charge ReLU ([66ce511](https://github.com/VRAXION/VRAXION/commit/66ce511d58b71cecbd92adc04f307299b3fc414b)) | Current mainline | Replacing symmetric clip with nonnegative charge unlocked flip accepts and improved English eval by about `+6%` | Promoted into `graph.py` forward paths |
| Flip mutation ([#112](https://github.com/VRAXION/VRAXION/issues/112)) | Validated finding | `flip` beat float weight perturbation by `+1.89%` on English 1024n | Not promoted into `graph.py` defaults; candidate for the mixed swarm implementation |
| Low theta + `projection_scale=1.0` ([#113](https://github.com/VRAXION/VRAXION/issues/113)) | Validated finding | `scale=1.0 + theta=0.03` beat `scale=3.0 + theta=0.1` (`12.91%` vs `11.01%`) | Not promoted into `graph.py` defaults |
| 8 ticks + current English candidate schedule ([2b4de88](https://github.com/VRAXION/VRAXION/commit/2b4de887656d5061a944d7f85b0bb2a875f767e4), [36086a0](https://github.com/VRAXION/VRAXION/commit/36086a0a58b02dad3413f883fdfd7d153108ed66), [fdae6d6](https://github.com/VRAXION/VRAXION/commit/fdae6d6cd79a3554f23fbe94ab5412cea3e216d1)) | Validated finding | `8` ticks beat `6`, and the current candidate on `main` now uses the triangle-derived `2 add / 1 flip / 5 decay` schedule | Promoted into the current English recipe candidate on `main`, not into `graph.py` defaults |
| Decay resample mutation ([a5419e2](https://github.com/VRAXION/VRAXION/commit/a5419e22795af522afa2e2d8e292dd495f6c909f)) | Validated finding | Single-neuron resample in `[0.01, 0.5]` beat local decay perturbation and produced differentiated decay rates `[0.081-0.235]` | Not promoted into the current recipe or `graph.py` defaults |
| Voltage medium leak schedule ([b971613](https://github.com/VRAXION/VRAXION/commit/b971613550d881a7298690a2016339486e4c8244)) | Validated finding | Strongest schedule result so far: `22.11%` peak / `21.46%` plateau | Not promoted into the current recipe or `graph.py` defaults |
| Decision-tree schedule ([f7e6185](https://github.com/VRAXION/VRAXION/commit/f7e618511217d9b2905d93b30d7523a0be1fd79d)) | Validated finding | `20.05%` at `156` edges, with the best edge quality among the learnable schedule policies tested | Not promoted into the current recipe or `graph.py` defaults |
| Sign+mag + magnitude resample ([41f3622](https://github.com/VRAXION/VRAXION/commit/41f3622e654a79ffba0c95421b5e8a5c0f354364)) | Validated finding | Bool sign + uint8 magnitude with full magnitude resample reached `18.69%` at `155` edges (`q=0.121`), beating sign+mag free and delivering the best quality-per-edge result in the edge-format sweep without taking the best raw accuracy overall | Not promoted into the current recipe candidate or `graph.py` defaults |
| Window=2 input superposition ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)) | Validated finding | `w=2` reached `21.8%`, beating `w=1` at `12.7%` and all wider tested windows on the current task-learning sweep | Not promoted into the current recipe or `graph.py` defaults |
| Word-pair log-likelihood eval ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)) | Validated finding | `23.8%` on short associative-memory probes, beating bigram cosine at `18.8%` | Not part of the canonical mainline yet |
| Binary Mask ([0f9eba0](https://github.com/VRAXION/VRAXION/commit/0f9eba0a340f1a94165d21096054817d23f79038)) | Validated finding | Binary `{0, 1}` mask matches ternary accuracy (`86.5%`) because inhibition is handled by the input projection. Enables multiply-free forward pass (`3.4x` faster). | Promoted to `graph.py` multiply-free path |
| Hub-inhibitor architecture — FlyWire validation ([6823ce7](https://github.com/VRAXION/VRAXION/commit/6823ce7)) | Validated finding | 10% I neurons with 2x fan-out achieves 8/8 separation at H=64+; matches FlyWire 10.2% I ratio and 2x out-degree | Not promoted into `graph.py` defaults |
| Binary weight sufficiency — FlyWire validation ([bd90845](https://github.com/VRAXION/VRAXION/commit/bd90845)) | Validated finding | Binary edges match float at all tested scales; topology determines computation, not edge precision | Consistent with existing Binary Mask finding |
| Tick = diameter rule ([65f07be](https://github.com/VRAXION/VRAXION/commit/65f07be)) | Validated finding | Optimal ticks ≈ 1.0x network diameter; too few = trivial, too many = dead network. Diameter scales as log₂(N). | Current recipe uses 8 ticks (near-optimal for H=1024) |
| Context-dependent task learning ([48f2657](https://github.com/VRAXION/VRAXION/commit/48f26579fe882f5ae9e5eab4bbe1264963b4685a)) | Experimental branch | Current next build target: input-window injection, word-pair memory, and stronger evaluation for nontrivial tasks | Not part of the canonical mainline yet |
| Tentacle I/O vs Holographic Projection (A/B/C/D sweep) | Validated finding | Four-way sweep (H=256, IO=64, 3000 step budget, plateau detection): **A** HOLOGRAPHIC=`1.2%`, **B** TENTACLES_IO=`4.4%`, **C** TENTACLES_RANDOM=`4.7%`, **D** RESONATOR_INIT=`2.4%`. Winner: **Mode C** (random 5% prefill + BFS connectivity, tentacle I/O). Tentacle I/O (first V=input, last V=output, mask edges learn routing) beats holographic projection **3.9x**. Structured init (ring+triangles+hubs from FlyWire data) underperforms random init at this scale — mutation+selection sculpts better topology from random chaos than from pre-structured order. Random init recommended; resonator structure may help at H=1024+. | Not promoted into `graph.py` defaults; recipe at `instnct/recipes/ab_projection_vs_tentacles.py` |
| V5 forward-pass mismatch fix | Validated finding | Recipe `train_english_1024n_18w.py` had a hardcoded v4.2 forward pass diverging from canonical `graph.py` (subtractive vs multiplicative decay, wrong C19 formula, missing hard reset). A/B smoke test confirmed: old vs new `max_diff=13.8`, new vs canonical are bit-identical. Recipe now delegates to `SelfWiringGraph.rollout_token()`. | Fix promoted into the recipe on `main`; test at `instnct/tests/test_recipe_canonical_ab.py` |
| 8-bit Binary I/O vs High-Dimensional Projection | Validated finding | A/B test (H=256): 8-bit binary encoding (IO=8, 240 hidden) peaked at `0.2%` vs 64-dim random projection (IO=64, 128 hidden) at `4.4%`. Despite 112 extra hidden neurons, 8-bit I/O cannot provide enough signal richness. High-dimensional input spreading is load-bearing; compact binary encoding is insufficient for this architecture. | Not promoted; recipe at `instnct/recipes/ab_binary_io.py` |
| SDR Input Encoding | Validated finding | Input encoding sweep (H=256, tentacle I/O): SDR_64 (20% sparse, K=13/64 active per byte) peaked at `7.3%` vs random 64-dim (`4.4%`), Fourier (`3.6%`), and multiscale (`7.1%`). Sparse distributed activation provides cleaner signal than dense projections — `+66%` over random baseline. Flip mutation dominates under SDR (81 vs 9), indicating active rewiring around sparse patterns. | Baked into `graph.py` as `input_mode='sdr'` option; sweep recipe at `instnct/recipes/ab_input_encoding.py` |
| Output Encoding: Random vs SDR Readout | Validated finding | Random 64-dim output projection (`7.3%`) beats SDR output (`3.4%`) and SDR-32 (`0.0%`). Input benefits from sparsity (SDR), output benefits from dense random mixing. Asymmetric I/O encoding is optimal: sparse in, dense out. | Not promoted; recipe at `instnct/recipes/ab_output_encoding.py` |
| Learnable Theta with Full Resample | Validated finding | Per-neuron theta with full resample `[0,16]` converges to mean `~6-7` regardless of starting point. Best result: `14.1%` (start=1.0, converged=6.12, firing=82.6%) — `+93%` improvement over fixed theta=5.0 (`7.3%`). Low start (1.0) outperforms because topology co-evolves with rising threshold. From 6.18 start, barely moved to 7.02 — sweet spot confirmed. Schedule must include theta mutation with full range resample, not small perturbation. | Not promoted into `graph.py` defaults; recipe at `instnct/recipes/sweep_theta_learnable.py` |
| Charge vs State Readout | Validated finding | Charge readout (`14.1%`) beats canonical state/spike readout (`10.3%`) in the tentacle I/O + SDR setup. Charge is continuous (richer signal for 256-class prediction), state is binary (less information). Spikes remain essential for internal propagation dynamics — this finding is readout-only. The canonical `forward()` uses state readout, which was designed for the older holographic projection. | Not promoted; recipe at `instnct/recipes/ab_readout_state_vs_charge.py` |
| 8-bit Binary I/O v2 (with learnable theta) | Validated finding | Retested with learnable theta: 8-bit in + 8-bit out still dead (`0.0%`). But 8-bit in + random 64 out reached `9.1%` — binary input works when paired with dense output. SDR input (`14.1%`) still superior. 8-bit encoding lacks signal richness even with learnable theta. | Recipe at `instnct/recipes/ab_binary_io_v2.py` |
| Potential-Aware Fitness (Gemini proposal) | Validated finding | Adding `w * mean_target_logit` to fitness to reward partial progress: standard (`14.1%`) > potential w=0.05 (`11.3%`) > potential w=0.10 (`8.3%`). The bonus accepts false positives because the 64-to-256 random projection mixes all byte logits — a target logit increase doesn't reliably mean the network is heading the right direction. May work better with direct 256 output neurons where each neuron maps 1:1 to a byte. | Recipe at `instnct/recipes/ab_potential_fitness.py` |
| Zero-Theta Trap (cross-validated with Gemini) | Validated finding | `THETA_INIT=0.0` in the recipe nullifies C19 Soft-Wave: `0.0 * (1 + rho * sin(wave)) = 0.0` (clipped to 1.0). This blocks freq/rho learning — all Musical Gating mutations produce delta=0 and get rejected. Independently confirmed by both Claude (theta sweep showing 0.0 is dead) and Gemini (multiplicative nullification analysis). Fix: learnable theta starting at 1.0 with full resample [0,16]. | Recipe fix pending promotion to `main`; see learnable theta finding |
| Claude vs Gemini graph.py A/B | Validated finding | Same test, same params: Claude's `graph.py` (C19 clip `[1, MAX_CHARGE]` + batch refractory + SDR support) = `14.1%`. Gemini's `graph.py` (no clip, no batch refractory, int8 mask, schema v3) = `11.3%`. Clip and refractory are load-bearing features. Gemini's removal of these is a regression. | Gemini branch `feature/axonal-delay-v5.0` not merged; recipe at `instnct/recipes/ab_claude_vs_gemini.py` |
| Output Dimension Sweep (multi-seed confirmed) | Validated finding | Sweeping random output projection dimension (H=256, SDR_64 input, learnable theta): 16=`0%`, 32=`0%`, 48=`10.1%`, 64=`14.1%`, 96=`17.2%`, 112=`18.8%`, 128=`12.9%`, 144=`16.6%`, 160=`20.0%`, 176=`18.4%`. Fine sweep confirmed peak at 160. **Multi-seed (3 seeds): mean=`18.2%` std=`0.6%` range=[17.4%, 18.8%]**. Theta converges to ~6.2 across all seeds. Note: 160/256 = 0.625 is near the golden ratio complement (1 - 1/phi = 0.618). `DEFAULT_OUTPUT_DIM = 160` baked into `graph.py`. | Promoted: `DEFAULT_OUTPUT_DIM` in `graph.py`; recipes at `instnct/recipes/sweep_output_dim.py`, `instnct/recipes/multiseed_od160.py` |
| Scale Sweep (phi ratio across H sizes) | Validated finding | The 0.625 output ratio scales across network sizes: H=128 `17.2%`, H=192 `19.2%`, H=256 `20.0%`, H=384 `19.4%` (3000 step budget limited). Proportions: in=`0.25*H`, out=`0.625*H`, hidden=`0.125*H`, SDR K=`20%` of in_dim. Theta converges to `~5-7` at every scale. These ratios match a **phi nested downshift**: `H/phi=158` (output) then `remainder/phi=61` (input) — within 2-4 neurons of measured optimum. The architecture appears to follow golden ratio proportioning. | Recipe at `instnct/recipes/sweep_scale.py` |
| Phi Overlap I/O (new best) | Validated finding | **`20.8%` new all-time peak** at H=256 with phi overlap: in=out=`round(H/phi)`=158, K=32 (20%), overlap zone=60 neurons (both I and O), pure hidden=0. Beats non-overlap (in=64, out=160, hidden=32) at `20.0%`. The network needs zero dedicated hidden neurons when I/O zones overlap — the overlap zone serves as processing substrate. `phi_overlap=True` mode baked into `graph.py`. | Promoted: `phi_overlap` parameter in `graph.py`; recipe at `instnct/recipes/test_phi_overlap.py` |
| Full Overlap (100%) vs Phi Overlap | Validated finding | 100% overlap (in=out=H=256, every neuron both I and O): `14.7%`. Worse than phi overlap (in=out=158, 60 overlap): `20.8%`. Full overlap is too noisy — SDR K=51 activates too many neurons, no processing separation. The phi ratio (H/phi) is the optimal overlap amount, not maximum. | Recipe at `instnct/recipes/test_full_overlap.py` |
| Empty Start: Fixed vs Learnable Schedule | Validated finding | Empty start (0% init density) with fixed schedule: `4.2%`, 5 edges. With **learnable schedule budgets** (each op type gets a learnable repeat count that mutates): `14.9%`, 89 edges — **3.5x improvement**. The evolved schedule: add=4, flip=6, theta=0, decay=1, remove=10. Flip dominates (349 accepts), heavy pruning (remove=10). Quality per edge: `0.167` (26x better than prefill's `0.0064`). The network self-organizes an ultra-sparse 89-edge topology from nothing. | Recipe at `instnct/recipes/test_learnable_schedule.py` |
| 8-bit Output / Repeated Binary Output | Validated finding | SDR_64 in + 8-bit out = `4.8%`, SDR_64 in + 8-bit x8 repeated = `0.2%`. 8 dimensions cannot separate 256 classes regardless of repetition — repeated binary lives in 8-dim subspace. Random projection works because 64+ random vectors are near-orthogonal in high-dim space. | Recipes at `instnct/recipes/ab_sdr_in_8bit_out.py` |
| Direct 256 Output (H=384) | Validated finding | 256 direct output neurons (neuron[i]=byte i, no projection) with learnable theta: `7.1%`, only 6 accepts. Theoretically perfect separation but H=384 is slow (1.2 sps) and 64 hidden neurons insufficient. Random 64-dim projection on H=256 (`14.1%`) is more practical. | Recipe at `instnct/recipes/test_direct256_learnable.py` |
| Dale's Law neuron polarity ([03b5952](https://github.com/VRAXION/VRAXION/commit/03b5952)) | Current mainline | 20% inhibitory neurons with fixed polarity (`-1`) achieved `+6.2%` accuracy win; matches the FlyWire-validated 10–20% I-neuron ratio | Promoted into `graph.py` — `polarity` per-neuron, `DEFAULT_INHIBITORY_FRACTION = 0.20` |
| Refractory Period + Partial Spike Reset ([774b6c6](https://github.com/VRAXION/VRAXION/commit/774b6c6), [899d27e](https://github.com/VRAXION/VRAXION/commit/899d27e)) | Current mainline | Per-neuron refractory gating achieved `+7.5%` accuracy win; single-token and batch forward paths now both enforce it | Promoted into `graph.py` — `refractory` per-neuron, active in both forward paths |
| Musical Spiking — Learned Freq + Phase ([cf08d58](https://github.com/VRAXION/VRAXION/commit/cf08d58)) | Current mainline | Per-neuron `freq` and `phase` co-evolved with C19 Soft-Wave gating; enables oscillatory resonance in the spike decision | Promoted into `graph.py` — `freq`, `phase` per-neuron, wired into both forward paths |
| C19 Soft-Wave activation ([947004e](https://github.com/VRAXION/VRAXION/commit/947004e)) | Current mainline | Continuous threshold modulation via phase-recurring wave; replaces hard threshold with soft gating on spike probability | Promoted into `graph.py` spike decision path |
| Learnable Rho — C19 modulation depth ([471dbc8](https://github.com/VRAXION/VRAXION/commit/471dbc8)) | Current mainline | Per-neuron `rho` controls C19 Soft-Wave modulation depth; `+1.6%` accuracy win over fixed rho | Promoted into `graph.py` — `rho` per-neuron, `DEFAULT_RHO = 0.3` |

## Historical Context

The earlier `Proven Findings` page belonged to the Diamond Code / pre-INSTNCT era. It is no longer the active public evidence surface. Current evidence lives here, in `Validated Findings`.

- External memory was already established as load-bearing in the older architecture line.
- Routing-first / content-second learning appeared as a recurring pattern before the current INSTNCT work.
- Task difficulty had to match actual architectural capability; impossible tasks created false bottleneck diagnoses.
- Processing depth mattered more than simply widening the local context window.

Those historical findings are useful context, not current mainline claims.

## How To Read This Repo

- If a result is in an issue but not in the canonical code path, treat it as a **Validated finding** or **Experimental branch**.
- If a page claims a setting is a live mainline default, that claim should be verifiable in [`instnct/model/graph.py`](instnct/model/graph.py).
- If code and narrative disagree, the code wins for “Current mainline,” and the narrative should be fixed.
