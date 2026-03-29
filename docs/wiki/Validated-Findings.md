<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Validated Findings

This page is the public evidence board for VRAXION. Use it for the strongest reproducible results, whether they are already **Current mainline**, still a **Validated finding**, or still only an **Experimental branch**.

## Best Current Evidence

- **Current mainline reference point:** [`instnct/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/instnct/model/graph.py) now ships explicit per-instance defaults `DEFAULT_THETA = 15.0`, `DEFAULT_PROJECTION_SCALE = 3.0`, `DEFAULT_EDGE_MAGNITUDE = 1.0`, plus per-neuron `theta` / `decay` and nonnegative charge dynamics.
- **Current recipe candidate on `main`:** [`instnct/recipes/train_english_1024n_18w.py`](https://github.com/VRAXION/VRAXION/blob/main/instnct/recipes/train_english_1024n_18w.py) now uses `8` ticks with a triangle-derived `2 add / 1 flip / 5 decay` schedule; it still uses the existing float signed edge mask.
- **Strongest schedule result so far:** voltage medium leak reached `22.11%` peak / `21.46%` plateau.
- **Best compact learnable control policy so far:** the 3-angle decision-tree schedule reached `20.05%` at `156` edges.
- **Best edge-representation quality result so far:** sign+mag + magnitude resample reached `18.69%` at `155` edges (`q=0.121`), but it is not promoted into the current recipe candidate or `graph.py` defaults.
- **Strongest current task-learning input result so far:** window=2 superposition reached `21.8%`, beating the `w=1` baseline by `+72%`.
- **Strongest current task-memory evaluation result so far:** word-pair log-likelihood eval reached `23.8%`, beating bigram cosine at `18.8%`.
- **Surface policy on `main`:** English remains the only first-class public lane; task-memory and GPU remain tracked only as secondary validation surfaces.

## Use This Page When

- you want the strongest evidence first
- you want to know which results are shipped, reproducible-but-unpromoted, or still active targets
- you need one page instead of issue archaeology

## Evidence Table

| Topic | Label | Result | Code status |
|---|---|---|---|
| Charge ReLU ([66ce511](https://github.com/VRAXION/VRAXION/commit/66ce511d58b71cecbd92adc04f307299b3fc414b)) | Current mainline | Replacing symmetric clip with nonnegative charge unlocked flip accepts and improved English eval by about `+6%` | Promoted into `graph.py` forward paths |
| Flip mutation ([#112](https://github.com/VRAXION/VRAXION/issues/112)) | Validated finding | `flip` beat float weight perturbation by `+1.89%` on English 1024n | Not promoted into `graph.py` defaults |
| Low theta + `INJ_SCALE=1.0` ([#113](https://github.com/VRAXION/VRAXION/issues/113)) | Validated finding | `scale=1.0 + theta=0.03` beat `scale=3.0 + theta=0.1` (`12.91%` vs `11.01%`) | Not promoted into `graph.py` defaults |
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
| Charge vs State Readout | Validated finding | Charge readout (`14.1%`) beats canonical state/spike readout (`10.3%`) in the tentacle I/O + SDR setup. Charge is continuous (richer signal for 256-class prediction), state is binary (less information). Spikes remain essential for internal propagation dynamics — this finding is readout-only. | Not promoted; recipe at `instnct/recipes/ab_readout_state_vs_charge.py` |
| 8-bit Binary I/O v2 (with learnable theta) | Validated finding | Retested with learnable theta: 8-bit in + 8-bit out still dead (`0.0%`). But 8-bit in + random 64 out reached `9.1%` — binary input works when paired with dense output. SDR input (`14.1%`) still superior. | Recipe at `instnct/recipes/ab_binary_io_v2.py` |
| Potential-Aware Fitness (Gemini proposal) | Validated finding | Adding `w * mean_target_logit` to fitness: standard (`14.1%`) > potential w=0.05 (`11.3%`) > potential w=0.10 (`8.3%`). False positives via 64-to-256 random projection. May work with direct 256 output. | Recipe at `instnct/recipes/ab_potential_fitness.py` |
| Zero-Theta Trap (cross-validated with Gemini) | Validated finding | `THETA_INIT=0.0` nullifies C19 Soft-Wave, blocking freq/rho learning. Independently confirmed by Claude (theta sweep) and Gemini (multiplicative analysis). Fix: learnable theta from 1.0 with full resample [0,16]. | Recipe fix pending promotion |
| Claude vs Gemini graph.py A/B | Validated finding | Same test: Claude (`14.1%`) > Gemini (`11.3%`). C19 clip and batch refractory are load-bearing. Gemini branch not merged. | `instnct/recipes/ab_claude_vs_gemini.py` |
| Output Dimension Sweep (multi-seed confirmed) | Validated finding | Sweet spot: out_dim=160 = `20.0%` peak, multi-seed mean=`18.2%` std=`0.6%`. Theta converges ~6.2. 160/256=0.625 near golden ratio complement. Baked into `graph.py`. | `instnct/recipes/sweep_output_dim.py`, `multiseed_od160.py` |
| Scale Sweep (phi ratio across H sizes) | Validated finding | 0.625 output ratio scales: H=128 `17.2%`, H=192 `19.2%`, H=256 `20.0%`, H=384 `19.4%`. Matches phi nested downshift. Theta ~5-7 at every scale. | `instnct/recipes/sweep_scale.py` |
| Phi Overlap I/O (new best) | Validated finding | **`20.8%` new peak** at H=256: in=out=158 (H/phi), overlap=60, pure hidden=0. Zero dedicated hidden neurons needed. Baked as `phi_overlap=True`. | `instnct/recipes/test_phi_overlap.py` |
| Decay: fix 0.16 constant | Validated finding | Fix 0.16 = `19.4%` vs learnable `20.8%` (-1.4% tradeoff). Phi/10 ratio. Int mode: subtract 1 every 6th tick (zero float ops). 0 bytes, 0 schedule cost. | `test_fix_decay.py` |
| Rho: fix 0.3 constant | Validated finding | int4=`15.2%` > fix=`14.5%` > float=`14.1%`. Fix chosen: 0 bytes, -0.7% tradeoff. | `ab_rho_float_vs_int.py` |
| Freq+Phase never trained | Validated finding | freq/phase were never in any recipe schedule — only `mutate_step()` drift (unused by recipes). All checkpoints contain random init values. Wave gating impact unknown until dedicated sweep. | `analyze_freq_phase.py` |
| Theta: int4 > float32 | Validated finding | Int4 `15.6%` > float32 `13.5%`. Smaller search space helps. Peak: theta=1 (relay, 25%) + theta=6 (compute, 8%). | `ab_theta_int4.py` |
| Theta Pareto (15 configs) | Validated finding | 3 winners: binary[1,15]=`10.1%`/1bit, quad[1,5,10,15]=`12.7%`/2bit, int4[1-15]=`15.6%`/4bit. Float32 DOMINATED. Int8 lazy (96% dead). Value 15 critical. Natural R/C/G zones. | `sweep_theta_bits.py` |
| Control Neurons / Binary Toggles | Validated finding | Meta-learning mutation strategy fails: linear ctrl `15.8%`, binary toggles `14.9%` (never converged), all < fix schedule `22.4%`. | `test_control_neurons.py`, `test_binary_control.py` |
| Tree-Wired Scaffold Init | Validated finding | 4-way tree scaffold `15.6%` < random 5% `22.4%`. Structured init consistently worse — evolution prefers chaos. | `test_tree_wiring.py` |
| Input: SDR confirmed best | Validated finding | SDR phi overlap `22.4%` > SDR 64 `14.5%` > FREQ `12.3%` > one-hot `9.5%`. Input needs sparse/discrete, output needs smooth/continuous. Asymmetric I/O. | `test_freq_input.py`, `test_onehot_input.py` |
| Output Projection: Language-Aware | Validated finding | FREQ_ORDER `22.4%` new peak > BIGRAM_SVD `21.8%` > RANDOM `20.8%` > RIVAL_PAIRS `19.0%`. Output topology must match target distribution. | `instnct/recipes/sweep_output_projection.py` |
| Full Overlap (100%) vs Phi Overlap | Validated finding | 100% overlap `14.7%` < phi overlap `20.8%`. Full overlap too noisy, phi ratio is optimal. | `instnct/recipes/test_full_overlap.py` |
| Learnable Schedule (empty start) | Validated finding | Empty start + learnable budgets: `14.9%` with only 89 edges (vs fixed sched 4.2%/5 edges). Evolved: add=4 flip=6 theta=0 decay=1 remove=10. Quality/edge 26x better than prefill. | `instnct/recipes/test_learnable_schedule.py` |
| Direct 256 Output | Validated finding | 256 direct neurons (H=384): `7.1%`, only 6 accepts. Slow + too few hidden. Random projection more practical. | `instnct/recipes/test_direct256_learnable.py` |

Raw run dumps, archived sweeps, and retired exploratory probes now live on `archive/instnct-surface-freeze-20260322`, not on active `main`.

<details>
<summary>Historical Context</summary>

The earlier `Proven Findings` page belonged to the Diamond Code / pre-INSTNCT era. It is no longer the active public evidence surface. Current evidence lives here, in `Validated Findings`.

- External memory was already established as load-bearing in the older architecture line.
- Routing-first / content-second learning appeared as a recurring pattern before the current INSTNCT work.
- Task difficulty had to match actual architectural capability; impossible tasks created false bottleneck diagnoses.
- Processing depth mattered more than simply widening the local context window.

Those historical findings are useful context, not current mainline claims.

</details>

## Read Next

- [VRAXION Home](Home)
- [INSTNCT Architecture](INSTNCT-Architecture)
- [Engineering Protocol](Engineering)
- [Project Timeline](Release-Notes)
- [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md)
