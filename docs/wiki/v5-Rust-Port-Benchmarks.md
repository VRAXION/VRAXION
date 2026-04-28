# Rust Implementation Surface

This page is the public Rust-facing surface for the `instnct-core` lane: current validation checkpoints, implementation status, performance decisions, and the active frontier for the Rust port. Performance claims below are benchmarked on the target machine (Ryzen 3900X, Windows 11); evolution findings are tied to the evaluation regimes stated beside them.

## Current Rust Frame

- Rust is the canonical public implementation surface for **INSTNCT**, not a parallel port. The current mainline path on `main` is the bias-free threshold grower in [`instnct-core/examples/neuron_grower.rs`](https://github.com/VRAXION/VRAXION/blob/main/instnct-core/examples/neuron_grower.rs).
- The core systems substrate is in place: owned `Network`, rollback snapshots, mutation API, CSR acceleration, learnable int8 readout, SDR table, and checkpoint persistence.
- The current public release is [`v5.0.0-beta.6`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.6) (grower-based + Phase A→B→D research-line checkpoint, on top of the [`neuron_infer`](https://github.com/VRAXION/VRAXION/blob/main/instnct-core/examples/neuron_infer.rs) standalone inference CLI shipped in beta.2 and the public beta training runbook at [`docs/PUBLIC_BETA_TRAINING.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/PUBLIC_BETA_TRAINING.md)). The originally released tag [`v5.0.0-beta.1`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.1) remains the historical Rust language-evolution beta — the **24.6% peak** smooth-fitness + 1+9 jackpot result still belongs to that line, but it is no longer the active mainline path on `main`.
- The B0 engine-freeze contract lives in [`docs/GROWER_RUN_CONTRACT.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/GROWER_RUN_CONTRACT.md), with the frozen golden snapshot at [`instnct-core/tests/fixtures/grower_regression_golden.json`](https://github.com/VRAXION/VRAXION/blob/main/instnct-core/tests/fixtures/grower_regression_golden.json) (six tasks, mean val `88.417`, max test `100.0`, mean neurons `5.667`, max depth `6`).
- The B1 promotion gate is **byte/opcode v1**: `1 byte data + 4 opcode -> 1 byte` over a frozen 8-bit-head latent + exact LUT translator. The contract is [`docs/BYTE_OPCODE_V1_CONTRACT.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/BYTE_OPCODE_V1_CONTRACT.md), the canonical builder is [`instnct-core/examples/byte_opcode_grower.rs`](https://github.com/VRAXION/VRAXION/blob/main/instnct-core/examples/byte_opcode_grower.rs), and the golden fixture at [`instnct-core/tests/fixtures/byte_opcode_golden.json`](https://github.com/VRAXION/VRAXION/blob/main/instnct-core/tests/fixtures/byte_opcode_golden.json) records the direct bitbank negative control at `75.0%` and the exact translator at `100.0%` over the full `1024`-sample domain (`distinct_keys=1024`, `conflicting_keys=0`, `key_bits=61`, `total_neurons=61`).
- Earlier Rust research lines (Equilibrium Propagation + C19 bidirectional learning, GPU ternary exhaustive search, the EP video upscaler POC, the recurrent ReLU chip and Connection Point work, and the released `v5.0.0-beta.1` smooth-fitness language line) are preserved in the validation checkpoints below as historical proof trail, not as the current grower mainline.
- Detailed chronology, reversals, and cross-surface context live on [Research Process & Archive](Timeline-Archive). For the public website snapshot of this lane, see [Website: Rust](https://vraxion.github.io/VRAXION/rust/).
- The benchmark tables and language-evolution data below are **historical evidence from earlier Rust research lines** — they are preserved unchanged for chronology and proof-trail purposes, and they are not promises about the current grower mainline.

## Public Rust Validation Checkpoints

| Topic | Status | Why it matters |
|---|---|---|
| CSR skip-inactive | Current mainline | `8.7x` at H=256, `19x` at H=512, and `4.8x` at H=4096 showed that the forward pass no longer has to scan every silent neuron |
| Mutation A/B: topology > parameters | Validated finding | Simple add/remove matched or beat the full 10-op schedule on the easy discrimination task, showing what the Rust lane needed first |
| Binary activation cannot represent distributions | Validated finding | Trigram-style loss stayed near uniform, confirming that language readout still needs a learnable projection surface rather than pure binary voting |
| Theta floor / zero-theta collapse | Validated finding | Relay-floor theta values remain essential; zero-theta networks collapse into indistinguishable activation patterns |
| Learnable int8 `W` | Current mainline | After the rollback fix, the Rust language lane reached a reproducible `16.7%` 3-seed mean with only `0.6pp` spread |
| Smooth cosine-bigram fitness | **Current mainline** | Broke the `17-18%` ceiling: `21.7%` peak with 1+1 ES, `24.6%` peak with 1+9 jackpot. Now the default fitness in `evolve_language.rs`. |
| 1+9 jackpot selection | **Current mainline** | 9 candidate mutations per step, best wins. `24.6%` peak vs `21.2%` for 1+1. `evolution_step_jackpot()` added to library. Python parity achieved. |
| Addition learning (seq_5x5) | **Validated finding** | Empty-start network learns 0-4 + 0-4 addition: `70%` mean, `80%` peak with only ~87 edges (freq baseline `20%`). Prefilled networks (3400 edges) only reach `53%` mean. First proof of real computation + sparse superiority. |
| Stable multi-window seed audit | Experimental branch | Five-window reranking showed that single-window winners can mis-rank the landscape even when a top-line score looks clean |

| Compact types (i8/u8/u16) | **Current mainline** | Per-neuron data 24→6 bytes. -30% propagation time at H=4096, -9% at H=256. All tests pass, disk format V1 preserved via convert-on-load. |
| Skip-inactive spike | **Current mainline** | Only process neurons with charge > 0. -48.7% at H=4096 sparse, with 5 adversarial correctness tests. |
| Fully sparse tick O(active) | **Current mainline** | All operations (incoming clear, activation clear, charge decay, scatter) are O(dirty) not O(H). Dirty set tracks neurons with nonzero charge/activation. |
| Sparse input API O(k) | **Current mainline** | `SdrTable::sparse_pattern()` + `Network::propagate_sparse()`. Input injection O(k) where k = active input neurons (~5% of H). -62% at H=4K, -72% at H=100K. |
| Copy-on-write evolution | **Current mainline** | `MutationUndo` enum (9 variants) + `apply_undo()` for O(1) rollback. Eliminates snapshot clone on the ~99% rejection path. |
| ListNet topology representation | **Validated finding** | Sorted `Vec<Vec<u16>>` fan-out per source: 6x faster evolution steps than HashSet+CSR at identical accuracy. Tested on Steam Deck (AMD Van Gogh). Candidate for replacing `ConnectionGraph` in library. |
| Edge budget sweep (H vs density) | **Validated finding** | Fixed 20K edge budget: H=4096 sparse (0.1%, 28 edges) peaked at 21.2% vs H=256 dense (30%, 6553 edges) at 18.0%. Bigger sparse networks find better circuits with fewer edges. |
| Eval token sweep | **Validated finding** | Fixed 60K eval budget: 20 tokens x 3 jackpot trials = 249 step/s, 18.2% peak. Jackpot denoises cheap evals. |
| Overnight ListNet validation | **Validated finding (corrected)** | ListNet simpler but **not faster** than INSTNCT library at H≥512 in fair 1+1 comparison. INSTNCT CSR+skip-inactive wins by +26% (H=512) to +130% (H=2048). The initial "6x" claim compared different search strategies (1+1 vs 1+9 jackpot), not topology storage. |
| Interference reduction (edge cap) | **Validated finding** | Cap=100 (20.7% mean) > cap=300 (20.6%) > cap=1000 (19.8%) at H=1024. Fewer edges = less mutual interference per mutation = better search. Independent of topology representation. |
| Packed NeuronParams | **Validated finding** | Packing threshold+channel+polarity into one 4-byte `#[repr(C)]` struct gives 8-10% spike loop speedup vs separate arrays. All three are read together per neuron. Charge+activation should stay separate (write-back pollution at H≥2048). |
| **Edges don't matter on bigram** | **Validated finding** | Deterministic ablation: trained (298 edges) = removed (0) = random (298) = params-only (0) = all 20.3% on bigram. Phi-overlap short-circuits: input charge directly visible at output zone. |
| **Edges MATTER for computation** | **Validated finding** | Addition ablation: INSTNCT 56-60% with edges vs 4% without (+52-56pp). Task-dependent: lookup = no edges, computation = edges critical. |
| Addition sweep | **Validated finding** | INSTNCT H=256: **72% best** (1506 step/s). ListNet sep I/O H=128: 56% best (6630 step/s). INSTNCT wins on accuracy, ListNet on speed. |
| Addition freeze-crystal | **Validated finding** | Freeze-crystal (cap=50/cycle, 8 cycles) = identical to flat cap=300 baseline. Same seeds produce same accuracy (60/72/60%). Freeze adds no value on this task. |
| Addition tick sweep | **Validated finding** | 6 ticks (72%) > 12 ticks (64%) > 24 ticks (56%). Shallow circuit, more ticks = noise. |
| **Zero generalization** | **Validated finding** | 0% test accuracy on held-out addition examples across all splits (0-4 and 0-9). Memorization capacity ≈ 1 example per edge. |
| 72% ceiling unbreakable | **Validated finding** | Jackpot (1+9), edge cap (100/300/500), 5min runs, freeze-grow cycles — all produce same seed-deterministic result. Ceiling = SDR+init dependent, not search-budget limited. |
| Addition diagnose | **Validated finding** | Sum=4 (5 combos) = 20-40% worst. Sum=0,8 (1 combo) = 100% best. Network memorizes input-output pairs, overuses default predictions. |
| Stateful training fails | **Validated finding** | No-reset between examples: train drops from 65% to 10-30%. State carry causes noise, not procedural learning. Edge count floods to cap. |
| 0-9 addition scaling | **Validated finding** | 100 examples, 19 classes: 27-30% accuracy with ~35 edges. Confirms memorization capacity ≈ 1 example/edge. |
| Exhaustive search proof | **Validated finding** | Only 2/6561 ternary configs generalize on 8-input addition: [+1,+1,...,+1] and [-1,...,-1]. Both = uniform weights = SUM neuron. Generalizing solution = 0.03% of search space. |
| **Incremental build: 100% generalization** | **BREAKTHROUGH** | Build 1 neuron at a time, exhaustive search per step, freeze. 10 neurons → 100% train + 100% test. Previous: 256 neurons → 72% train, 0% test. The search space per step (3^19) is tractable; the full space (3^90) is not. Validates brain-like incremental development. |
| Tick robustness | **Validated finding** | The 10-neuron circuit only works at tick=8. The circuit learned timing, not algorithm. No-decay version same. Need tick-variable training. |
| Resting potential | **BREAKTHROUGH** | Per-neuron resting potential replaces explicit BIAS neuron. ALL 9 logic gates work with 2 neurons + resting + ternary edges, ZERO hidden. Turing-complete base. |
| Holographic vs pathway | **BREAKTHROUGH** | 1-step matrix multiply: 0.0025% solutions generalize. 8-tick pathway: 0% in 2M samples. Same W — holographic is fundamentally superior. |
| Shared W > layered W | **Validated finding** | Shared W tick=2 solves PARITY (100%). Layered W (frozen W1 + independent W2) = 87%. Recurrence = self-compatibility constraint that HELPS search. |
| Task hierarchy | **Validated finding** | ADD (1-tick linear) < PARITY (2-tick shared W) < MUL (unsolved). C19 fix = no improvement, but C19 per-neuron C = ADD+PAR 100%. |
| Activation sweep | **Validated finding** | Swish = best for MUL (75%). C19 per-C = best for ADD+PAR (100%+100%). ReLU = best overall. All normalizations (softmax, proportion) worse. Task-dependent activation = biologically correct. |
| GCD neuron | **Experimental** | Common denominator processing: a==b? best (88%) but GCD=1 on thermometer+ternary inputs. Concept valid, needs wider weight range. |
| **Readout = bottleneck** | **BREAKTHROUGH** | output/calibration→round BROKE multiplication (divides by zero when target=0). Nearest-mean readout unlocks MUL and 6 other tasks. The architecture was always capable — the readout was wrong. |
| **ALL 8 tasks solved** | **BREAKTHROUGH** | Float gradient + per-neuron bias + signed square + nearest-mean: ADD, MUL, SUB, MAX, MIN, a==b, \|a-b\|, PAR = ALL 100%. N=8 neurons, 72 float params. Per-connection bias WORSE (overparameterized). |
| Weight range scaling | **Validated finding** | Binary ±1 solves MUL at N=3 (margin=0.3, exhaustive 4K). ±2 margin=30. ±4 margin=73. More bits = bigger margin. More neurons compensate fewer bits. |
| Per-neuron > per-connection bias | **Validated finding** | Per-neuron bias (72 params) beats per-connection bias (112 params) on MUL with gradient. Fewer params = better convergence. |
| **C19 exhaustive ALU** | **Validated finding** | All 9 logic gates (AND/OR/NOT/XOR/NAND/NOR/XNOR/IMPLY/BUF) with 2 neurons + resting potential, zero hidden. 3-input: XOR3+MAJ = 1 neuron, full adder = 2. Complete 8-bit ALU + multiplier. Verilog synthesis: 6708 gates (~4× standard ALU). |
| **EP + C19 bidirectional learning** | **BREAKTHROUGH** | Equilibrium Propagation with C19 rho=8: **100% on ALL byte-level tasks** (popcount, nibble class, byte add). Contrastive Hebbian (W += not -=), 18-tick convergence. EP trained weights: 100% settle, only 63% feedforward — bidirectional dynamics ARE the computation. |
| **Cortex routing** | **Validated finding** | EP-trained cortex dispatches ALU ops at 100% float, 96.1% frozen. Non-math decisions: 99.6-100%. Full pipeline (256 scenarios): 83.6%. C19 > tanh for cortex tasks. |
| **EP ≠ feedforward** | **BREAKTHROUGH** | EP-trained weights give 63% in feedforward LutGate bake vs 100% with bidirectional settle. The settle dynamics cannot be replaced by a single forward pass. This is architecturally fundamental: settle hardware IS the computation, not an optimization. |
| **Greedy freeze** | **Validated finding** | Neuron-by-neuron int8 quantization with EP retraining: 100% with write-back (float eval). Weight quantization works — the int8 inference pipeline rescaling is the bottleneck, not the weights. Per-layer quantization > global. |
| **Int16 settle** | **Validated finding** | Integer-only EP inference: 96.1% (popcount), 99.2% (nibble class) vs float. Zero overflow (>>7 accumulator shift fix), 9-18 tick convergence, zero float hardware needed. 6 adversarial attacks tested. |
| **EP→Distill→LutGate** | **Validated finding** | EP trains oracle (all-input settle), random search builds feedforward LutGate neurons reproducing oracle. 100% on 8-bit popcount and nibble class. ByteAdd (16→5): 46% (search space too large for 5-bit output). |
| **Functional LUT** | **Validated finding** | Run EP on ALL possible inputs, store results. 100% accuracy guaranteed but only scales to ≤14-bit input (GPU exhaustive). Beyond 14 bit: combinatorially impractical. |
| **GPU ternary exhaustive** | **Validated finding** | RTX 4070 Ti SUPER: N=13 561K/s (8.5s), N=14 403K/s (36s), N=16 99K/s (22min est). GPU 31× faster than CPU (24 threads). VRAM-safe (1GB cap). Practical limit: N=14 per neuron. Sparse: 14 inputs/neuron, width unlimited. |
| **Video upscaler POC** | **Validated finding (negative)** | EP temporal upscaler (32→64→16, YCbCr, 3.1 KB): -0.8 dB vs bilinear. EP regression fails — contrastive Hebbian too weak for pixel-level continuous values. 2-layer → NaN. EP = classification, not regression. |

The rows above are preserved historical validation checkpoints from the released `v5.0.0-beta.1` Rust language-evolution line, the EP/C19 research, and the earlier spiking-network and Connection Point work. They are not the current grower mainline; for the active path on `main`, treat the bias-free grower contract in [`docs/GROWER_RUN_CONTRACT.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/GROWER_RUN_CONTRACT.md) and the byte/opcode v1 contract in [`docs/BYTE_OPCODE_V1_CONTRACT.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/BYTE_OPCODE_V1_CONTRACT.md) as the canonical surfaces, and use the golden fixtures (`grower_regression_golden.json`, `byte_opcode_golden.json`) as the only authoritative numbers.

## Implementation Status

`instnct-core` is now a real evolution substrate rather than a forward-pass demo. It owns topology, per-neuron controls, reusable state, checkpointing, and the language readout surface in one library-level package.

| Area | Status | Notes |
|---|---|---|
| Network core and propagation | Done | Owned `Network`, reusable workspace, CSR cache, integer-only forward pass |
| Rollback and mutation API | Done | Snapshot restore plus all 10 Python mutation operators ported |
| Readout and input surfaces | Done | Learnable int8 projection plus validated `SdrTable` |
| Persistence | Done | Genome save/load plus bundled checkpoint format |
| Init strategies | Done | Highway sweep completed; chain-50 remains the best public default, forest was rejected, and WS small-world init did not beat the ceiling on an edge-efficient basis |
| Crystallize / breed | Partial | Iterative crystallize compresses networks, but breed and overlay merges still have not produced a breakthrough |
| Pocket chains / depth lines | Active experimental branch | Pocket pair (2×H=256 via charge transfer) reached 19.6% peak at parity, but shared-interface tests did not produce a mergeable breakout path |
| Full English recipe parity | Planned | Port the strongest Python English recipe into the Rust library lane |

The "Implementation Status" rows above are preserved from the released `v5.0.0-beta.1` Rust language-evolution lane. They are not the active grower mainline. The current canonical grower path on `main` is the bias-free threshold builder in [`instnct-core/examples/neuron_grower.rs`](https://github.com/VRAXION/VRAXION/blob/main/instnct-core/examples/neuron_grower.rs); the live B0 freeze is the grower regression bundle and its golden snapshot, and the active promotion gate is byte/opcode v1. The Python `graph.py` lane remains in-repo as a historical reference/support surface, not the canonical English lane.

<details>
<summary>Crate structure</summary>

```text
instnct-core/src/
  lib.rs                - crate root and public API surface
  checkpoint.rs         - save_checkpoint / load_checkpoint
  evolution.rs          - evolution_step() with separated edge cap and quality gate
  init.rs               - InitConfig defaults and build_network()
  network.rs            - Network struct, snapshot, 10 mutations, CSR cache, genome bytes
  network/disk.rs       - Wire DTOs for genome save/load
  parameters.rs         - runtime constants and timing defaults
  projection.rs         - Int8Projection and rollback-safe weight handling
  propagation/
    mod.rs              - integer-only forward pass
    tests.rs            - propagation tests
  sdr.rs                - SdrTable encoding
  topology.rs           - ConnectionGraph and edge storage

instnct-core/examples/
  evolve_minimal.rs
  evolve_ab_test.rs
  evolve_language.rs
  circuit_trace.rs
  mutation_profile.rs
  deep_tune.rs
  highway_sweep.rs
  clustered_evolution.rs
  ratchet_evolution.rs
  voting_sweep.rs
  overlap_sweep.rs
  zone_diag.rs
  locality_sweep.rs
  topology_sweep.rs
  butterfly_knee.rs
  butterfly_scale.rs
  adaptive_butterfly.rs
  dense_bench.rs
  edge_cap_sweep.rs
  accept_sweep.rs
  chain_polarity_ablation.rs
  breed_v2.rs
  annealing_sweep.rs
  projection_sweep.rs
  ensemble_test.rs
  ensemble_scored.rs
  sdr_sweep.rs
  pocket_proto.rs
  pocket_chain.rs
  pocket_pair.rs
  pocket_breed.rs
  pocket_cross.rs
  pocket_evolve.rs
  csr_benchmark.rs
  time_breakdown.rs
  loop_breakdown.rs
  size_sweep.rs
  bottleneck.rs
```

Public exports include `Network`, `NetworkSnapshot`, `ConnectionGraph`, `DirectedEdge`, `propagate_token`, `PropagationConfig`, `PropagationState`, `PropagationWorkspace`, `evolution_step`, `EvolutionConfig`, `StepOutcome`, `Int8Projection`, `SdrTable`, `save_checkpoint`, `load_checkpoint`, and `CheckpointMeta`.

Lint policy remains `#![forbid(unsafe_code)]` plus `#![deny(missing_docs, unreachable_pub)]`.

</details>

## Performance and Optimization Decisions

| Topic | Current decision | Why |
|---|---|---|
| Integer types | `u32` / `i32` in the hot path, `u16` for indices, `u8` for channel | Matches register width, preserves inhibitory sign, keeps indices compact |
| Loop form | `for` range loops | Best auto-vectorization behavior |
| Scatter-add | `chunks_exact(4)`-style traversal | Best compiler lowering in the tested builds |
| Layout | SoA in the hot path | Bitpacking saves memory, not runtime, at the tested sizes |
| Phase gating | Cosine LUT | Smooth modulation beat coarse binary alternatives |
| Benchmark policy | Deterministic harness with explicit noise-floor control | Small wins are not credible without stability checks |

### Optimization Verdict

| Candidate | Current verdict | Why |
|---|---|---|
| Edge sort by target | Rejected | Consistently slower after deterministic reruns |
| `select_unpredictable` | Rejected | The apparent win was a run-ordering artifact |
| AVX2 `target_feature` | Rejected | Neutral at best and harmful at larger sizes |
| PGO (`gnullvm`) | Inconclusive | Workflow is valid, but the claimed perf gain did not survive isolation |
| Topology refactor | Kept | Reducing redundant edge storage is directionally real even if exact speedup magnitude is noisy at H=1024 |

### Forward Pass Performance Summary

| Platform | H=256 ns/token | H=1024 ns/token | Notes |
|---|---|---|---|
| Python (`graph.py`) | ~400,000 | - | Reference implementation, not a systems target |
| Rust (`instnct-core`, before topology cleanup) | ~27,000 | ~820,000 | Early port baseline |
| Rust (`instnct-core`, current) | ~27,000 | ~237,000 | H=1024 magnitude is directionally real but still measurement-sensitive |
| C (`instnct/edge/` — archived at tag `archives/python-research-20260420`) | ~14,000 | - | Best small-size throughput baseline, pre-Rust lane |

The important systems result is no longer "Rust can be made fast in theory." It is that the Rust lane now has a deterministic benchmark policy, a real sparse-runtime advantage through CSR skip-inactive, and a stable set of accepted hot-path decisions.

## Language Evolution Frontier

> **Historical framing.** The "Language Evolution Frontier" section below is preserved as the proof trail for the released `v5.0.0-beta.1` Rust language-evolution lane (smooth cosine-bigram fitness + 1+9 jackpot, **24.6% peak**). It is not the current mainline path on `main`. The current mainline is the bias-free threshold grower in [`instnct-core/examples/neuron_grower.rs`](https://github.com/VRAXION/VRAXION/blob/main/instnct-core/examples/neuron_grower.rs); the current public release tag is [`v5.0.0-beta.6`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.6). The grower lane does not run `evolve_language.rs`; treat all "now default in `evolve_language.rs`" claims below as defaults of the prior released beta line, not promises about the active grower path.

### Smooth fitness breakthrough (2026-04-06)

A deep comparison of the Python and Rust implementations revealed that the `17-18%` ceiling was caused by the **fitness function**, not the architecture. The Python lane uses cosine similarity against the bigram distribution (a smooth, continuous signal), while the Rust lane used binary argmax accuracy (a discrete step function with flat plateaus).

**Test 1 — Fitness function (6 seeds, 30K steps, 1+1 ES):**

| Fitness | Mean% | Best% | Peak% |
|---|---|---|---|
| Stepwise (argmax accuracy) | `15.6%` | `17.5%` | `19.1%` |
| **Smooth** (cosine-bigram) | `15.8%` | `20.5%` | **`21.7%`** |

**Test 2 — Jackpot (6 seeds, 30K steps, smooth fitness):**

| Search | Mean% | Best% | Peak% | Accept |
|---|---|---|---|---|
| 1+1 ES | `18.1%` | `20.1%` | `21.2%` | `6.2%` |
| **1+9 jackpot** | `18.2%` | `22.4%` | **`24.6%`** | `25.3%` |

Jackpot per-seed:

| Seed | 1+1 peak | Jackpot peak | Delta |
|---|---|---|---|
| 42 | `20.2%` | `21.9%` | +1.7pp |
| 123 | `18.8%` | `21.4%` | +2.6pp |
| 7 | `17.8%` | **`24.6%`** | **+6.8pp** |
| 1042 | `17.4%` | `20.3%` | +2.9pp |
| 555 | `17.8%` | `21.2%` | +3.4pp |
| 8042 | `21.2%` | `12.7%` | -8.5pp |

Both smooth fitness and 1+9 jackpot are now defaults in `evolve_language.rs`. The library exports `evolution_step_jackpot()` and `Int8Projection::raw_scores()`.

### Python vs Rust root cause analysis

Three factors explain the gap between Python (`24.4%`) and Rust (`21.7%`):

| Factor | Python | Rust | Impact |
|---|---|---|---|
| Fitness function | Cosine to bigram distribution (smooth) | Was argmax accuracy (discrete) — **now fixed** | +2.6pp peak |
| Search regime | 9-worker jackpot: best-of-9 mutations per step | 1+1 ES: one mutation per step | Not yet tested in Rust |
| W projection | Fixed (never mutated) | Co-evolves with topology (causes deadlock) | Not yet tested in Rust |

### Current frontier

| Area | What is settled | What remains open |
|---|---|---|
| Fitness function | Smooth cosine-bigram fitness is the proven default | Further fitness shaping (curriculum, multi-objective) could push beyond 24.6% |
| Search regime | 1+9 jackpot + smooth fitness = `24.6%` peak — Python parity | Higher N (25, 50), population-based methods (CMA-ES, MAP-Elites) |
| Readout surface | W co-evolution confirmed unhelpful (0% accept rate in adaptive test) | Fixed W with bigram-aware init (Python uses frequency-ordered projection) |
| Seed stability | Jackpot reduces variance (5/6 seeds > 20%) but one seed (8042) still underperforms | Longer runs, warm restarts, or breed across seeds |
| Addition learning | seq_5x5 (0-4 + 0-4) = **70% mean, 80% peak** from empty start (87 avg edges). Prefilled (3400 edges) only reaches 53% mean. | Scale to larger digits, multi-step arithmetic, curriculum transfer to language |
| Empty start (addition) | 0-edge start builds targeted circuits: 80% with 83 edges vs 64% with 3400 prefilled. Sparse = better signal for evolution. | Task-dependent: empty wins on addition, prefill wins on language. |
| Empty start (language) | Empty start: 23.2% peak with **167 edges**. Prefill: **25.8% peak** with 4533 edges. Both above Python 24.4%. | Prefill better for language (complex task needs more capacity), but empty remarkable edge-efficiency. |
| SDR density sweep | **20% is optimal**: 22.3% mean, 24.6% peak. 40% second (21.8% mean). Very sparse (5%) and dense (80%) both underperform. | SDR density settled at 20% |
| Tick sweep v2 (smooth+jackpot) | **ticks=6: 24.6% peak, 22.3% mean** (BEST). ticks=12: 23.0%. ticks=18: 22.9%. ticks=4: 20.1% (too short). All 15/15 complete. | Default ticks=6 confirmed optimal |
| Addition scaling | 5×5 (0-4): 78% mean, 84% peak from empty. **10×10 (0-9): ~10% = frequency baseline.** Does not scale to larger digit ranges with current architecture. | Need larger network, different encoding, or curriculum approach for 10×10 |
| Pocket pair (smooth+jackpot) | Prefilled pocket pair: **22.2% peak** (best seed). Crystal-first inconsistent. | Pocket pair needs guaranteed connectivity; crystallize too aggressive for Male. |
| Beyond Python parity | Architecture matched; further gains need new capabilities | Deeper pockets, breed/crystallize on jackpot-evolved networks, longer context eval |

## Archived Rust Research

<details>
<summary>Optimization and benchmark archive</summary>

### 2026-04-03 to 2026-04-04 - Systems defaults

| Topic | Candidate | Result | Outcome |
|---|---|---|---|
| Integer types | `u32` vs `u8` hot-path storage | `u32` = `27,000 ns/token`, `u8` = `67,500 ns/token` at H=256 | Keep `u32` / `i32` hot path |
| Loop form | `for` vs `while` | `for` was roughly `2x` faster (`27 us` vs `54 us` at H=256) | Keep `for` loops |
| Scatter-add | `chunks(4)` vs plain loop | `chunks(4)` won at H=256 and H=1024 | Keep chunked traversal |
| Layout | SoA vs bitpacked params | SoA stayed `1.2-8.5%` faster while packed was only smaller | Keep SoA for runtime |

The hot-path rule that survived these passes is simple: prefer compiler-friendly layout and loop structure over clever packing or opaque branch tricks.

### 2026-04-03 to 2026-04-04 - Phase-gating and temporal variants

**Cosine LUT vs Walsh family**

| Config | Final accuracy | Best accuracy | Outcome |
|---|---|---|---|
| Cosine LUT | **12.9%** | **13.1%** | Keep as default |
| Walsh 3-bool | 5.9% | 7.5% | Rejected |
| Walsh 2-bool | 6.1% | 6.1% | Rejected |

Walsh LUT matched cosine on speed, but the binary amplitude profile was too coarse for training. If Walsh-style structure returns later, it should select smooth harmonics rather than binary on/off thresholding.

**Threshold modulation sweep**

| Config | Best | Final | Outcome |
|---|---|---|---|
| Multiplicative cosine (`0.7..1.3`) | **14.3%** | **12.7%** | Confirmed |
| Additive positive only | 8.1% | 7.7% | Inferior |
| Additive symmetric | 8.1% | 2.8% | Inferior |
| Additive negative only | 5.7% | 2.0% | Inferior |
| No gating | 4.8% | 3.8% | Inferior |

Multiplicative modulation survived because it scales with threshold. Additive offsets over- or under-shoot low- and high-theta neurons unevenly.

**Float vs coarse integer cosine**

| Config | Best | Final | Outcome |
|---|---|---|---|
| Float cosine | **10.1%** | **6.9%** | Python-side reference |
| Integer x10 | 9.1% | 2.8% | Too much rounding loss |

The Rust lane already solved this by using x1000 fixed-point scaling, which preserves the cosine values without floats in the hot path.

**Other temporal variants**

- Rotation buckets (`x2/x1/x0/x-1`) were too coarse. Burst-style firing stayed biologically interesting but did not beat the smooth cosine schedule.
- Theta random init was rejected because the network needs a relay floor to bootstrap signal flow.
- Damping and per-tick parameterizations added search space much faster than they added useful selectivity.
- One-byte bitpacking achieved major memory savings, but its runtime upside only appeared at larger sizes and was not strong enough to replace the current SoA default.

### 2026-04-04 to 2026-04-05 - Benchmark hardening and rejected micro-optimizations

The deterministic harness changed the interpretation of multiple "wins." The public rule now is: any performance claim below roughly `10%` must survive a scalar-vs-scalar noise-floor control before it is promoted.

| Candidate | Initial claim | Deterministic rerun | Final outcome |
|---|---|---|---|
| Edge sort by target | Slower | Still slower | Rejected |
| `select_unpredictable` | Mixed, including large H=1024 win | Artifact from run ordering | Rejected |
| AVX2 `target_feature` | Small consistent win | Neutral or worse | Rejected |
| PGO | Large gain at H=1024 | Not isolated enough | Inconclusive |

**Forward benchmark hardening**

| Case | Rerun 1 | Rerun 2 | Noise-floor verdict |
|---|---|---|---|
| `propagate_h256_12ticks_i32` | `33.3 us` | `29.8 us` | stable / stable |
| `propagate_h1024_16ticks_i32` | `240 us` | `282 us` | stable / borderline |

The value of the harness is not identical reruns. The value is that the benchmark now tells the truth about when a throughput number should be read as a real claim and when it should be treated as a rough snapshot.

### 2026-04-04 - Library and bench housekeeping

- Variable naming was cleaned up where readability could improve without touching runtime behavior.
- Bench-only helpers were centralized so different benchmark binaries could not silently drift in policy.
- Documentation and `doc(hidden)` bench hooks were separated from the public API surface.

</details>

<details>
<summary>Language-eval archive</summary>

### 2026-04-05 - Early language route

- **Trigram cross-entropy failed.** Binary activation plus Laplace smoothing stayed near uniform and did not provide a viable evolutionary language signal.
- **Theta diagnostic clarified the floor.** Zero-theta relay behavior saturated the network into nearly identical patterns, which is why moderate or randomized nonzero theta values immediately improved differentiation.
- **Sequential next-char prediction became the main task.** This aligned the Rust lane with the Python-style language setup and gave the port a realistic public benchmark.

### 2026-04-05 - Learnable projection beats fixed random readout

| Variant | Step 1K | Step 3K | Step 5K | Step 8K | Final |
|---|---|---|---|---|---|
| Fixed `W` | 13.4% | 13.6% | 12.9% | 13.5% | **13.5%** |
| Learnable int8 `W` | 9.2% | **17.3%** | 14.3% | **17.8%** | **15.1%** |

The single-seed peaks above frequency baseline did not all survive later clean reruns, but the directional result did: the projection surface has to be learnable. Fixed random readout caps the lane too early.

### 2026-04-05 - Eval variance is part of exploration

| Eval method | Total chars | Final | Interpretation |
|---|---|---|---|
| 1x100 | 100 | **15.0%** | Baseline; noisy but alive |
| 1x500 | 500 | 8.4% | Too greedy |
| 3x100 | 300 | 4.6% | Even worse; variance reduced too far |

Longer or cleaner eval did not help. It removed the implicit exploration that noisy short eval was providing.

### 2026-04-05 - Paired eval fixed a real interpretation bug

| Metric | Unpaired | Paired |
|---|---|---|
| Final (5K chars) | 15.0% | **17.8%** |
| Accept rate at step 10K | 0% | **97%** |
| Trend | Plateau | Still climbing |

This was not a cosmetic fix. Comparing before/after on different text segments was contaminating mutation decisions with segment difficulty. Paired eval became a durable requirement for believable Rust comparisons.

### 2026-04-05 to 2026-04-06 - Multi-seed discipline and the stable band

- Multi-window seed audits showed that single-window winners can mis-rank the landscape.
- Clean 3-seed reruns with learnable int8 `W` settled near a reproducible `16.7%` mean with only `0.6pp` spread.
- The tested 1+1 ES recipe repeatedly converged into the same `17-18%` band, with several experiments still touching transient `19.1%` peaks.

**Consolidated public reading**

| Finding | Outcome |
|---|---|
| Learnable int8 projection | Required |
| Paired eval | Required |
| Multi-window seed discipline | Required for ranking |
| Stable convergence band | `17-18%` under the tested setup |
| Transient peak | `19.1%`, but not a stable breakthrough |

The important change in doctrine is that the ceiling no longer looks like a single bug or one missing scalar tweak. It now looks like a property of the tested exploration regime.

</details>

<details>
<summary>Topology, depth, and ensemble archive</summary>

### 2026-04-05 - Init strategy sweeps

| Experiment | Outcome |
|---|---|
| Highway structured init | Chain-50 emerged as the best practical default |
| Forest topology | Rejected |
| Structured init in general | Helps the floor more than the ceiling |

The durable lesson is that structured init can raise early convergence, but it does not by itself create a new accuracy regime.

### 2026-04-05 to 2026-04-06 - Topology and systems constraints

| Experiment | Result |
|---|---|
| Locality sweep | Rejected; random topology stayed better |
| Butterfly distance knee | Useful speed signal |
| Adaptive butterfly at H=1024 | Roughly `48%` propagation speed benefit |
| Projection dimension sweep | Not the bottleneck |
| SDR active-rate sweep | Not the bottleneck |
| Edge-cap sweep | Not the bottleneck |

This cluster matters because it removed multiple easy explanations. The Rust ceiling is not just bad SDR rate, bad output dimension, or a too-tight edge cap.

### 2026-04-06 - Ensemble, breed, annealing, and ratchet

| Line | Best reading |
|---|---|
| Ensemble oracle | Corrected top-4 oracle = `17.2%`; predictions are highly correlated |
| Breed v1 / v2 | Failed to create a breakthrough |
| Shared-female mergeability | Refuted; same function still emerged from different wiring, and merge did not recover complementary information |
| Simulated annealing | Did not beat the existing noisy-eval behavior |
| Ratchet + iterative crystallize | Improved edge efficiency by about `10.8x`, but did not break the ceiling |

Ratchet stayed valuable because it compresses networks into much smaller edge sets without destroying performance. Breed and annealing stayed valuable mainly as falsified routes.

### 2026-04-06 - Pocket-chain depth line

**Post-hoc chaining**

- Failed near random because pockets trained on SDR input cannot directly consume dense charge patterns from prior pockets.

**Spatial pocket-chain v2**

| Pockets | H | Mean | Best | Peak | Outcome |
|---|---|---|---|---|---|
| 1 | 256 | 4.3% | 5.2% | 9.2% | Weak control under custom mutator |
| 2 | 452 | **13.2%** | **14.9%** | **17.5%** | Viable |
| 4 | 844 | 8.4% | 16.4% | 16.9% | Mixed, seed-sensitive |
| 6 | 1236 | 2.6% | 2.9% | 8.6% | Failed |

The 2-pocket result keeps the depth idea alive, but the experiment still had confounds: no flat control at matching H, no edge cap, under-mutation at higher pocket counts, and a weaker custom mutator than the library evolution path.

**Pocket pair: two separate H=256 pockets chained via charge transfer (2026-04-06)**

Architecture change: instead of one Network(H=452) with spatial constraints, use two independent Network(H=256) pockets chained end-to-end. Female processes SDR input, her output charge feeds directly into Male's input (no learned projection between them). Only Male's output goes through the W projection. Trained end-to-end: mutation in either pocket, eval on the full chain. 10 units (5 pairs × F/M) evolved in parallel on 10 cores via rayon.

| Rank | Unit | Final | Peak | F_edges | M_edges | Accept |
|---|---|---|---|---|---|---|
| 1 | BM | **18.1%** | 18.1% | 3775 | 3495 | 22% |
| 2 | EM | 17.0% | **19.6%** | 3725 | 3673 | 28% |
| 3 | BF | 16.4% | 17.0% | 3557 | 3393 | 16% |
| 4 | AM | 15.5% | 17.3% | 3538 | 3351 | 12% |
| 5 | DM | 15.1% | 17.4% | 3647 | 3423 | 16% |
| 6 | EF | 14.5% | 17.5% | 3621 | 3327 | 16% |
| 7 | AF | 13.5% | 16.9% | 3812 | 3622 | 30% |
| 8 | DF | 5.9% | 12.2% | 3639 | 3510 | 21% |
| 9 | CF | 5.4% | 16.8% | 3598 | 3540 | 23% |
| 10 | CM | 3.7% | 5.9% | 3855 | 3607 | 29% |

Mean: 12.5%. Top-5 mean: 16.4%. **Peak: 19.6% (EM).** Runtime: 417s (10 units parallel).

This confirms that the charge-transfer chain works at parity with the single-network baseline, with EM reaching 19.6% peak but BM finishing strongest at 18.1%. The result kept depth alive as an experiment, but it did not justify promoting pocket breeding as the next default bet.

**Shared-female critical test**

| Signal | Result |
|---|---|
| Male Jaccard overlap | `2.5-3.6%` |
| Prediction agreement among top males | `88-98.5%` |
| Top-5 oracle | `15.8%` vs best single `17.0%` |
| Overlay merge | `5.6%` after merge |

This refuted the hope that one shared upstream Female would force compatible downstream Male pockets. The Males learned nearly identical predictions through almost completely different edge sets, so topology diversity still did not create complementary information to combine.

**Watts-Strogatz small-world init**

| Init | Mean | Peak | Edges | Reading |
|---|---|---|---|---|
| chain-50 | `14.3%` | `19.1%` | 3,601 | Better edge-efficiency baseline |
| WS k=20 / 40 | `16.7%` | `19.1%` | 5,117-10,240 | Mean lift came with much denser graphs |

The small-world line improved mean quality only by spending far more edges. It did not produce a new peak regime, so chain-50 remained the better public default.

</details>

<details>
<summary>Library milestones and side branches</summary>

### 2026-04-04 to 2026-04-06 - Core library milestones

| Milestone | Outcome |
|---|---|
| `Network` as the owned runtime object | Established the first real batteries-included Rust surface |
| `NetworkSnapshot` rollback | Enabled safe evaluate / restore loops |
| Full mutation API | Ported all 10 Python operators into the library lane |
| Checkpoint persistence | Bundled `Network` + `Int8Projection` + metadata into one atomic file |
| Cooperative evolution workers | Preserved as an important side branch for faster exploration |

**Checkpoint example**

```rust
save_checkpoint("run_42.ckpt", &net, &proj, CheckpointMeta {
    step: 30_000,
    accuracy: 0.175,
    label: "pocket_chain 2p seed=42".into(),
}).unwrap();
```

The checkpoint format uses atomic temp-and-rename writes plus adversarial tests for round trips, corrupted bytes, missing files, overwrite behavior, and functional propagation identity.

### Side branches that mattered mainly by failing

- Voting output without a learnable `W` matrix was rejected; the projection surface is essential.
- Deep tune and excessive per-parameter tuning were rejected; topology and exploration mattered more.
- Several housekeeping passes clarified API naming, bench hooks, and public exports without changing the canonical runtime decisions.

</details>

## Quantization championship (2026-04-17/18)

### Absolute winner: QAT int8 (Quantization-Aware Training with STE)

- nf=1024 FineWeb char-LM: **86.40% eval** vs pure float 86.20%
- 4× memory compression vs float32
- Training cost: ~20 sec on RTX 4070 Ti Super
- Beats pure float AND staged INQ in same task

### Pareto frontier (one entry per precision level, best of any method)

| Precision | Compression | FineWeb acc | Best method |
|---|---|---|---|
| float32 | 1× | 86.20% | long training |
| int8 | 4× | **86.40%** | QAT STE |
| int4 | 8× | 84.75% | staged INQ |
| binary | 32× | 71.50% | QAT STE |

### Protocol revelations (2026-04-17/18)

- The earlier "int4 +1.4pp win" finding was a **protocol artifact** — staged INQ gives 200 extra training epochs vs the float baseline. Apples-to-apples comparison (same epoch count, no quant): float equals int4 at ~84.50%; float with more epochs reaches 86.20%.
- Ternary's catastrophic 55% with staged INQ was a **protocol bug**, not a fundamental limit. QAT STE lifts ternary to 71.50% (+16.5pp). The staged scale/2 threshold over-prunes ternary; QAT avoids this.
- Binary "info-ceiling" at 49% was a **capacity limit**, not information limit. At nf=1024 binary reaches 70-71%. BitNet b1.58 literature confirmed.

### Failed approaches (documented negative results)

- Progressive growing + per-neuron int4: −14.85pp vs batch training at same params
- Generational cluster growth: −5.20pp vs single-shot
- Random-rotation sparse training: dominated by QAT
- Stacked exhaustive clusters: dominated by float+PTQ in every metric
- True ternary exhaustive D=16: 21.25% vs float 30.25% (−9pp sparsity cost)

### Deploy recommendations

- Cloud/server: QAT int8
- Mobile/edge: staged int4 (sweet spot at 8× compression, minimal accuracy loss)
- IoT/FPGA: QAT binary + Beukers gate LUT (native bit-ops, 32× compression)
- Micro-encoders only: true ternary exhaustive (guaranteed mathematical optimum, D ≤ 20 max)

### Reproduction artifacts

- Scripts: `tools/diag_quant_sweep_gpu.py`, `tools/diag_qat_ste.py`, `tools/diag_float_extended_control.py`, etc.
- Visualization: `docs/playground/quant_final_verdict.html` (Pareto frontier + per-precision best)
- Total evidence: 50+ experiment runs, ~2.5h total wallclock

## Read Next

- [Vraxion Home](Home) — mission-level front door
- [INSTNCT Architecture](INSTNCT-Architecture) — current implementation line
- [Theory of Thought](Theory-of-Thought) — theoretical framing
- [Research Process & Archive](Timeline-Archive) — chronology and retained proof trail
