# Vraxion Research Process & Archive

This page is the canon research record for Vraxion: the run contract behind public claims, the latest-first chronology of what changed, and the archive residue still worth keeping after the raw noise is stripped away.

## Current Frame

- The stable public release is still [`v4.2.0`](https://github.com/VRAXION/VRAXION/releases/tag/v4.2.0), while the active architecture line is [INSTNCT Architecture](INSTNCT-Architecture).
- The Rust `v5.0.0-beta` lane is now substantial enough to deserve rich chronology here, but it is still a beta implementation surface rather than the shipped default.
- The biggest unresolved pressure is no longer basic trainability; it is whether language evaluation, seed variance, and context-dependent task learning can survive repeated, adversarial reruns.
- [Vraxion Home](Home) is the mission-first front door, [INSTNCT Architecture](INSTNCT-Architecture) is the implementation explainer, [Website: Research](https://vraxion.github.io/VRAXION/research/) is the public website snapshot, and [Rust Implementation Surface](v5-Rust-Port-Benchmarks) carries detailed Rust validation. This page keeps the protocol, chronology, and retained research record in one place.
- `Confirmed` means backed by direct logs, code, charts, releases, or rerunnable evidence. `Inferred` means reconstructed from surrounding evidence. `Archived` means historically retained, not the live default.

## Research Protocol

This is the contract behind public research claims. It keeps shipped code, validated findings, and experimental lines separate even when the chronology gets messy.

- Every meaningful run needs an objective metric, one budget mode, hard fail gates, and a minimum evidence bundle.
- The minimum bundle is `run_cmd.txt`, `env.json`, `metrics.json`, and `summary.md`.
- A `Current mainline` claim must match code on `main`; reproducible but unshipped results stay under `Validated finding`, and active non-default work stays under `Experimental branch`.

### Required Evidence

| Artifact | Purpose |
|---|---|
| `run_cmd.txt` | Exact command and flags used for the run |
| `env.json` | Environment snapshot: OS, GPU/runtime, Python, package versions |
| `metrics.json` | Time series and summary metrics for the run |
| `summary.md` | Human verdict, including PASS/FAIL and the reason |

Optional extras such as checkpoints, plots, CSV exports, or live logs are useful, but they do not replace the core evidence bundle.

### Fail Gates

| Gate | Trigger |
|---|---|
| OOM / runtime failure | Any out-of-memory or driver/runtime failure invalidates the run |
| NaN / Inf | Any NaN or Inf in tracked metrics invalidates the run |
| Step-time explosion | `p95(step_time) > 2.5 × median(step_time)` |
| Heartbeat stall | No progress after warmup for `max(60s, 10 × median step time)` |
| VRAM guard breach | Peak reserved VRAM exceeds `0.92 × total VRAM` |

These gates apply to probes, sweeps, and training runs alike.

### Sweep Discipline

- Choose exactly one budget mode per sweep: `iso-VRAM`, `iso-params`, or `iso-FLOPs/step`.
- Run the systems curve first: throughput, stability, step-time tails, and resource limits.
- Only run the quality curve after the systems curve is stable.
- Start with a coarse sweep, then rerun the best cells with multiple seeds under the same contract.
- If a result does not reproduce under the same contract, treat it as unconfirmed.

### Status Labels

- **Current mainline** means the setting is actually shipped in code on `main`.
- **Validated finding** means the result is reproducible, but not yet promoted into the canonical code path.
- **Experimental branch** means the direction is active, but should not be described as a default.

If code and docs disagree about **Current mainline**, the code wins.

## Milestone Rail

| Era / jump | What changed | Where to read deeper |
|---|---|---|
| Early 2026 — Diamond Code era | Public story still centered on LCX / swarm / external-memory framing before INSTNCT became the active center. | Older Timeline below, [INSTNCT Architecture](INSTNCT-Architecture) |
| 2026-03-22 — Canon consolidation | Canonical docs hardened, archive branches were cut back, and the public line narrowed around English + evidence discipline. | [Vraxion Home](Home), entries below |
| 2026-03-27 to 2026-03-29 — I/O and schedule breakthrough | Tentacle I/O, SDR input, phi overlap, and compact learnable channel results clarified the current public architecture line. | [INSTNCT Architecture](INSTNCT-Architecture), [Vraxion Home](Home) |
| 2026-04-02 to 2026-04-05 — Rust v5 beta foundation | `instnct-core` became a real evolution substrate: owned `Network`, snapshots, full mutation API, CSR acceleration, genome persistence, and multi-seed parallel search. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-06 — Hyperparameter exhaustion and frontier narrowing | 11 tuning and strategy axes failed to lift the stable 17-18% band. Pocket-pair depth, shared-interface merge tests, and Watts-Strogatz init all clarified what does not create a new regime, while Rust library hardening continued around checkpoint persistence plus separated edge-cap/quality logic. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-06 (continued) — Pocket pair parity, shared-female refutation, and WS falsification | Two H=256 charge-chained pockets reached **19.6% peak** at parity only; shared-female tests kept male Jaccard near `3%` despite `88-98.5%` prediction agreement; Watts-Strogatz init matched the same ceiling with worse edge efficiency. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-07 — Performance deep dive and topology representation experiments | Smoke-port branch merged: compact types (-30%), skip-inactive (-49%), fully sparse tick O(active), sparse input API (-62-72%), CoW evolution snapshots. Steam Deck local benchmarking revealed ListNet (sorted 2D list topology) is 6x faster than INSTNCT's HashSet+CSR at identical accuracy. Branch cleanup: all merged/stale branches deleted, single `main` remains. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 — Overnight ListNet sweeps + fair comparison correction | Overnight: edge cap=100 > cap=1000 (interference reduction validated), 20% band is 1+1 ES ceiling, no cache cliff. **Correction**: the initial "6x faster" claim compared ListNet 1+1 vs INSTNCT 1+9 jackpot (unfair). Fair 1+1 vs 1+1: INSTNCT library wins at H≥512 (+26-130%) due to CSR skip-inactive. ListNet wins only at H=256 (+8%). Packed NeuronParams (threshold+channel+polarity = 4 bytes) gives 8-10% spike speedup at all H. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — EDGES DON'T MATTER on bigram (deterministic proof) | Ablation: trained (298 edges) = removed (0) = random (298) = params-only (0) = all 20.3%. The network learns purely from neuron params, not topology. Edge mutations are noise on the bigram task. Freeze-layer, burst, prune all confirmed same ceiling. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — Edges MATTER for computation: addition +52pp | Addition ablation: INSTNCT 56-60% with edges vs 4% without (+52-56pp). Edges are the computing substrate — irrelevant for lookup (bigram) but critical for computation (addition). Phi overlap short-circuits bigram: input charge is directly visible at output, no propagation needed. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — INSTNCT memorizes, does NOT generalize | Generalization test: 0% test accuracy on held-out addition examples. Train/test split on 0-4 and 0-9 ranges all show pure memorization. Memorization capacity ≈ 1 example per edge. The spiking network builds lookup tables, not algorithms. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — 72% ceiling unbreakable + stateful training fails | Push100 (jackpot + 5min + cap 100/300/500): all same 72% seed-deterministic. Freeze-grow: 0 new edges after cycle 0. Stateful (no reset): state carry causes noise (train 10-30% vs 60-65% with reset). The ceiling is SDR+init deterministic, not search-budget limited. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — Spike erases amplitude, fly brain analysis, LIF/Hebbian dead ends | Micro traces showed binary spike destroys input amplitude. Fly LIF (dual g+v) too complex for mutation search (5% train). Hebbian on random topology = no signal. Fly brain (Shiu 2024) uses additive synapses + dual exponential decay — fundamentally different from single-charge INSTNCT. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — Exhaustive proof: only 2/6561 configs generalize | On 8-input thermometer addition, exactly 2 ternary weight configs achieve 100% generalization: [+1,+1,+1,+1,+1,+1,+1,+1] and [-1,-1,-1,-1,-1,-1,-1,-1]. Both = uniform weights = SUM neuron. The generalizing solution is 0.03% of the search space. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| **2026-04-08 (continued) — INCREMENTAL BUILD: 100% generalization with 10 neurons** | **BREAKTHROUGH.** Build 1 neuron at a time, exhaustive search per step, freeze what works. 10 neurons achieve 100% train + 100% test (held-out sum=4). Previous best: 256 neurons → 72% train, 0% test. The key: each step searches a tiny space (3^19 max) instead of the full space (3^90). This is how the brain develops — incrementally, not all at once. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — Tick robustness: circuit is tick=8 only | The 10-neuron circuit only works at tick=8. Other tick counts → 0% test. The circuit learned TIMING, not algorithm. No-decay version same problem. Need tick-variable training for true algorithm. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — Resting potential replaces BIAS neuron | Per-neuron resting potential (decay toward resting, not toward 0) replaces explicit BIAS neuron. ALL 9 logic gates (AND/OR/NOT/XOR/NAND/NOR/XNOR/IMPLY/BUF) work with just 2 neurons + resting param + ternary edges, ZERO hidden neurons. Turing-complete base confirmed. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — Holographic vs pathway: mathematical proof | Holographic (1-step matrix multiply) has 0.0025% generalizing solutions. Pathway (8-tick shared W) has 0% in 2M samples. Same W matrix — the 1-step application generalizes, the 8-tick recurrence does not. Holographic is fundamentally superior for generalization. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — Shared W recurrence > layered W stacking | Shared W tick=2 solves PARITY (100%). Layered W (W1 frozen + W2 independent) stays at 87%. Shared W = self-compatible (auto-constraint), layered W = independent = harder to search. Recurrence is a feature, not a limitation. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — Task difficulty hierarchy confirmed | ADD (linear, 1-tick) < PARITY (XOR-like, 2-tick shared W) < MUL/equality/abs-diff (unsolved, need more). C19 activation = no improvement over ReLU at 1-step. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-08 (continued) — Activation function sweep + GCD neuron | Swish wins MUL (75%), C19 per-neuron C wins ADD+PAR (100%+100%). Standard normalization (softmax, proportion) all WORSE than ReLU. GCD neuron: a==b? best (88%) but thermometer+ternary gives GCD=1 always → not enough input diversity. Shared W recurrence confirmed > layered W stacking. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-09 — Overnight capability map + encoding breakthrough | Complete holographic task map: ADD/MAX/PARITY/\|a-b\|/MUL(area)/a÷b = 100% solved. MUL(thermo)=81%, a==b=87%, MIN=93%, SUB=75% unsolved. Area encoding proves MUL is an ENCODING problem not a network problem. Depth task-specific: helps PARITY, hurts SUB. Width/weight scaling don't help. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-09 (continued) — READOUT WAS THE BOTTLENECK: 7/10 tasks solved | Switching from output/calibration→round to nearest-mean readout unlocked MUL (100%!), SUB, MIN, a==b, |a-b|. The output/cal readout divided by zero for MUL (1×0=0). 3 neurons + integer ±2 + signed square + nearest-mean = 7/10 tasks at 100%. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-09 (continued) — Float gradient solves ALL 8 tasks | Per-neuron bias + float weights + numerical gradient + nearest-mean readout = MUL ✓, PAR ✓, a==b ✓ — all previously unsolved tasks now 100%. Per-connection bias WORSE (overparameterized). N=8 neurons, 72 float params, 30 starts × 10K gradient steps. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-09 (continued) — Weight range scaling | Binary ±1 solves MUL at N=3 (margin=0.3). More neurons = bigger margin: N=15 binary margin=13. ±2 N=3 margin=30. Bits × neurons = constant quality tradeoff. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-09 (continued) — CHIP COMPOSITION WORKS: 100% on 3-input addition | Frozen ADD chip (3 neurons) + searched wiring → 100% on ADD(a,b,c). Pipeline composition beats flat search (100% vs 92%). Perturbation finds solution in 14K steps. 4-input: 98.9% with 3 chained chips. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-09 (continued) — Recurrent chip: same W, multiple ticks | Same chip reused across ticks (one digit per tick). Signed square EXPLODES (7% at 3-input). Normalized: partial (78% 3→4). Key: activation function determines generalization. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| **2026-04-09 (continued) — ReLU PERFECTLY GENERALIZES across tick depth** | **BREAKTHROUGH.** ReLU is the ONLY activation achieving 100% recurrent generalization. 3 neurons, trained on 3-input → 100% on 2,3,4,5,6,7,8 inputs. 17/20 random seeds perfect. tanh=18%, sigmoid=3%, signed_square=0%. ReLU's max(0,x) is linearin positive range (preserves accumulation) but clamps negative drift. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-09 (continued) — ALL ops work with per-neuron bias, per-connection unnecessary | Tested ADD/MUL/MAX/MIN/AND/OR/XOR/NAND: per-neuron bias = per-connection bias on all ops. XOR also 100% generalization. MUL collapses beyond 4-input (bilinear, non-accumulative). | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| **2026-04-09 (continued) — MINIMUM VIABLE CHIP: ADD = 1 neuron, binary, no bias** | ADD works with 1 neuron, binary ±1 weights, ZERO bias. 5 bits total = 32 exhaustive configs. XOR needs 2 neurons minimum. MAX needs ternary + 2 neurons. Every config tested exhaustively. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| **2026-04-09 (continued) — NATIVE OUTPUT: charge IS the answer, no readout** | The 1-neuron ADD chip W=[1,1,1,1,1] bias=0 outputs charge = sum EXACTLY. No nearest-mean, no centroids, no calibration. 10-input (9.7M examples): 100% with just round(charge). COUNT chip also works natively: W=[1,1,0,0,0]. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-09 (continued) — Byte ALU: binary encoding harder than thermometer | 8-bit binary in/out: ADD 28%, XOR 25% — carry propagation too hard. Hybrid thermo→binary: only OR works. Binary encoding is fundamentally harder for neural chips than thermometer. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| 2026-04-09 (continued) — Multi-seed search fixes OR generalization | OR was never fundamentally broken — bad seed was the problem. 8-seed search → OR 100% at all depths. Readout method doesn't matter (tested 5 methods, all ~equal). The bottleneck is chip weights/seed, not readout. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| **2026-04-09 (continued) — Connection Point architecture VALIDATED** | Shared bulletin boards between neurons. Each neuron reads CPs + local neighbors + input, writes to one CP. Search space CONSTANT: 12 params = 531K from neuron 4 onwards. Recurrent ADD through CP: 100% at all depths. CP provides 1-tick delayed shared register for inter-cluster communication. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| **2026-04-09/10 — CONNECTOME ARCHITECTURE: passive relay wins** | Sweep: passive/active/sparse connectome × 5 tasks × 8 seeds. **Passive relay wins** (96% mean |a-b| vs 94% active). Sparse random connections = zero benefit. Binary ±1 = ternary performance (100x faster). Connectome needed ONLY for |a-b| (72% → 100%). 4 architectures tested, passive simplest+best. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| **2026-04-10 — SOLUTION DENSITY: binary space has ZERO native-output solutions** | Exhaustive scan of all 8192 binary configs for 1 worker: ZERO achieve even 40% accuracy (native output). The gradient does ALL the work — binary search is just a random starting point. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| **2026-04-10 — FLOAT GRADIENT: 200/200 solve rate for ADD** | Random float init + gradient descent: ADD 200/200 (3 workers), \|a-b\| 199/200 (6 workers), MUL 158/200 (6 workers). Init scale=0.5 best. Loss landscape is smooth — one big valley, no local minima. Gradient converges in ~50-100 steps. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| **2026-04-10 — INT8 QUANTIZATION: float train → integer deploy** | Post-training quantization: int4 (16 levels) = 100% for ADD on single seed. int8 (256 levels) = 85/100 seeds survive for ADD. Step size 0.008 = max error 0.004. The pipeline: float gradient → i8 quantize → integer-only inference. Native Rust `i8` type, already used in `Int8Projection`. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| **2026-04-10 — 2D LOSS LANDSCAPE: smooth valley, no traps** | ASCII heatmap visualization confirms: ADD landscape is one smooth funnel. Close-up: ●●● zone (96-99%) fills most of ±0.3 around solution. Wide view: small bright peak in ░ desert. Very wide (±5): needle in haystack, but gradient ALWAYS finds it because the slope is consistent. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks) |
| Current frontier | **Float gradient + connectome + native output = the pipeline.** Train: random float init → MSE gradient (50-100 steps) → 100%. Deploy: i8 quantize → integer-only inference. Passive connectome (relay) for inter-cluster communication. Next: benchmark i8 inference speed, test on harder tasks (language/bigram), integrate with INSTNCT library. | [Rust Implementation Surface](v5-Rust-Port-Benchmarks), [INSTNCT Architecture](INSTNCT-Architecture) |

## Latest Entries

---

### 2026-04-08 — INCREMENTAL BUILD: 100% generalization with 10 neurons (BREAKTHROUGH)

**Status:** Confirmed

**What changed**

- **Incremental building** achieves what no other approach could: 100% train AND 100% test (generalization to held-out sum=4 examples) on addition, using only 10 neurons.
- The method: start with 0 neurons. Add 1 neuron at a time. For each new neuron, exhaustive (or large random sample) search over all possible ternary connections to existing neurons. Keep the best config. Freeze. Add next neuron.
- Growth trace: neuron 0 = 0%, neuron 1 = 60% test, neuron 5 = 100% test (30% train), neuron 9 = 100% both.
- Previous best with 256 neurons (any method): 72% train, 0% test. The incremental approach with 10 neurons completely dominates.

**Insight chain that led here:**
1. Spike erases amplitude (microscope proof) → need continuous charge
2. Only 2/6561 ternary configs generalize (exhaustive proof) = 0.03% of search space
3. Both winners = uniform weights [+1,+1,...] or [-1,-1,...] = SUM neuron
4. SUM neuron abstracts perfectly (same sum → same charge) but readout was wrong
5. The generalizing solution EXISTS but random/mutation search can't find it (0.03%)
6. Incremental build solves this: each step searches 3^19 (tiny) instead of 3^90 (impossible)
7. Frozen layers provide stable foundation for next neuron → no interference

**Why it mattered**

- This is the first time the project achieved **generalization** (test accuracy on unseen examples > 0% consistently).
- It validates the "brain development" hypothesis: the brain grows incrementally (embryo → adult), adding neurons to a functioning system. Not random search over the full space.
- It proves the architecture CAN generalize — the bottleneck was always the SEARCH METHOD, not the neuron model, not the topology representation, not the edge weights.

**Evidence**

| Neurons | Train | Test (held-out) | Method |
|---:|---:|---:|---|
| 10 (incremental) | **100%** | **100%** | build 1-at-a-time, freeze |
| 256 (INSTNCT) | 72% | 0% | try-keep-revert all-at-once |
| 256 (ListNet) | 56% | 0% | try-keep-revert |
| 64 (LIF) | 5% | 0% | try-keep-revert |
| 64 (Hebbian) | 5% | 0% | reward-modulated |

**Promotions / Rejections**

- **Promoted:** Incremental neuron-by-neuron building as the primary growth strategy. Thermometer encoding for numeric inputs. Continuous charge readout (not binary spike). Every-neuron-is-I/O topology.
- **Confirmed:** The generalizing solution is rare (0.03%) but findable with incremental search. Frozen layers prevent interference.
- **Rejected:** All-at-once search (try-keep-revert on full network), Hebbian learning on random topology, LIF with mutation search, stateful training, freeze-crystal cycles.

---

### 2026-04-08 — 72% ceiling unbreakable, stateful fails, zero generalization confirmed

**Status:** Confirmed

**What changed**

- **Push100** (jackpot 1+9, 5min/seed, edge_cap 100/300/500): every configuration produces the same seed-deterministic result. Seed 42=60%, seed 1042=72%, seed 2042=60%. The ceiling is not search-budget limited.
- **Addition diagnose**: the network predicts by memorizing input-output pairs, not by computing. Sum=4 (5 input combos) = worst at 20-40%. Sum=0 and sum=8 (1 input each) = easiest to memorize. Prediction histogram shows overuse of a few default outputs.
- **Freeze-grow** (prune-freeze cycles): flatlines after cycle 0. The evolution adds ~19 edges in the first cycle, then finds 0 beneficial mutations in subsequent cycles. The 60% ceiling is reached in one shot.
- **Stateful training** (no reset between examples): state carry causes noise, not procedural learning. Train accuracy drops from 60-65% (with reset) to 10-30% (no reset). Edge count floods to cap (300). Test generalization: still 0%.
- **0-9 addition** (100 examples, 19 classes): only 27-30% accuracy with ~35 edges. Memorization capacity ≈ 1 example per edge, confirming the lookup table model.

**Evidence**

| Experiment | Key result |
|---|---|
| Push100 all configs | 72% max, seed-deterministic |
| Diagnose sum=4 | 20-40% (worst, 5 combos) |
| Diagnose sum=0 | 100% (best, 1 combo) |
| Freeze-grow | 0 new edges after cycle 0 |
| Stateful train | 10-30% (noise, worse than reset) |
| 0-9 addition | 30% at 35 edges (≈ 1 ex/edge) |
| Generalization | 0% test on all splits |

**Promotions / Rejections**

- Confirmed: pure memorization. No algorithmic generalization in any tested configuration.
- Rejected: jackpot as ceiling breaker (same result as 1+1). Freeze-grow (flatlines). Stateful training (makes things worse). More ticks (degrades). Higher edge cap (same seed-deterministic ceiling).
- The limiting factor is not search budget, edge cap, propagation time, or state management. It is the **architecture's inability to build compositional/algorithmic circuits** from the current spiking dynamics + random SDR encoding.

---

### 2026-04-08 — Edges MATTER for computation tasks (addition +52pp ablation)

**Status:** Confirmed

**What changed**

- Addition task (a+b, a,b in 0-4, 25 examples, random=11.1%):
  - INSTNCT library: **56-60%** with edges, **4%** without = **+52-56pp edge contribution**
  - ListNet (separated I/O): **24-44%** with edges, **4%** without = **+20-40pp**
- This definitively answers the "do edges matter" question: **task-dependent**.
  - **Bigram** (next char prediction): edges = 0pp contribution. The phi-overlap geometry lets the projection read input charge directly — no propagation needed. A lookup task.
  - **Addition**: edges = +52pp contribution. The input (two digits) must be COMBINED to produce the sum. Edge topology builds the computing circuit that transforms input→output. Without edges, no signal reaches the output zone.

**Why it mattered**

- Resolves the apparent contradiction between "topology is knowledge (28.5σ)" findings and the overnight "edges don't matter" ablation. Both are correct — for different tasks.
- The phi-overlap geometry was specifically designed for bigram/language where input→output proximity helps. For computation tasks (addition, logic), separated I/O is more appropriate.
- INSTNCT library outperforms ListNet on addition (56-60% vs 24-44%) despite both using the same edge count (~20-55). The library's CSR propagation, phi geometry, and Int8Projection produce better charge patterns.

**Evidence**

| Config | Task | With edges | Without edges | Edge diff |
|---|---|---:|---:|---:|
| INSTNCT library | Addition | 56-60% | 4% | **+52-56pp** |
| ListNet sep I/O | Addition | 24-44% | 4% | **+20-40pp** |
| INSTNCT library | Bigram | 20.3% | 20.3% | 0pp |
| ListNet phi | Bigram | 20.3% | 20.3% | 0pp |

**Promotions / Rejections**

- Promoted: task-dependent edge importance as core architectural understanding. Addition as the canonical "edges matter" task. Separated I/O as the correct geometry for computation tasks.
- Corrected: "edges don't matter" downgraded from universal finding to bigram-specific finding.
- Retained: all bigram ablation results remain valid. The phi-overlap short-circuit is a feature for language tasks, not a bug.

---

### 2026-04-08 — EDGES DON'T MATTER on bigram (deterministic ablation proof)

**Status:** Confirmed

**What changed**

- Deterministic ablation on H=512 network trained for 50K steps on 100KB Alice corpus:
  - Trained network (298 edges): **20.3%** (20316/100000)
  - All edges REMOVED: **20.3%** (20316/100000) — identical
  - Random edges (298): **20.3%** — identical
  - Params-only trained (zero edges ever): **20.3%** (20317/100000)
- The network achieves 20.3% using **only neuron parameters** (threshold, channel, polarity). Edge topology contributes zero measurable accuracy.
- This was confirmed by three independent approaches: (1) expensive 200-token eval rejected all edges as noise, (2) freeze-crystallize found no edges worth freezing, (3) deterministic full-corpus ablation showed exact equality.

**Why it mattered**

- This invalidates all topology representation experiments (ListNet, VarNet, FireNet, FlatNet, INSTNCT CSR) as accuracy-relevant for this task. The speed differences are real, but the edges themselves are payload-free.
- The bigram task is solvable via input→output direct projection through the phi-overlap zone. Input neuron parameters (channel/threshold) create sufficient frequency-selective filtering to separate bigram distributions. Hidden-to-hidden edges add nothing.
- Breaking beyond 20.3% requires either (a) a harder task where multi-hop signal propagation through edges is necessary, or (b) a learnable readout that can extract richer information than argmax over output neurons.

**Evidence**

| Config | Accuracy | Correct/Total | Edges |
|---|---:|---:|---:|
| Full trained | 20.3% | 20316/100000 | 298 |
| Edges removed | 20.3% | 20316/100000 | 0 |
| Edges restored | 20.3% | 20316/100000 | 298 |
| Random edges | 20.3% | 20316/100000 | 298 |
| Params-only | 20.3% | 20317/100000 | 0 |

**Promotions / Rejections**

- Promoted: "edges don't matter on bigram" as a confirmed finding. Neuron parameters (threshold/channel/polarity) are sufficient for the 20.3% ceiling.
- Rejected: all freeze-layer strategies (naive, prune, burst) as ineffective — the problem was not noise in crystallize, but that edges have zero signal on this task.
- New direction needed: a task where topology demonstrably matters (e.g., addition, sequence memory, multi-class classification beyond bigram).

---

### 2026-04-08 — Fair comparison correction + memory layout profiling

**Status:** Confirmed (corrects previous entry)

**What changed**

- The initial overnight "6x faster" ListNet claim was based on an **unfair comparison**: ListNet used 1+1 ES (2 evals/step) while INSTNCT used 1+9 jackpot (10 evals/step). The speedup was mostly from doing fewer evaluations, not from topology storage.
- A fair 1+1 vs 1+1 comparison showed **INSTNCT library wins at H≥512**: H=512 +26%, H=1024 +65%, H=2048 +130%. The CSR + skip-inactive optimization in the library is highly effective at larger H. ListNet wins only at H=256 (+8%).
- Memory layout profiling identified two optimizations:
  - **Packed NeuronParams** (threshold + channel + polarity in one 4-byte struct): **8-10% faster** spike loop vs separate arrays at all H values. The three fields are always read together per neuron.
  - **Charge + activation stay separate**: packed state helps at H≤512 but hurts at H≥2048 due to write-back cache pollution. Separate is safer across all H.
- Four alternative topology representations were benchmarked end-to-end:
  - ListNet (sorted Vec<Vec<u16>>, scatter): 3917 step/s at H=256
  - VarNet (fixed fan-in=3, gather): 3721 step/s — degrades at H>1024
  - FireNet (fan-in gather, no scatter): 3276 step/s — activation clone overhead
  - FlatNet (fixed [u16; 16] arrays): 3740 step/s — wastes memory

**Evidence**

| Metric | Result |
|---|---|
| Fair 1+1 comparison H=256 | OptNet 3428 vs INSTNCT 3167 = OptNet +8% |
| Fair 1+1 comparison H=512 | OptNet 2060 vs INSTNCT 2596 = **INSTNCT +26%** |
| Fair 1+1 comparison H=1024 | OptNet 1091 vs INSTNCT 1795 = **INSTNCT +65%** |
| Fair 1+1 comparison H=2048 | OptNet 574 vs INSTNCT 1322 = **INSTNCT +130%** |
| Packed params spike speedup | -8 to -10% at H=1024-4096 |
| Packed state (charge+activation) | -4% at H=256, +4% at H=2048 (mixed) |

**Promotions / Rejections**

- Promoted: packed NeuronParams struct (threshold+channel+polarity = 4 bytes). Interference reduction via low edge cap (100-300). The 20% accuracy ceiling as confirmed 1+1 ES search limit.
- **Corrected**: "ListNet 6x faster" downgraded to "ListNet simpler but not faster at H≥512 in fair comparison." The INSTNCT library's CSR + skip-inactive is the speed leader.
- Retained: ListNet is valid as a simpler alternative at H=256 or for code clarity. Edge cap and interference findings remain valid (independent of topology storage).
- Rejected: all-in-one rows (ListNet2, params + topology in one Vec) — 35% slower due to AoS penalty on spike sweep. FireNet gather approach. FlatNet fixed arrays.

---

### 2026-04-08 — Overnight ListNet validation (5 sweeps, Steam Deck)

**Status:** Confirmed

**What changed**

- Five overnight sweeps validated ListNet (sorted `Vec<Vec<u16>>` topology) as a production-worthy replacement for INSTNCT's HashSet+CSR. Head-to-head at H=256: ListNet 3847 step/s vs INSTNCT 564 step/s (**6.8x**), accuracy noise-equivalent (20.4% vs 20.6%). At H=2048: **2.5x** speedup, identical accuracy.
- Edge cap sweep at H=1024 showed cap=100 (20.7% mean) > cap=300 (20.6%) > cap=1000 (19.8%). Fewer edges = less interference = better mean accuracy. Supports the interference reduction thesis.
- Long run at H=512 (300s/seed, 5 seeds): best=20.8%, mean=20.0%, spread=1.4pp, **2.0 µs/token**. The 20% band is identical to 120s runs — the ceiling is search-regime-bound, not time-bound.
- Cache A/B/C/D sweep showed smooth O(H) scaling with no discontinuities at L1/L2 boundaries. INSTNCT showed cache cliffs; ListNet does not, due to simpler memory layout.

**Why it mattered**

- This closes the topology representation question with strong evidence: sorted `Vec<Vec<u16>>` is the recommended storage for sparse networks at E<=300. No HashSet, no CSR, no parallel Vecs needed.
- The interference finding (cap=100 > cap=1000) directly supports the "sparse + many neurons = searchable space" thesis from the earlier session.
- The 20% ceiling confirmation redirects effort: further speedup of the representation layer is pointless. The bottleneck is the 1+1 ES search regime. Jackpot selection (1+9), better fitness functions, or multi-population strategies are the only paths to break the band.

**Evidence**

| Metric | Result |
|---|---|
| ListNet vs INSTNCT speedup | 2.5x (H=2048) to 6.8x (H=256) |
| Best overnight accuracy | **23.8%** (H=512, seed 4042) |
| Mean accuracy band | 19.7-21.1% across all configs |
| Edge cap winner | cap=100 (20.7% mean) |
| Long run ceiling | 20.0% mean at 300s = same as 120s |
| Propagation speed | **2.0 µs/token** (H=512 ListNet) |
| Cache scaling | Smooth O(H), no cliff |

**Promotions / Rejections**

- Promoted: ListNet as recommended topology representation. Edge cap=100-300 as optimal range. H=512 as recommended Steam Deck local config.
- Confirmed: 20% accuracy ceiling is a search-regime property (1+1 ES), not representation or capacity.
- Rejected: precompiled/CSR approach adds complexity without measurable benefit over `Vec<Vec<u16>>` at E<=300. FlatNet fixed arrays waste memory. FireNet gather approach slower than ListNet scatter due to activation clone overhead.

---

---

### 2026-04-07 — Performance deep dive, topology representation experiments, and branch cleanup

**Status:** Confirmed

**What changed**

- Merged the `claude/check-smoke-port-status` branch (22 commits) into `main`: compact neuron types (i8/u8, -30% at H=4096), skip-inactive spike (-49%), fully sparse tick O(active), sparse input API O(k) (-62-72% at H=4K-100K), and copy-on-write evolution snapshots. Prefetch, nibble packing, and bitset dirty_member were correctly rejected with benchmarked evidence.
- Completed first local benchmarking session on Steam Deck (AMD Van Gogh APU, L1=32KB, L2=512KB, L3=4MB, 16GB RAM). Measured propagation and evolution step speed across H=128-8192 with 30-200 edges.
- Ran fixed-wall-clock (60s/seed, 3 seeds) accuracy comparison at H=256-4096 with empty init, edge_cap=300, and 1+9 jackpot on 100KB Alice in Wonderland corpus. H=2048 won at 21.4% best / 20.1% mean with only 6K steps — confirming that larger sparse networks find better circuits per step (interference reduction thesis).
- Designed and tested four alternative topology representations against INSTNCT baseline:
  - **ListNet** (sorted `Vec<Vec<u16>>`, fan-out per source): **3917 step/s** — 6x faster than INSTNCT (654), identical accuracy (~20%).
  - **VarNet** (fixed fan-in=3 per node): 3721 step/s, but accuracy degrades at H>1024 due to fixed edge count scaling.
  - **FireNet** (fan-in gather, no scatter): 3276 step/s — slower than ListNet due to per-tick activation clone.
  - **FlatNet** (fixed `[u16; 16]` array per neuron): 3740 step/s — comparable to ListNet but wastes memory on empty slots.
- ListNet's speed advantage comes from eliminating HashSet (O(1) lookup not needed at E<=300), CSR rebuild, and parallel Vec bookkeeping. A simple sorted `Vec<Vec<u16>>` with linear scan is sufficient and cache-friendly.
- Deleted all merged, stale, and superseded remote and local branches. Repository now has a single `main` branch.

**Why it mattered**

- The topology representation is a pure storage/access optimization — the spiking dynamics, mutation schedule, and try-keep-revert loop are unchanged. The 6x speedup is free performance at identical accuracy.
- The interference reduction thesis was directly supported: H=2048 (44KB WSS, L2) beat H=256 (6KB, L1) and H=4096 (89KB, L2) in fixed-time accuracy. More neurons + sparse edges = less interference per mutation = more valuable steps, even though fewer steps fit in the same time.
- Branch cleanup removed 18 stale branches, leaving a clean single-branch repository.

**Evidence**

| Metric | Result |
|---|---|
| ListNet H=256 step/s | 3917 (vs INSTNCT 654 = **6.0x**) |
| ListNet H=512 step/s | 2126 (vs INSTNCT 360 = **5.9x**) |
| ListNet H=1024 step/s | 1104 (vs INSTNCT 192 = **5.7x**) |
| Accuracy (60s, best seed) | ListNet 20.8%, INSTNCT 20.4% (noise-equivalent) |
| H=2048 fixed-time accuracy | 21.4% best / 20.1% mean (winner at 98 step/s) |
| Smoke branch optimizations | compact types -30%, skip-inactive -49%, sparse input -62-72% |
| Branches deleted | 18 remote + 10 local |

**Promotions / Rejections**

- Promoted: smoke-port performance branch merged to main (compact types, skip-inactive, sparse tick, sparse input, CoW snapshots). ListNet topology representation validated as 6x faster drop-in replacement.
- Rejected: FireNet gather approach (slower than scatter due to activation clone), FlatNet fixed arrays (no advantage over dynamic Vec), prefetch/nibble/bitset micro-optimizations.
- Experimental: ListNet integration into `instnct-core` library as replacement for current `ConnectionGraph`.

---

### 2026-04-06 — Hyperparameter exhaustion, topology falsification, and library hardening

**Status:** Confirmed

**What changed**

- 11 independent tuning and strategy axes were pushed through overnight sweeps, including edge cap, chain init, polarity, topology constraints, butterfly scaling, breed v1/v2, simulated annealing, ratchet, projection dimension, ensemble oracle, and SDR rate. None lifted the stable 17-18% next-char band, even though several sweeps still produced transient 19.1% peaks.
- A corrected oracle check clarified the stronger claim: top-4 oracle = 17.2% vs best single = 17.0% on the good-seed ensemble, even though the networks shared only about 4% Jaccard edge overlap. Within the tested 1+1 ES setup, different topologies were still converging onto near-identical predictions.
- Pocket-chain exploration split cleanly: a 2-pocket network (H=452) reached 17.5% peak inside one spatially constrained model, while post-hoc chaining of separately trained pockets failed at 6.1% because the I/O modalities did not compose.
- A separate two-pocket charge-transfer line pushed the depth test further: two independent H=256 pockets reached **19.6% peak**, but only at parity with the familiar band. Male-overlay merge plus continuation peaked at 17.45% and 18.85%, then drifted back down instead of stabilizing into a stronger recipe.
- The shared-female critical test refuted the strongest merge hypothesis: even with one frozen upstream Female, the Males still shared only `2.5-3.6%` Jaccard overlap while reaching `88-98.5%` prediction agreement. Merge collapsed to `5.6%`, which means topology diversity was not creating complementary information under this regime.
- Watts-Strogatz small-world init also failed to create a new regime. It matched the same `19.1%` peak as chain-50, and its higher means only arrived with `1.5-3x` more edges. Chain-50 therefore stayed the more edge-efficient public default.
- `EvolutionConfig` was refactored so edge cap (hard topology limit) and quality gating (`accept_ties`) became independent. The default moved to strict no-tie acceptance after a 6-seed sweep showed a +4.2pp mean advantage.
- Checkpoint persistence landed as an atomic bundle for Network + Projection + metadata, with 7 adversarial round-trip and functional-match tests passing.

**Why it mattered**

- This substantially closes the single-network tuning line: the remaining problem no longer looks like edge cap, polarity, annealing, simple breed overlays, or small-world init tweaks. It looks increasingly like a limit of the current 1+1 ES exploration and evaluation regime.
- It also narrowed the depth story instead of promoting it. Pocket pair stayed alive as evidence that charge-chained pockets can run at parity, but the shared-female and merge results cut against the idea that pocket breeding is the immediate breakout path.
- Library hardening mattered because longer and more adversarial runs now need strict acceptance logic, resumable state, and atomic checkpointing instead of one-shot benchmark scripts.

**Evidence**

| Metric | Result |
|---|---|
| Tuning and strategy axes tested | 11 |
| Good-seed oracle vs best single | 17.2% vs 17.0% |
| Shared-female male Jaccard / agreement | 2.5-3.6% / 88-98.5% |
| Pocket pair | 19.6% peak / 18.1% best final |
| WS init vs chain-50 | Same 19.1% peak; higher mean only with 1.5-3x more edges |
| Checkpoint round-trip tests | 7/7 passing |

<details>
<summary>Carry-over findings</summary>

| Sweep family | Result |
|---|---|
| Edge cap | no ceiling break; `19.1%` peak appeared at all tested caps |
| Chain + polarity ablation | polarity mutation was catastrophic on mean quality and caused edge bloat |
| Breed v1 / v2 + male overlay merge | no breakthrough; merged descendants peaked below the best individual and stayed unstable |
| Shared female | refuted the interface-convergence hypothesis; high prediction agreement still came with ~`3%` topology overlap |
| Watts-Strogatz init | no ceiling break; the mean lift came from denser graphs, not a new structural regime |
| Pocket pair | parity-level depth result; useful evidence, but not a promoted breakout line |

</details>

**Promotions / Rejections**

- Promoted: strict acceptance (no ties) as default, checkpoint persistence, and the reading that the stable Rust band now looks like a property of the tested exploration/evaluation regime rather than one missing scalar fix.
- Rejected or downgraded: polarity mutation on the main library loop, post-hoc pocket chaining, pocket-level breeding as the immediate breakout claim, shared-interface convergence as a merge strategy, and Watts-Strogatz init as a ceiling-breaking fix.

---

### 2026-04-05 — Rust v5 beta hardening, seed variance mapping, and archive discipline

**Status:** Confirmed

**What changed**

- The Rust port crossed from “fast forward pass” into “real evolution substrate”: `NetworkSnapshot`, full 10-op mutation API, CSR skip-inactive, genome save/load, refractory support, and rayon-backed multi-seed evolution all landed in the active beta line.
- Language-eval work exposed seed variance as a first-class problem rather than anecdotal noise.
- A separate GPT quick map of **base seeds `1..100`** showed a rugged landscape with weak and strong bands, not a simple seed formula.

**Why it mattered**

- Multi-seed search became practical enough to treat seed quality as a search problem instead of luck.
- Evidence handling also changed: seed maps, charts, and log extracts now deserve stable archive leaves instead of only living inside transient benchmark prose.

**Evidence**

| Metric | Result |
|---|---|
| Rayon 12-seed run | `1m57s` on 12 cores |
| Best reported Rust next-char result | **18.5%** |
| GPT quick base-seed map mean | **9.1%** |
| Quick map best / worst | **17.5%** (`seed 17`) / **0.3%** (`seed 24`) |

> The important update is not that one seed won; it is that the landscape is visibly rugged enough to justify stable rerank, archive-grade tracking, and multi-seed workflow discipline.

![GPT Seed Sweep Check 2](seed-sweep-check-2.png)

Short excerpt from the seed map:

```text
baseline = 16.9%
mean     = 9.1%
best     = 17.5% @ seed 17
worst    = 0.3%  @ seed 24
spread   = 17.2pp
```

<details>
<summary>Open full quick seed table (1..100)</summary>

| Seed | Score | Seed | Score | Seed | Score | Seed | Score | Seed | Score |
|---|---:|---|---:|---|---:|---|---:|---|---:|
| 1 | 16.4% | 21 | 7.1% | 41 | 6.3% | 61 | 16.9% | 81 | 16.5% |
| 2 | 10.7% | 22 | 14.1% | 42 | 9.7% | 62 | 12.1% | 82 | 1.0% |
| 3 | 9.1% | 23 | 7.1% | 43 | 4.5% | 63 | 6.7% | 83 | 7.7% |
| 4 | 4.1% | 24 | 0.3% | 44 | 3.7% | 64 | 14.8% | 84 | 16.9% |
| 5 | 7.2% | 25 | 3.8% | 45 | 4.5% | 65 | 2.0% | 85 | 8.6% |
| 6 | 4.5% | 26 | 9.1% | 46 | 7.7% | 66 | 3.8% | 86 | 7.5% |
| 7 | 1.1% | 27 | 12.9% | 47 | 3.4% | 67 | 6.7% | 87 | 6.9% |
| 8 | 16.9% | 28 | 7.1% | 48 | 4.6% | 68 | 16.9% | 88 | 16.9% |
| 9 | 6.6% | 29 | 17.0% | 49 | 5.7% | 69 | 16.3% | 89 | 16.9% |
| 10 | 11.8% | 30 | 16.9% | 50 | 7.2% | 70 | 7.2% | 90 | 13.4% |
| 11 | 6.6% | 31 | 3.4% | 51 | 5.7% | 71 | 16.9% | 91 | 16.1% |
| 12 | 15.2% | 32 | 7.6% | 52 | 3.0% | 72 | 6.3% | 92 | 5.3% |
| 13 | 9.3% | 33 | 14.6% | 53 | 0.8% | 73 | 6.1% | 93 | 16.9% |
| 14 | 16.1% | 34 | 8.5% | 54 | 5.3% | 74 | 16.9% | 94 | 7.2% |
| 15 | 6.7% | 35 | 7.2% | 55 | 7.6% | 75 | 9.1% | 95 | 7.6% |
| 16 | 7.5% | 36 | 16.4% | 56 | 7.5% | 76 | 16.9% | 96 | 6.2% |
| 17 | 17.5% | 37 | 7.7% | 57 | 7.2% | 77 | 3.2% | 97 | 1.2% |
| 18 | 3.3% | 38 | 16.9% | 58 | 2.8% | 78 | 10.4% | 98 | 16.8% |
| 19 | 4.5% | 39 | 1.5% | 59 | 10.1% | 79 | 16.9% | 99 | 5.8% |
| 20 | 15.3% | 40 | 4.5% | 60 | 7.2% | 80 | 5.9% | 100 | 14.8% |

</details>

**Promotions / Rejections**

- Promoted into the active Rust beta line: genome persistence, rayon multi-seed evolution, and archive-grade seed evidence.
- Not promoted: any claim that there is already a simple “good seed” formula.

---

### 2026-04-02 to 2026-04-03 — Deterministic benchmarking and owned Rust `Network`

**Status:** Confirmed

**What changed**

- `v5.0.0-beta` became a clean Rust-port branch with an owned `Network` abstraction, checked propagation, and topology cleanup.
- Deterministic benchmark discipline replaced looser micro-optimization claims: core pinning, repeated runs, noise-floor control, and same-logic comparisons became the standard.
- Several tempting optimizations were re-tested and downgraded from “fast” to “rejected or inconclusive”.

**Why it mattered**

- This was the shift from “bench anecdotes” to reproducible performance claims.
- It also turned `instnct-core` into a library surface that could safely absorb snapshots, mutation API, persistence, and later multi-seed search.

**Evidence**

- `Network` landed as an owned topology + params + state object with deterministic hand-calculated tests.
- H=1024 throughput improved after redundant edge storage was removed.
- AVX2, edge sorting, and PGO claims were all re-checked under deterministic harness rules instead of being trusted from noisy runs.

**Promotions / Rejections**

- Promoted: deterministic benchmark policy, topology cleanup, owned `Network`.
- Rejected or downgraded: early overstated micro-optimization claims that did not survive controlled reruns.

---

### 2026-03-29 — Compact parameter stack, learned channel, and breed breakthrough

**Status:** Confirmed

**What changed**

- Wave/freq/phase learning was overturned: frozen random temporal diversity beat learnable freq/phase, and the line was compressed into a compact per-neuron `channel`.
- The compact parameter stack became much clearer: mask + polarity + theta + channel carry the learnable burden; decay and rho moved toward fixed/baked roles.
- `breed + crystallize` hit **24.4%**, becoming the first reported result to beat the best fixed schedule line of that phase.

**Why it mattered**

- This was the point where the architecture stopped paying rent for several expensive floating control surfaces and moved toward a smaller, more evolvable parameterization.
- It also showed that consensus structure plus pruning could beat pure single-line mutation search.

**Evidence**

| Signal | Result |
|---|---|
| Learnable channel | **23.8%** |
| Breed + crystallize | **24.4%** |
| Reverse-heavy schedule | `20.8%` and strong structural role |

<details>
<summary>Carry-over findings</summary>

| Finding | Signal |
|---|---|
| Theta int4 | `15.6%` vs float `13.5%`; compact discrete theta beat float search |
| Theta Pareto | 1-bit `10.1%`, 2-bit `12.7%`, 4-bit `15.6%`; float32 was dominated |
| Rho fix | fixed `0.3` hit `14.5%`, close to int4 `15.2%` and ahead of float `14.1%` |
| Decay fix | fixed `0.16` held `19.4%` vs learnable `20.8%`, showing small fixed tradeoffs could buy huge simplicity |
| Binary wave | `21.4%`; 2-bit wave types outperformed heavier temporal parameterizations |
| Learnable channel | `23.8%`; one compact `channel` byte beat the older freq/phase stack |
| Reverse-heavy schedule | `20.8%`; `enhance/reverse/mirror` made topology shaping visibly stronger |
| Breed + crystallize | `24.4%`; consensus structure plus pruning beat the best single-line fixed schedule |
| Big ReLU controller | `23.6%`; learned phase transitions got close to the fixed schedule line but did not clearly replace it |

</details>

<details>
<summary>Carry-over doctrine</summary>

- `Tree-Wired Scaffold Init` reached `15.6%` versus random 5% prefill at `22.4%`; structured beauty kept losing to evolvable chaos.
- `Learnable Schedule (empty start)` reached `14.9%` with only `89` edges; quality-per-edge was impressive, but the line still trailed the stronger prefill schedule family.
- The `Navigable infinity principle` hardened here: compact discrete or fixed controls consistently beat wider continuous alternatives on the core sweeps of that phase.

</details>

**Promotions / Rejections**

- Promoted: compact channel-centric temporal specialization as a serious architecture line.
- Rejected: learnable freq/phase as a practical public default.

---

### 2026-03-28 — Phi overlap and richer output readout become the public center

**Status:** Confirmed

**What changed**

- Output-dimension sweeps, phi-style proportioning, and overlap experiments converged into the strongest public architecture story of that week.
- `output_dim=160` and then phi overlap pushed the architecture to new reported peaks.
- The public framing shifted from “more hidden is always better” to “the overlap/readout geometry matters more”.

**Why it mattered**

- It clarified that the network’s representational bottleneck sat at the I/O and overlap geometry, not just raw neuron count.
- It also reinforced the asymmetry that sparse input + smooth output is the winning public pattern.

**Evidence**

- Output dim sweep found **20.0%** at `out_dim=160`.
- Phi overlap reached **20.8%**.
- Multi-seed confirmation and scale sweep made the result harder to dismiss as a one-off.

<details>
<summary>Carry-over findings</summary>

| Finding | Signal |
|---|---|
| Output-dim sweep | `out_dim=160` peaked at `20.0%`; multi-seed mean `18.2%` with `0.6%` std |
| Scale sweep | `0.625` output ratio held across H=`128/192/256/384`, keeping the phi-style downshift alive beyond one size |
| Phi overlap | `20.8%` at H=256 with in=out=158 and overlap=60; zero dedicated hidden neurons were needed |
| Language-aware output projection | `FREQ_ORDER=22.4%` beat `BIGRAM_SVD=21.8%` and random `20.8%`; output topology had to respect target distribution |
| Full overlap | `14.7%` vs phi overlap `20.8%`; too much overlap added noise instead of signal |
| Direct 256 output | `7.1%`; slow, too few accepts, and worse than projection-based readout |
| Later input confirmation | SDR phi overlap stayed best at `22.4%`, ahead of FREQ `12.3%` and one-hot `9.5%` |

</details>

**Promotions / Rejections**

- Promoted: phi overlap as the key active public architecture finding of that moment.
- Rejected: direct 256 output and repeated binary-output simplifications as practical defaults.

---

### 2026-03-27 — I/O architecture overhaul makes sparse-in / dense-out the live line

**Status:** Confirmed

**What changed**

- Tentacle I/O beat holographic projection.
- SDR input beat the other tested input encodings and was baked into the Python stack.
- Charge readout beat state readout, and learnable theta opened the next quality jump.

**Why it mattered**

- This was the public evidence pivot that made the present I/O story legible.
- It also killed several cleaner-looking but weaker representations before they could pollute the active architecture line.

**Evidence**

| Result | Outcome |
|---|---|
| Tentacle I/O vs holographic | `4.7%` vs `1.2%` |
| SDR input | **7.3%** |
| Learnable theta | **14.1%** |
| Charge readout vs state | `14.1%` vs `10.3%` |

<details>
<summary>Carry-over findings</summary>

| Finding | Signal |
|---|---|
| Tentacle I/O vs holographic | `4.7%` vs `1.2%`; random 5% prefill + BFS tentacles beat both holographic projection and structured resonator init |
| 8-bit binary I/O baseline | `0.2%` vs `4.4%` for 64-dim projection; compact binary I/O starved the architecture of signal richness |
| SDR input | `7.3%` vs random `4.4%`; sparse distributed input cleaned up the routing problem enough to become the live Python default |
| Random vs SDR output | random 64-dim out `7.3%` vs SDR out `3.4%`; sparse-in / dense-out became the winning asymmetry |
| Learnable theta | `14.1%` vs fixed theta `7.3%`; low-start full-resample theta clearly beat fixed thresholds |
| Charge readout | `14.1%` vs state `10.3%`; charge stayed the richer readout surface even though spikes remained load-bearing internally |
| 8-bit I/O v2 | 8-bit in + random 64 out reached `9.1%`, but SDR input still won cleanly and 8-bit in + 8-bit out stayed dead at `0.0%` |

</details>

**Promotions / Rejections**

- Promoted: SDR input and asymmetric sparse-in / dense-out architecture.
- Rejected: 8-bit binary I/O as a public path.

---

### 2026-03-22 to 2026-03-26 — English line hardens around schedules, controls, and edge format

**Status:** Archived

**What changed**

- The English line settled around a conservative public recipe while multiple side branches competed on schedules, mutation rules, decay handling, and edge representation.
- Voltage medium leak, decision-tree control, and sign+mag magnitude resample became the strongest historical schedule/control/edge-quality results of that phase.
- Several side probes clarified what not to over-promote: potential-aware fitness, controller-heavy alternatives, and codepath variants could score locally without displacing the simpler public line.

**Why it mattered**

- This was the last phase where a separate findings hub still made sense; the surviving signal now lives here as chronology instead of as a standalone page.
- It explains why the public English line stayed relatively conservative even while stronger-looking local variants kept appearing around it.

<details>
<summary>Carry-over findings</summary>

| Finding | Signal |
|---|---|
| Flip mutation | `+1.89%` over float weight perturbation on English 1024n |
| Low theta + `INJ_SCALE=1.0` | `12.91%` vs `11.01%`; smaller scale and lower theta beat the harsher baseline |
| 8 ticks + current English candidate schedule | `8` ticks beat `6`, and the line settled on `2 add / 1 flip / 5 decay` for the public English recipe candidate |
| Decay resample mutation | full single-neuron resample in `[0.01, 0.5]` beat local perturbation and produced differentiated decay bands |
| Voltage medium leak schedule | strongest historical schedule result at `22.11%` peak / `21.46%` plateau |
| Decision-tree schedule | `20.05%` at `156` edges, the strongest compact learnable control policy of that phase |
| Sign+mag + magnitude resample | `18.69%` at `155` edges (`q=0.121`), the best quality-per-edge result in the edge-format sweep |

</details>

<details>
<summary>Carry-over doctrine</summary>

| Finding | Signal |
|---|---|
| Potential-aware fitness | standard `14.1%` beat both weighted variants (`11.3%`, `8.3%`); false positives through the projection surface made the idea too brittle |
| Claude vs Gemini `graph.py` A/B | Claude `14.1%` beat Gemini `11.3%`; C19 clip and batch refractory remained load-bearing |
| Control neurons / binary toggles | `15.8%` / `14.9%`; controller-heavy meta-learning still lost to the simpler fixed schedule line |
| C edge port | `95K` tokens/sec vs Python `3.2K` — a `29.4x` speedup, but as an edge-side branch rather than the public default |

</details>

## Older Timeline

### Early 2026 — Diamond Code Era

**Status:** Archived

**What changed**

- Diamond Code / LCX dominated the public architecture story before INSTNCT became the active center.
- External memory, dreaming, observability, and Goldilocks-style probes were still the visible front line.
- The surviving signal from that phase now lives here as chronology, not as a current architecture recommendation.

**Why it mattered**

- This is the historical substrate that the later INSTNCT line replaced rather than a current public recommendation.

### Early Feb 2026 — Research intake wave around swarm scaling, memory stability, and dimensionality

**Status:** Archived

**What changed**

- Early intake pages gathered swarm scaling postmortems, memory stability research, and biological dimensionality arguments while the public architecture line was still unsettled.
- The swarm/bus postmortem had already started collapsing “more beings / wider bus” into later bounded-bandwidth and stable-semantics doctrine.
- VRA-43 and dimensionality notes kept the theory pressure alive even before the architecture surface was clean enough to promote durable claims.
- A first explicit "resolution over reshuffling" doctrine also formed here: keep the global interface fixed-width, prefer local checkpoint-time refinement over global rewiring, preserve stable address semantics, and recover state from explicit pointers before broad rescans.

**Why it mattered**

- This was an intake layer underneath later doctrine, not doctrine itself.
- It explains why a small number of research pages still remain as hidden archive despite no longer being part of the canon stack.
- It was also the clearest early statement that scaling should mean bounded-bandwidth resolution plus deterministic recovery, not just wider buses or repo-wide reshuffling.

### 2026-02-08 to 2026-02-10 — Swarm sprint, AGC diagnostics, and pre-INSTNCT consolidation

**Status:** Archived

**What changed**

- A 34-session sprint established the AGC diagnostic campaign, pushed the architecture away from older GRU-style framing toward `think_proj`, and made dashboard-style observability mandatory.
- Swarm topology, ant-ratio exploration, GPU scaling, and visualization concepts were still first-class workstreams.
- VRA-78 frontier scans explicitly separated GPU "biomass" pressure from expert-count "brain mass", ranking swarm-era expert layouts against device-fit limits rather than treating head count as a free scaling knob.
- Early legacy probes also established that depth plus embedding capacity, not dual-pointer ornamentation, unlocked addition and small multi-task learning, while simplified phase embeddings outperformed mathematically cleaner Mobius-style variants.
- Platinum-code-era consolidation reached the point where several older concepts were already being triaged into keep / discard / migrate buckets.

**Why it mattered**

- This was the last dense burst before the wiki and architecture surfaces started splitting proof, doctrine, and raw sprint logs into cleaner layers.
- It also marks one of the earliest explicit capacity-budget framings later inherited by cleaner architecture budgeting, even though the swarm-era artifact pages themselves were retired.
- It was also an early lesson that learnability beat elegance: the project kept whichever inductive bias actually trained, even when the mathematically prettier formulation lost.

<details>
<summary>Carry-over findings</summary>

| Legacy probe | Historical takeaway |
|---|---|
| Addition ablation | depth + embedding capacity mattered more than dual-pointer ornamentation |
| Minimal multi-task model | a compact 128D / 2-layer line was enough to solve small arithmetic/logical tasks |
| Mobius vs simplified phase | simplified phase embeddings beat the more mathematically "correct" Mobius variant |

</details>

### 2026-02-14 to 2026-02-18 — Diamond Code v3 consolidation and bottleneck redesign

**Status:** Archived

**What changed**

- VRAM leak bugs, detach mistakes, and observability gaps were identified and fixed during the Diamond Code v3 consolidation sprint.
- Score-margin telemetry, LCX bottleneck design, and one-level LCX architecture decisions were hardened during this period.
- Goldilocks Nano / Goldilocks Ant, binary-bits encoding, and the first major parameter-efficiency push became the dominant direction by 2026-02-18.
- A larger probe wave locked the LCX read path into a two-layer C19 squeeze around a 10:1 bottleneck at D=6180, with read-only LCX and non-residual/no-norm variants beating more ornamental alternatives.
- Binary-bits encoding beat byte-token baselines on parameter efficiency, and a pre-beta 4096D harvest showed a sharp late crystallization phase plus one-level LCX dominance.
- Theme-based dialogue training switched from opaque `.traindat` streams to explicit JSONL input/output pairs, with curriculum/gist mixing and alternating input/output positions as the training format.

**Why it mattered**

- This is the bridge between the older LCX-heavy story and the later insistence on compact, testable, architecture-facing claims.
- Several later doctrine pages inherited their evidence discipline from this consolidation window even though the architecture itself was later superseded.
- It also marked an early move toward more inspectable data contracts and clearer training semantics, even though the whole theme-training surface stayed in the pre-INSTNCT lineage.
- It converted a pile of aesthetic defaults into measured rules and helped kill the multi-level LCX story in favor of one-level-plus-grow.

<details>
<summary>Carry-over findings</summary>

| Finding | Historical takeaway |
|---|---|
| 2x618 C19 bottleneck | the learned squeeze, not raw width, carried LCX integration |
| Binary-bits vs byte-token | binary-bits was dramatically more parameter-efficient at edge scale |
| Pre-beta 4096D harvest | phase transition near step 945 and only L0 active, reinforcing one-level-first design |

</details>

### 2026-02-19 — Probe ladder and cold-brain sweep burst

**Status:** Archived

**What changed**

- A dense probe burst tested cold-brain activation, depth, learning rate, attention temperature, state EMA, jump tau, LCX tau, top-k, and undetach behavior on small deterministic harnesses.
- The cold-brain ladder mostly closed obvious low-cost hyperparameter questions instead of revealing a single dominating fix.
- A dedicated LCX bootstrap probe showed that LCX slightly hurts cold training on tasks the brain can already solve alone, while still bootstrapping routing behavior in parallel.
- Slot count made no meaningful difference to bootstrap speed or accuracy in the cold regime, so the real issue was when to enable LCX, not how many slots to expose.
- A focused top-k sweep then showed that retrieval count barely moved accuracy at small slot counts, but at 1000 slots `k=2` matched or beat higher values while running much faster.
- The output of this day became a pile of narrow probe leaves rather than durable doctrine.

**Why it mattered**

- This is the clearest example of why raw probes no longer belong on the primary canon surface.
- The useful signal was mostly eliminative: which knobs were flat, which fixes were overclaimed, and which bottlenecks were probably elsewhere.
- It was also one of the clearest arguments for progressive training: let the brain learn first, then switch LCX on after the base path is stable.
- It explained why several earlier CPU probes looked inconclusive: the harness was testing LCX before the model had a chance to bootstrap cleanly.
- It also killed one of the more aesthetic but weak defaults: phi-derived `top_k=6` lost to the simpler `k=2` operating point once speed and robustness were measured together.

<details>
<summary>Mini table: LCX bootstrap</summary>

| Config | mean_tail | Reading |
|---|---|---|
| `tt0_noLCX` | **1.0000** | perfect on the trivial baseline task |
| `tt1_50slots` | 0.9935 | slight LCX drag, routing still bootstraps |
| `tt1_500slots` | 0.9932 | same story; slot count did not matter |

</details>

<details>
<summary>Mini table: LCX top-k</summary>

| k | mean_tail | seed_gap | s/step | Reading |
|---|---|---|---|---|
| 1 | 0.811 | 0.007 | 0.47s | faster, but noisier and borderline weaker |
| **2** | **0.816** | **0.003** | **0.61s** | locked floor |
| 4 | 0.815 | 0.001 | 0.95s | similar quality, slower |
| 6 | 0.813 | 0.002 | 1.29s | slower baseline, no quality gain |

- Final production reading: `k=2` kept the same quality band while cutting LCX retrieval cost by roughly 2.1x versus `k=6`.

</details>

### 2026-02-20 to 2026-02-22 — LCX bottleneck, contamination, and plateau diagnosis

**Status:** Archived

**What changed**

- BN width, D ablation, depth, bootstrap, LCX write-strategy, catastrophic-forgetting, and plateau probes converged on a harsher picture of memory-side behavior.
- LCX whiteboard underperformance was traced to cross-batch contamination and weak write-selection signal rather than a clean universal scaling win.
- A dedicated bottleneck-width series then showed that the D=2048 choke was a pipe-width problem rather than a brain-width problem: widening the BN pipe let D=2048 match D=4096-class results at far lower parameter cost.
- The same series showed that the learned squeeze transform was essential: no-BN and single-layer full-width variants underperformed, while a 2:1 BN:bits ratio emerged as the production sweet spot.
- A stressed D-ablation then showed that 8-bit I/O had been hiding the real dimension question: under 200-bit pressure D=2048 clearly choked, while D=4096, D=6180, and D=8192 all escaped the bottleneck.
- Above that choke point, larger D mostly changed learning trajectory rather than final quality: D=6180 led earlier, while D=4096 and D=8192 accelerated later and finished in the same narrow tail band.
- A full-scale depth probe showed that shallower stacks beat deeper ones across both easy and hard tasks, locking `depth=2` and killing the older golden-ratio depth aesthetic.
- The depth result sharpened the interpretation that LCX carried the real integration burden while the processing stack mostly acted as translation/plumbing.
- Write-strategy probes showed that the whiteboard mechanism could help when reads stayed relevant, but random-batch training turned it into a contamination channel rather than a reliable memory boost.
- Evolutionary selection and raw-overwrite variants failed to rescue the effect, shifting attention from "how do we write?" to "how do we preserve read relevance across batches?"
- The curriculum-transition failure was eventually traced away from simple replay lore toward a three-factor optimization stack: too-low LR, LCX noise from random init, and the extra overhead of the full SwarmByteRingModel.
- The early offset-selectivity theory was investigated hard, but later probes showed echo256 was learnable without explicit position encoding; the real failure mode was optimization and interference, not a hard architectural impossibility.
- The first real GPU English-text training run on FineWeb-Edu still showed the brain could learn, but LCX-on variants and eval-time sweeps exposed how fragile the story was.

**Why it mattered**

- This window explains the shift from “LCX as obvious scaling win” to a much more adversarial attitude toward memory-side claims.
- It also marks the last major English-training result in the pre-INSTNCT architecture family before the public center moved elsewhere.
- It was the cleanest evidence that LCX read quality depended on the translation pipe, not just on making the whole model bigger.
- It also exposed a future scaling trap: the old fixed `D/10` rule would eventually collapse to a 1:1 pipe at higher bit widths, so bottleneck sizing had to become capacity-aware rather than aesthetic.
- It turned the D question from "which size wins?" into a more precise diagnosis: once the bottleneck ratio cleared the choke zone, wider brains alone stopped being the main limiter.
- It also preserved the historical reason D=6180 stayed in the public line: not because it crushed D=4096 empirically, but because it sat in the safe band while keeping the golden-ratio framing and reasonable VRAM.
- It turned a vague architecture hunch into a concrete resource win: fewer params, less VRAM, and slightly better quality by making the stack shallower instead of deeper.
- It also gave the clearest mechanistic explanation for why LCX-on variants kept underdelivering in practice: the issue was read-side relevance across batches, not just the write formula.
- That diagnosis is what made later sleep / double-buffer style consolidation ideas feel like engineering follow-through instead of pure metaphor.
- It also prevented the project from overfitting to the wrong story: the event looked like pure catastrophic forgetting, but the actual stack mixed negative transfer, slow learning, and LCX-induced noise.
- That revision tightened later curriculum doctrine: test learnability and optimization first, then talk about replay, distillation, or anti-forgetting machinery.

<details>
<summary>Mini table: processing depth</summary>

| Depth | Params | VRAM | Final acc | Reading |
|---|---|---|---|---|
| 1 | 12M | 1.3G | 53.6% | converged fast but chosen as too minimal |
| **2** | **50M** | **2.0G** | **53.2%** | locked compromise |
| 6 | 203M | 4.6G | 52.1% | slower, larger, and worse |

- Reported savings versus `depth=6`: roughly 153M fewer params, 2.6G less VRAM, and about 23% faster throughput.

</details>

<details>
<summary>Mini table: D ablation</summary>

| D | Tail acc | VRAM | Reading |
|---|---|---|---|
| 2048 | 50.39% | 0.5 GB | choked at 1.0:1 |
| 4096 | 50.63% | 1.1 GB | late accelerator |
| **6180** | **50.60%** | **2.0 GB** | early leader, later plateau |
| 8192 | 50.67% | 3.1 GB | late accelerator, no clear practical win |

- Key takeaway: `D >= 4096` beat `D=2048` clearly, but `D=4096 / 6180 / 8192` stayed statistically close at the tail.
- This kept `D=6180` as a historical operating point inside the safe band rather than a decisive empirical knockout.

</details>

<details>
<summary>Historical evidence: catastrophic-forgetting root-cause stack</summary>

| Factor | Impact | Reading |
|---|---|---|
| LR too low | -19.9% | strong contributor, but not fatal alone |
| LCX noise from random init | -12.8% | major interference source |
| Architecture overhead | -6.4% | real but smaller tax |
| **Total explained** | **~39%** | layered optimization/interference failure |

</details>

<details>
<summary>Historical evidence: offset-selectivity revision</summary>

- Early inspection made the failure look like an architectural position/offset problem.
- That hypothesis was useful as a probe generator, but later position-encoding tests showed echo256 could be learned without explicit positional machinery.
- The mini model learned the task cleanly; the full model failed mainly because optimization and LCX-side interference buried the signal.
- The final diagnosis therefore moved from "architecturally impossible" to "learnable, but easy to sabotage with bad training conditions."

</details>

<details>
<summary>Historical evidence: supporting probes</summary>

- Position encoding: no meaningful gain on the mini probe.
- LR ablation: a major contributor, but not the whole failure.
- Real-model ablation: the architecture cleared; LCX added substantial noise from random init.
- Weight decay: effectively irrelevant on this task.

</details>

<details>
<summary>Mini table: bottleneck width</summary>

| Config | Tail acc | Reading |
|---|---|---|
| no BN | 50.13% | raw LCX add was basically unusable |
| BN=204 | 50.39% | native `D/10` pipe, still choked |
| **BN=409** | **50.64%** | sweet spot, matched D=4096-class result |
| BN=618 | 50.63% | no real gain beyond 2:1 |

- Key takeaway: `D=2048 + BN=409` reached `D=4096`-level quality with far fewer params, proving the pipe was the limiter.

</details>

<details>
<summary>Mini table: LCX write strategy</summary>

| Probe | Core result |
|---|---|
| Evo whiteboard | no real gain; selection behaved like a coin flip |
| Fixed batch | clear gain; cross-batch contamination confirmed |
| Snapshot tournament | whiteboard content did matter once held against fixed weights |
| Raw writes | write rule barely mattered; read relevance stayed the real problem |
| Double-buffer | emerged as the proposed fix direction |

</details>

### 2026-02-26 — Hidden/slot split and v4 precompute sprint

**Status:** Archived

**What changed**

- Training-run analysis turned up major GPU bottlenecks in the v4 line.
- On 2026-02-24, parameter sweeps locked the byte-granular baseline around `B=8`, `D=256`, and `M=256`-class ring sizing, while showing throughput was constrained by the sequential `T x N` loop rather than raw VRAM.
- Two optimizations landed: precomputed attention weights and the hidden_dim / slot_dim split.
- The split architecture dramatically reduced ring-clone VRAM pressure and reset the next training configuration around a more explicit capacity model, separating expert-brain width from ring-memory width.

**Why it mattered**

- This was the point where raw GPU systems work started feeding into cleaner architecture budgeting rather than ad hoc scaling.
- It was the first clean bridge from sweep-heavy parameter tuning into explicit capacity budgeting in the v4 line.
- The result belongs to the historical pre-INSTNCT line, but it is still part of the path that produced the later evidence discipline.
- It also clarified that several apparent "scaling" questions were really throughput and memory-layout questions rather than evidence for widening the whole system.

<details>
<summary>Historical evidence: locked v4 baseline</summary>

| Field | Locked value / decision |
|---|---|
| Input width `B` | `8` (byte-granular baseline) |
| Legacy width `D` | `256` sweep winner before hidden/slot split |
| Ring size `M` | `256` for `seq_len <= 128`; scale with longer sequences |
| Expert count `N` | `6` sweep winner, later reduced to `2` after the split for roughly 3x speed at less than 2% loss cost |
| Hidden width | `hidden_dim = 4096` |
| Ring-cell width | `slot_dim = 32` |
| Other locked defaults | `R=2`, `S=0.05`, `Je=0.9`, `Jw=0.1` |

- Ticket context: [#106](https://github.com/VRAXION/VRAXION/issues/106), [#107](https://github.com/VRAXION/VRAXION/issues/107), and [#108](https://github.com/VRAXION/VRAXION/issues/108).
- Precomputed attention weights landed alongside the split and were recorded as roughly a 40% speedup with bit-identical outputs.

</details>

<details>
<summary>Mini tables: v4 sweeps</summary>

**Input width sweep at `D=256` (GPU, echo task)**

| B | D/B ratio | best_loss | converged? |
|---|---|---|---|
| **8** | 32x | **0.924** | yes |
| 16 | 16x | 1.159 | no |
| 32 | 8x | 1.298 | no |
| 64 | 4x | 1.755 | no |
| 128 | 2x | ~2.1 | no |

**Slot-width sweep**

- `D=256` was the best loss point (`0.304`).
- `D > 256` still fit in VRAM, but degraded under the available step budget; memory was not the binding constraint.

**Expert-count sweep**

| N | best_loss | note |
|---|---|---|
| **6** | **0.2656** | best quality |
| 2 | 0.2708 | close result, roughly 3x faster |

**Ring-size sweep at `seq_len=64`**

| M | echo | delay_echo |
|---|---|---|
| 32 | 0.502 | 0.566 |
| 64 | 0.380 | 0.383 |
| 128 | 0.282 | 0.303 |
| 256 | 0.269 | 0.290 |
| 512 | 0.272 | 0.280 |

- Working rule: `M >= 2 x seq_len`.

</details>

### 2026-03-21 — Canonical Docs & Schedule Research

**Status:** Archived

**What changed**

- Repo-tracked docs became canonical and the GitHub wiki became the mirror surface.
- Schedule-control work became the main live research frontier.
- Roadmap, theory, archive, and glossary roles collapsed into a single timeline-style public surface.

**Why it mattered**

- This was the documentation governance pivot that made later consolidation possible.

### 2026-03-22 — Recipe consolidation and canon freeze

**Status:** Archived

**What changed**

- Triangle convergence distilled into the fixed English recipe: `2 add / 1 flip / 5 decay`.
- Sign+mag edge representation became the best quality-per-edge evidence line of that phase.
- Task-learning experiments started displacing the older swarm line.
- Canon boundaries tightened and archive branches were cut ahead of the public beta push.

**Why it mattered**

- This narrowed the tracked public surface and made archive discipline a first-class rule.

### 2026-03-25 — Resonator Theory

**Status:** Archived

**What changed**

- Resonator Chamber theory was formalized with FlyWire validation.
- The public theory framing locked onto destructive interference as the fixed-point mechanism.

**Why it mattered**

- This is the major theory milestone connecting the public architecture line to biological-scale evidence.

## Active Research Gates

These are the open gates that still block stronger promotion claims or a cleaner public-beta posture.

| Gate | What still must hold | What promotion would change |
|---|---|---|
| Public-beta hardening | Tighten the newcomer path, make known limitations explicit, improve public intake routing, and keep canonical / validated / experimental claims visibly separate under higher traffic. | Turn the current beta-prep lane into a cleaner public-beta surface instead of an internal hardening track. |
| Context-dependent task learning | Show that word-pair memory, framed tasks, and windowed input gains hold under reruns and stronger evaluation without collapsing back to context-free behavior. | Promote the task-learning line from active frontier to validated finding and make it the next serious architecture-update candidate. |
| Input-window promotion | Show that `w=2` superposition keeps winning across reruns and task families without unstable overflow or masking effects. | Promote a windowed injection policy from evidence into the current recipe discussion. |
| Voltage-aware schedule pressure | Show that a voltage-style schedule policy wins on plateau accuracy under confirmation reruns, not only on isolated peaks. | Promote the policy from interesting schedule evidence to a stronger recipe candidate. |
| Compact learnable schedule control | Show that a low-parameter learnable controller, such as the 3-angle tree, can match or beat the best fixed schedules without unstable drift or overflow. | Promote the controller from exploratory mechanism to validated schedule candidate. |
| Edge representation promotion | Show matched-budget reruns that sign+mag + magnitude resample keeps its quality-per-edge advantage and justifies changing the current English candidate. | Promote a new edge format or mutation policy into the current recipe line instead of leaving sign+mag as evidence only. |
| Decay resample promotion | Show that single-neuron decay resample in `[0.01, 0.5]` keeps winning over local perturbation across reruns and budgets. | Promote the resample mutation policy into the current recipe line. |
| Low-theta / low-scale generalization | Re-run `INJ_SCALE=1.0` with low theta against the stronger current English recipe stack instead of the older baseline only. | Promote the low-scale line from older validated evidence into the current recipe discussion. |

## Archive Method

This record absorbs durable findings into canon chronology and only leaves raw material separate when it still carries unique source value.

| Retired surface | Current home |
|---|---|
| `Glossary`, `Hypotheses`, and roadmap-style status pages | This record now carries the live terms, active gates, and chronology. |
| Earlier evidence hub | [Vraxion Home](Home), [INSTNCT Architecture](INSTNCT-Architecture), and this record now split front-door, implementation, and chronology roles on purpose. |
| `Diamond Code v3 Architecture` | [INSTNCT Architecture](INSTNCT-Architecture) for the current line, plus `Older Timeline` below for the retained LCX-era record. |
| Original `Theory of Thought` ledger | [Theory of Thought](Theory-of-Thought) now carries the active theory line. |

- Keep a raw page only if it still carries unique config, ticket, source, script, or long-form result detail that is not safely compressed into canon prose.
- Demote raw material out of the primary stack and point back to its canon replacement surface.
- If a raw leaf and a canon page disagree, the canon page wins.
- When a new important finding lands, add it at the top of `Latest Entries`, keep `What changed` to 2-3 bullets and `Why it mattered` to 1-2 bullets, add one inline evidence object, and say explicitly whether the finding changed canon or stayed experimental.

**Inferred spans and remaining migration work**

- **2026-03-30 to 2026-04-01 (Inferred):** the public record suggests a transition period rather than a single headline discovery, with work shifting from Python-side architecture gains into Rust-port stabilization and benchmark-methodology hardening.
- **Archive migration still incomplete:** several probe leaves, research-intake pages, workbench-era notes, and benchmark-era raw dumps still exist outside this record. They should only disappear once their unique ticket/config/source value is safely captured elsewhere.

## Key Terms

<details>
<summary>Open key terms</summary>

**Current mainline**
What is actually shipped in code on `main`. If code and docs disagree about current behavior, the code wins.

**Validated finding**
A reproducible result that has not yet been promoted into the shipped code path.

**Experimental branch**
An active build target or design direction that is not the live public default.

**Confirmed**
Backed by direct evidence such as logs, code, charts, releases, or a reproduced run.

**Inferred**
Reconstructed from surrounding evidence rather than first-hand proof.

**Archived**
Historical context preserved for lookup, not a current default or active recommendation.

</details>
