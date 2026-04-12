# VRAXION Validated Findings

Canonical evidence summary. Repo-tracked docs are canonical; the GitHub wiki is a mirrored secondary surface.

## Current State

The repo is in a transition state:

- **Released public tag:** `v5.0.0-beta.1` — Rust language-evolution beta
- **Current mainline on `main`:** Rust grower (`instnct-core/examples/neuron_grower.rs`)
- **Reference/support lane:** Python `instnct/model/graph.py`

### Grower lane: proven on `main`

| Finding | Result | Status |
|---|---|---|
| **Bias-free threshold neuron** | Persistent grower now stores/evaluates neurons directly as `dot >= threshold`; redundant bias removed from search and state. | **Current mainline** |
| **Scout oracle** | Mainline parent ranking now uses single-signal scores, connect-all probe weights, and pair-lift shortlist before ternary search. | **Current mainline** |
| **Non-strict accept gate** | Compositional stepping-stones are no longer rejected on equal-val intermediate steps; `four_parity` now reaches 100%. | **Current mainline** |
| **Persistent grower state** | Authoritative resume file is bias-free `state.tsv` with per-step JSON checkpoints. | **Current mainline** |
| **4-bit curriculum entry** | Small exhaustive four-bit pattern tasks solve quickly and act as a compositional smoke floor for the grower. | **Validated finding** |
| **Exact byte readout via latent LUT** | On `1 byte + 4 opcode -> 1 byte`, direct bitbank stalls at `75%`, but the frozen hidden binary latent supports a collision-free LUT translator at `100%` exact over the full `1024`-sample domain. | **Validated finding** |

### Released beta.1 Rust lane: historical peak results

| Finding | Result | Status |
|---|---|---|
| **Smooth cosine-bigram fitness** | 21.7% peak with 1+1 ES (+2.6pp over stepwise argmax) | **Current mainline** — default in `evolve_language.rs` |
| **1+9 jackpot selection** | 24.6% peak (+3.4pp over 1+1 ES). Python parity achieved. | **Current mainline** — `evolution_step_jackpot()` in library |
| **Addition learning** | 80% on 0-4 + 0-4 from empty network (83 edges). Freq baseline 20%. | **Validated finding** |
| **Empty start >> prefilled** | 80% with 83 edges vs 64% with 3400 prefilled edges on addition task | **Validated finding** |
| **W mutation nearly useless** | Adaptive ops test: 0% accept rate for projection mutations across all seeds | **Validated finding** — W reduced to 5% in schedule |
| **Loop mutations** | `mutate_add_loop(len=2/3)` added to library. Critical for sparse/empty network bootstrap. | **Current mainline** — 10% of mutation schedule |
| **Charge pattern similarity** | Trained addition networks: same-sum charge cosine ~0.91, cross-sum ~0.92. Topology is under-differentiated. | **Validated finding** |
| CSR skip-inactive | 8.7x at H=256, 19x at H=512 | **Current mainline** |
| Learnable int8 readout | `Int8Projection` with `raw_scores()` for smooth fitness | **Current mainline** |
| Theta floor / zero-theta collapse | Zero-theta networks collapse into indistinguishable activation patterns | **Validated finding** |
| Chain-50 init | Raises worst-seed floor from 6.5% to 16.1% at H=256 | **Current mainline** for H<512 |

### Python lane: historical reference results

| Finding | Result | Notes |
|---|---|---|
| Breed + crystallize | 24.4% | Consensus structure + pruning (2026-03-29) |
| Learnable channel (C19 Wave Gating) | 23.8% | Cos-shaped LUT, replaces sin/phase/rho |
| Voltage medium leak schedule | 22.11% peak / 21.46% plateau | Fixed schedule, not promoted to defaults |
| Word-pair log-likelihood eval | 23.8% | Task-memory evaluation, not canonical mainline |

### Key architectural findings (cross-lane)

| Finding | Evidence |
|---|---|
| Binary masks sufficient | Binary {0,1} matches ternary accuracy (86.5%). Multiply-free forward pass. |
| Topology > edge precision | Binary edges match float at all tested scales |
| Hub-inhibitor architecture | 10% inhibitory neurons with 2x fan-out (matches FlyWire biological data) |
| Sparse evolution > dense prefill | Evolution with few edges produces targeted circuits; dense prefill = noise |
| Fitness function shape matters | Smooth (continuous) > discrete (step). The #1 bottleneck was not architecture but fitness signal quality. |

## How To Read This Page

- **Current mainline**: shipped in code on `main` right now.
- **Validated finding**: experimentally supported, not yet promoted into canonical code path.
- **Experimental branch**: active target or design direction, not yet validated as default.

If code and docs disagree, **code wins for "Current mainline."**
