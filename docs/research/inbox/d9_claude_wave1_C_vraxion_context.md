# D9 Wave 1 — Agent C: VRAXION Context

## 1. Genome / network structure today

**Rust side — the canonical network structure:**

The `Network` struct (S:\Git\VRAXION\instnct-core\src\network.rs:323) is the runtime spiking network:
```rust
pub struct Network {
    graph: ConnectionGraph,           // topology: source/target neuron indices
    spike: Vec<SpikeData>,           // per-neuron: charge[0..15], threshold[0..15], channel[1..8]
    polarity: Vec<i8>,                // per-neuron: ±1 (excitatory/inhibitory)
    activation: Vec<i8>,              // ephemeral state during propagation
    refractory: Vec<u8>,              // per-neuron: 0=ready, 1=cooling
    workspace: PropagationWorkspace,  // scratch buffer for propagation
    // ...CSR acceleration structures
}
```

On-disk (checkpoint) format in `NetworkDiskV1` (S:\Git\VRAXION\instnct-core\src\network\disk.rs:15):
```rust
pub struct NetworkDiskV1 {
    pub version: u8,                      // wire format version = 1
    pub graph: ConnectionGraphDiskV1,     // { neuron_count, sources[], targets[] }
    pub threshold: Vec<u32>,              // per-neuron [0..15]
    pub channel: Vec<u8>,                 // per-neuron [1..8]
    pub polarity: Vec<i32>,               // per-neuron ±1
}
```

Serialization: custom Serde-based format, stable wire format version 1. See `S:\Git\VRAXION\instnct-core\src\network\disk.rs:35` for validation rules.

**Initialization template (`InitConfig`):**

The initialization "recipe" (S:\Git\VRAXION\instnct-core\src\init.rs:46) is the closest current artifact to a "network descriptor":

```rust
pub struct InitConfig {
    pub neuron_count: usize,      // H
    pub phi_dim: usize,           // H / 1.618 (golden ratio overlap)
    pub chain_count: usize,       // 3-hop chain highways through overlap
    pub density_pct: usize,       // edge density target (e.g., 5%)
    pub threshold_max: u8,        // random threshold init range
    pub channel_max: u8,          // random channel init range
    pub inhibitory_pct: u32,      // % of neurons initialized inhibitory
    pub edge_cap_pct: usize,      // hard cap on edge growth
    pub accept_ties: bool,        // mutation acceptance mode
    pub propagation: PropagationConfig, // ticks, input duration, decay
}
```

Deterministic builder: `InitConfig::phi(neuron_count)` (S:\Git\VRAXION\instnct-core\src\init.rs:73) + `build_network(config, rng)` (S:\Git\VRAXION\instnct-core\src\init.rs:135).

**Python side:**

No explicit genome class found. Python blocks (byte_encoder, merger, embedder) operate on byte streams and low-level representations, not full network abstractions. No schema file exists.

**Critical gap:** No canonical intermediate representation ("genome ID" + provenance + rule trace) is logged. What exists: individual evolved networks, checkpoints with topology + parameters, but **no structured "z -> genome -> network" compiler and no latent seed parameter yet**.

---

## 2. φ (phi) — behavior fingerprint

**Definition:**
- Not a formal mathematical object yet; name collision with "phi" (golden ratio overlap dimension).
- In D8 audit context (S:\Git\VRAXION\docs\research\PHASE_D8_ARCHIVE_PSI_REPLAY_AUDIT.md), "phi" appears to refer to the 8-feature *geometry* used for cell atlas.

**Dimensions:**
The `PanelMetrics` struct (S:\Git\VRAXION\instnct-core\examples\evolve_mutual_inhibition.rs) contains:
```rust
struct PanelMetrics {
    panel_probe_acc: f64,      // probe accuracy (task-specific)
    unique_predictions: usize, // cardinality of predicted outputs
    collision_rate: f64,       // rate of duplicate predictions
    f_active: f64,             // fraction of active neurons per token
    h_output_mean: f64,        // mean output zone activation
    h_output_var: f64,         // variance of output zone activation
    stable_rank: f64,          // Frobenius/spectral norm ratio (Dong et al. 2021)
    kernel_rank: usize,        // numerical rank of output kernel
    separation_sp: f64,        // spec proof of class separation (undefined formula in logs)
}
```

These 8 scalar features (not 237-dimensional as phi_dim suggests) form the *behavior cell atlas space* in D8.6+. Logged per panel in `panel_timeseries.csv`.

**Pipeline:**
1. Network runs on task (S:\Git\VRAXION\instnct-core\examples\evolve_mutual_inhibition.rs)
2. Post-propagation metrics computed (accuracy, activation stats, rank)
3. Metrics bundled into CSV row with step, seed, H, edge count, etc.

No explicit "phi computation function" — metrics are computed inline during evaluation.

---

## 3. Ψ (psi)

**Definition:**
Predictive fingerprint of future fitness gain. A single scalar f64 per panel. Distinct from φ.

**Dimensions:** 1 (scalar).

**Computation:**
- Off-line model trained in D8.0 on historical panel data.
- Model: features (φ geometry + time metrics) → `psi_pred` (f64)
- Prediction uses seed-held-out cross-validation where available (S:\Git\VRAXION\docs\research\PHASE_D8_ARCHIVE_PSI_REPLAY_AUDIT.md: "seed-held-out CV").
- Validation: Spearman correlation = 0.6397 (psi) vs 0.3101 (score-only) (S:\Git\VRAXION\docs\research\PHASE_D8_ARCHIVE_PSI_REPLAY_AUDIT.md, decision block).

**Logged:** 
`psi_pred: Option<f64>` per panel state (S:\Git\VRAXION\instnct-core\examples\evolve_mutual_inhibition.rs). 

Used in parent selection: `P2_PSI_CONF = psi_pred * scan_depth_confidence` (S:\Git\VRAXION\docs\research\PHASE_D8_ARCHIVE_PARENT_MICROPROBE.md).

**Code location:** Psi model instantiated and invoked in `evolve_mutual_inhibition.rs` (references at `d8_p2_model.as_ref()` and `psi_pred` usage visible).

---

## 4. D8 atlas state

**Cell Atlas Layout (D8.6):**

- **Input:** 5840 panel rows from 8 source phases (B, D1–D7)
- **Output:** 156 atlas cells (behavior-space clusters)
- **Geometry basis:** 8-feature PanelMetrics (stable_rank, kernel_rank, separation_sp, collision_rate, f_active, unique_predictions, edges, accept_rate_window)
- **Projection:** Deterministic 2D PCA/SVD atlas for visualization; exact high-D geometry tracked via k-nearest neighbors (S:\Git\VRAXION\docs\research\PHASE_D8_CELL_ATLAS.md, "Geometry Contract")

**Latest Verdict (D8.7, Scan Delta, 2026-04-28):**

Target cell H128/C2 was downgraded post-scan:
- Pre-scan: 3 samples, `mean_psi = 0.0182`, `mean_future_gain = 0.0340`, `basin_precision = 1.000`
- Post-scan: 16 total samples, `mean_psi = 0.0092`, `mean_future_gain = 0.0123`, `basin_precision = 0.500`

Interpretation: "H128/C2 was worth scanning, but after enough samples it should not be promoted... better treated as de-risked / downgraded" (S:\Git\VRAXION\docs\research\PHASE_D8_TARGET_CELL_H128_C2.md).

Scan opened 8 new cells, reinforced 10 existing cells, cooled 0 cells across H∈{128, 256, 384}.

**Recent D8 Status:** Atlas is frozen for D8. Live runs now use D8's P2 psi model for parent selection, but no new atlas recomputation is active.

---

## 5. Latent / seed / template artifacts (if any)

**`InitConfig` is the closest proto:**

It is a deterministic *architecture descriptor* that fully specifies initialization:
- Neuron count H
- Phi-overlap fraction (golden ratio, hardcoded PHI = 1.618…)
- Chain highway topology (50 chains for H<512, 0 for H≥512)
- Edge density target (5% proven default)
- Parameter ranges (threshold, channel, polarity, inhibition %)

**What is missing for D9:**
- No *structured latent seed* (z) — only raw RNG seed
- No *rule-based compiler* (D(z) -> Network) — only random fill + mutations
- No *provenance logging* — no record of init seed, rule trace, or motif origin within genome
- No *intermediate representation* between InitConfig and Network object
- No *inverse mapping* from network → seed → rules

**Artifacts that *could* be reused:**
- `InitConfig` as a starting point for rule templates
- `PanelMetrics` (8-D) as φ baseline (though Gemini suggests richer features)
- Serde serialization infrastructure (NetworkDiskV1) for future genome provenance
- Checkpoint/archive system (exists but no latent code stored)

---

## Findings summary

1. **Genome structure exists and is stable:** Network is a spiking circuit (directed graph + per-neuron thresholds, channels, polarity); serializes via Serde V1 to edges + parameters. No "latent code z" wrapping it yet.

2. **Phi (behavior fingerprint) is 8-dimensional and well-instrumented:** PanelMetrics captures geometry (rank, separation, activation). These are logged, cross-validated for psi prediction (Spearman r=0.64), and used to build a 156-cell atlas.

3. **Psi is scalar, predictive, and validated:** Single f64 per panel predicting future gain. Beats score-only baseline. Used in live parent selection. Off-line regression model trained on historical D8 data.

4. **D8 atlas is currently frozen:** 156 cells, H∈{128,256,384}, latest scan suggests most cells are mature; no live improvement loop yet.

5. **Latent "z" does not exist yet:** InitConfig is the closest artifact — a deterministic template — but it is hardcoded (phi_dim ratio, chain count rules, density %). No structured seed hierarchy, no rule compiler, no provenance logging, no inverse mapping.

6. **No canonical "D9.0 toy" compiler scaffold in place:** Python blocks are low-level (byte encoding, embedding); no high-level genome generator that takes a seed → rules → topology.

---

## Compared to Gemini digest

**Alignment:**
- Gemini's D8 vs D9 distinction is validated: D8 (cell atlas observation) exists and is frozen. D9 (genome compiler) is not yet started.
- Psi prediction signal is confirmed and strong (Spearman 0.64 vs 0.31 baseline).
- Locality framing is applicable: PanelMetrics cluster into 156 cells, suggesting some structure; D9.0 can inherit this geometry.

**Divergence:**
- Gemini proposes z → D(z) → genome. VRAXION today has InitConfig → build_network(rng) → Network. The intermediate layer (structured rule compiler) is absent.
- Gemini suggests hierarchical seeds (root_seed → module_seed → motif_seed). VRAXION uses flat RNG seeding without rule-based derivation.
- Gemini pushes graph grammar / production rules. InitConfig is procedural (chains + fill + mutate), not declarative grammar.
- Gemini emphasizes provenance logging. VRAXION logs panels and psi predictions but not genome rule traces or motif origins.

**Gate status for D9.0 (per Gemini):**
- D9.0 should NOT require full graph grammar or neural decoder upfront → PASS (Python toy can use simple rules).
- D9.0 should NOT collapse validity → PASS (InitConfig + build are stable).
- Locality should be measurable → PASS (PanelMetrics 8-D, atlas cells, archive row IDs all available).
- Negative controls are possible → PASS (can shuffle rules, labels, or use random hash baseline).

**Conclusion:** The Gemini direction is **compatible with the codebase state**. VRAXION has the D8 observation layer and psi model. D9.0 can proceed offline as a Python toy that generates InitConfig-like templates + rules, compares locality against negative controls, and logs provenance. D9.1 integration into Rust (live compiler in loop) is blocked only by absence of actual rule compiler (not a stop-gate, just not yet implemented).
