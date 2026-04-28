# D9 Inbox: GPT Deep Research Latent Genome Compiler

Source: `C:/Users/kenes/Downloads/deep-research-report (2).md`

Date ingested: 2026-04-28

Status: source digest, not yet accepted architecture.

## Executive Read

GPT's report strongly agrees with the D9 direction:

```text
D8 = observed behavior atlas
D9 = compiler-native generative genome space
```

The central recommendation is to stop treating behavior cells as invertible. Instead, define a native latent genome chart:

```text
z -> D(z) -> executable network g -> phi(g), score, psi, atlas cell
```

The strongest unique contribution in this report is the warning that latent search needs a feasibility / in-distribution constraint. A search optimizer can otherwise drift into latent regions where the decoder is technically defined but produces invalid, uninteresting, or out-of-support networks.

## Accepted Claims For D9.0

### Compiler-Native First

The report correctly separates compiler-native exact inverse from legacy approximate inverse:

```text
compiler-native:
  z is logged at generation time, so g -> z is exact by lookup

legacy/external graph:
  g -> z is approximate inverse search or learned encoding
```

D9.0 should only claim exact inverse for compiler-generated samples.

### Rule-Based Compiler Before Learned Decoder

The report ranks approaches correctly for our immediate requirements:

1. rule/grammar compiler over restricted motifs
2. restricted DAG autoencoder later
3. target-cell inverse search after compiler dataset exists
4. safe branch trials only after target-cell hit rate is real

Accepted D9.0 interpretation:

```text
Build a deterministic hand-written toy compiler first.
Do not train a D-VAE/NAO/graph diffusion model in D9.0.
```

### Structured z Tuple

The proposed `z` form is useful:

```text
z = (z_c, z_d, s, v)
```

Where:

- `z_c`: continuous genes
- `z_d`: discrete genes
- `s`: deterministic seed/tie-breaker
- `v`: compiler version

This is compatible with the Gemini digest's structured recipe framing.

### Feasibility / OOD Constraint

This is a key addition over the Gemini report.

The report argues that target-cell search over `z` must optimize:

```text
reward(z) - lambda * OOD(z)
```

or include equivalent constraints:

- density prior
- feasibility classifier
- decoder validity model
- distance-to-training-manifold penalty

Accepted D9.0 simplification:

For a hand-written toy compiler, OOD can mean:

```text
invalid decode
constraint repair required
edge/motif count outside target range
latent tile with low historical validity
```

### Topology / Parameter Split

The report recommends separating:

```text
topology compiler
parameter / weight generator
```

This is useful but should be staged:

- D9.0 toy: topology + tiny deterministic scalar parameters only
- D9.1 VRAXION-lite: topology grammar plus threshold/edge parameter genes
- later: optional hypernetwork/parameter emitter

## Deferred Claims

### Restricted DAG Autoencoder

The report recommends a D-VAE/NAO-like DAG autoencoder for archive integration.

This is plausible, but not D9.0.

Deferred to:

```text
D9.3 or later: legacy/archive approximate inverse
```

Reason:

- needs canonical graph serialization
- needs enough training examples
- introduces learned-model confounds
- does not test the core hand-written compiler hypothesis

### Flow / Diffusion / Autoregressive Graph Models

GraphRNN, GRAN, GraphAF, GraphDF, DiGress, NGG, GPrinFlowNet and related models are background only.

Deferred because they:

- require more data/compute
- weaken inspectability
- do not give the exact native `z` logging guarantee
- can produce latent dead-region issues

### Safe Branch Trial Integration

The report's side-branch framing is good:

```text
compiler-sourced states can run as short-lived side branches
champion is never replaced without measured improvement
```

But this is not D9.0.

Deferred until:

- compiler validity passes
- locality passes
- target-cell hit rate is measured
- feasibility constraints work

## Rejected / High-Risk For Immediate Implementation

### Behavior Cell Direct Inversion

Rejected for D9.0:

```text
cell(phi) -> arbitrary graph
```

Reason:

Behavior-cell partitions are not generative models.

### Surrogate-Only Targeting

Rejected for D9.0:

```text
train predictor for psi/cell and trust it without evaluator confirmation
```

Reason:

D8/D7 already showed that offline signals can fail live. Any latent target-cell hit must be confirmed by decode + evaluate.

### Neural Decoder First

Rejected for D9.0.

Reason:

The whole point of D9.0 is to falsify the deterministic compiler geometry before adding black-box learned components.

## Proposed z Fields From Report

Useful fields to consider for D9.0/D9.1:

```text
motif_family          categorical
depth_bias            continuous
skip_bias             continuous
recur_bias            continuous
fanout_bias           continuous
sparsity              continuous
threshold_regime      categorical
inhibition_bias       continuous
weight_scale          continuous/log-scale
noise_seed            uint64 tie-breaker
weight_seed           uint64 tie-breaker
compiler_version      metadata
```

D9.0 should use a smaller toy subset:

```text
motif_family
edge_density/sparsity
loop_bias
inhibition_bias
threshold_regime
fanout_bias
symmetry_bias
seed
compiler_version
```

## Metrics To Keep

The report emphasizes:

- valid network rate
- exact native roundtrip by logging `z`
- z-distance vs genome distance
- z-distance vs behavior distance
- basin clustering in `z`
- target-cell hit rate
- evaluations-to-first-hit
- feasibility / OOD failure rate
- progressive scan efficiency

Accepted D9.0 metric set:

```text
valid_network_rate
determinism_exact_rate
native_roundtrip_log_rate
latent_to_genome_spearman
latent_to_behavior_spearman
nearest_neighbor_preservation
tile_basin_enrichment
progressive_scan_efficiency
negative_control_gap
```

## Negative Controls Needed

The report does not focus as much as Gemini on explicit controls, so D9 synthesis must add them:

```text
random_hash_decoder
nonlocal_decoder
shuffled_fitness
shuffled_behavior
random_cell_assignment
seed_only_decoder
```

Important addition from GPT report:

```text
OOD / feasibility control
```

For example, an optimizer that maximizes a surrogate but lands mostly in invalid/unsupported decodes must fail.

## Implementation Artifacts Proposed By Report

The report suggests these eventual tools:

```text
build_phase_d9_compiler_dataset.py
train_phase_d9_rule_compiler.py
train_phase_d9_dag_autoencoder.py
train_phase_d9_surrogate.py
run_phase_d9_target_cell_search.py
build_phase_d9_compiler_dashboard.py
```

Accepted D9.0 narrowing:

Only implement one first tool:

```text
tools/analyze_phase_d9_latent_genome_toy.py
```

It should output:

```text
output/phase_d9_latent_genome_toy_YYYYMMDD/analysis/summary.json
latent_samples.csv
locality_audit.csv
tile_scan_audit.csv
negative_controls.csv
docs/research/PHASE_D9_LATENT_GENOME_TOY.md
```

## Comparison Against Gemini Digest

Agreement:

- deterministic compiler first
- rule/grammar over neural decoder
- exact inverse only for compiler-native generated genomes
- structured `z`, not random hash
- locality metrics required before geometric claims
- Python-first toy audit before Rust integration

GPT-specific useful addition:

- feasibility / OOD penalty for latent search
- target-cell acquisition framing
- topology/parameter split
- explicit staged path from compiler dataset to safe branch trials

Gemini-specific useful addition:

- stronger negative-control emphasis
- DNP gate framing
- hierarchical seed provenance
- stronger warning against overclaiming inverse parsing

Net synthesis:

```text
D9.0 = deterministic toy compiler + locality + negative controls
D9.1 = VRAXION-lite grammar only if D9.0 passes
D9.2 = target-cell scan/search with feasibility constraints
D9.3+ = learned legacy inverse / DAG AE only later
```

## Bottom Line

GPT's report strengthens the D9 plan by adding latent feasibility and target-cell acquisition concerns. It should not move us toward learned decoders immediately. The safe near-term plan remains:

```text
hand-written deterministic compiler
synthetic toy graphs
locality/searchability audit
negative controls
no Rust live behavior change
```
