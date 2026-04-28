# D9 Wave 2 — Agent E: Architecture Designs

Three designs with genuinely different inductive biases. All deterministic Python; no learned components. Anti-hash arguments stated per field.

---

## Design 1: Parametric InitConfig (PIC)

Continuous lift of `InitConfig` (Agent C: `instnct-core/src/init.rs:46`). `z` directly parameterizes existing structural knobs; sub-seed only resolves *which* neurons fill each slot.

### 1.1 z specification (12 dims, all f32 in [0,1] unless noted)
- `z[0] density`           edge density target → `0.01 + 0.14·z[0]` [direct empirical, Agent C density_pct=5%]
- `z[1] inhib_ratio`        inhibitory fraction → `[0.0, 0.5]`
- `z[2] phi_overlap`        golden-ratio overlap fraction → `[0.3, 0.8]`
- `z[3] chain_density`      3-hop chain count per 100 neurons → `[0,5]`
- `z[4] thr_mean`           threshold mean in [0..15]
- `z[5] thr_spread`         threshold std
- `z[6] channel_skew`       channel distribution Dirichlet temp
- `z[7] recurrence_bias`    P(target_idx<source_idx) bias
- `z[8] symmetry`           mirror-pair fraction
- `z[9] modularity`         block-diagonal weight on edge sampler
- `z[10] irreg_amp`         irregularity-slot amplitude in [0, 0.10] (capped)
- `z[11] irreg_phase`       irregularity-slot rotation in [0, 1)
- `H` (neuron count) is treated as scenario-fixed, not in z.

### 1.2 Decoder algorithm
```python
def D_pic(z, root_seed, H):
    cfg = InitConfigContinuous.from_z(z, H)        # deterministic field map
    rng_struct = SubseedRNG(root_seed, "structure")
    G = empty_graph(H)
    polarity = assign_polarity(H, z[1], rng_struct.fork("polarity"))
    blocks   = partition_blocks(H, z[9], rng_struct.fork("blocks"))
    place_chains(G, H, z[3], rng_struct.fork("chains"))
    sample_edges(G, target_density=z[0],
                 recurrence_bias=z[7], symmetry=z[8],
                 modularity=z[9], blocks=blocks,
                 rng=rng_struct.fork("edges"))
    apply_irregular_overrides(G, z[10], z[11],
                              budget=0.10*edge_count(G),
                              rng=rng_struct.fork("irreg"))
    thr     = sample_thresholds(H, z[4], z[5], rng_struct.fork("thr"))
    channel = sample_channels(H, z[6], rng_struct.fork("ch"))
    return Genome(G=G, polarity=polarity, thr=thr, channel=channel,
                  rule_trace=trace_log)
```
Edge sampler uses a deterministic *score* `s(i,j;z) = w0 + w1·rec(i,j;z[7]) + w2·sym(i,j;z[8]) + w3·block(i,j;z[9])`; top-`K(z[0])` edges chosen, ties broken by sub-seed.

### 1.3 Sub-seed scheme
`root_seed → SHA256(root_seed||stage)` for each stage in {polarity, blocks, chains, edges, irreg, thr, channel}. Sub-seed acts only as **tie-breaker** in `top-K` selection and in inhibitory neuron picking when the count from `z[1]·H` is non-integer or scores tie. **Field-by-field anti-hash table:**
- `z[0]` controls edge *count* directly (K = round(z[0]·H²)); seed only picks among same-score edges.
- `z[1]` controls inhibitory *count*; seed picks identities.
- `z[7],z[8],z[9]` reshape the score function `s`; seed never enters `s`.
- `z[10]` caps how many overrides exist; seed picks which sites.

### 1.4 Genome JSON schema
```json
{"version":"d9.0-pic-1",
 "z_logged":[...12 floats...],
 "root_seed":"u64-hex",
 "rule_trace":[{"rule_id":"place_chain","site":[i,j,k],"subseed":"hex"},
               {"rule_id":"sample_edge","site":[i,j],"score":0.81,"subseed":"hex"},
               {"rule_id":"set_polarity","site":i,"value":-1,"subseed":"hex"},
               {"rule_id":"irreg_override","site":[i,j],"op":"add","subseed":"hex"}],
 "genome":{"H":H,"edges":[[i,j],...],"polarity":[...],"thr":[...],"channel":[...]},
 "validity_flag":true,
 "validity_reasons":[]}
```

### 1.5 Anti-random-hash argument
For each structural z-field there exists a closed-form `F_k` such that the marginal property `P_k(D(z)) = F_k(z[k]) + O(1/H)` independent of seed. Concretely: edge count K is exactly `round(z[0]·H²)`; inhibitory count is exactly `round(z[1]·H)`; recurrence imbalance `Σ_{i>j}edge − Σ_{i<j}edge` has expectation monotone in `z[7]`. Therefore `‖D(z)−D(z+ε)‖_edit` is bounded by `O(ε·H²)` regardless of seed: a 0.01 perturbation of `z[0]` flips at most ~`H²/100` edges, not the entire graph. Seed cannot force a discontinuous re-selection because there is no codon→rule lookup. [theoretical, fortified by Agent A NEAT alignment + rejecting GE wrapping per Agent B]

### 1.6 Approximate inverse encoder
For external network `g`, set `z_hat[0] = density(g)`, `z_hat[1] = inhib_frac(g)`, `z_hat[7] = forward_minus_back_edge_ratio(g)`, etc. — closed-form moment matching, no search. Only structural fields recover; irregularity fields default to 0. Stored z + rule_trace are exact inverse for D9-generated networks.

### 1.7 Validation order (cheapest killer first)
1. `DNP_VALIDITY_COLLAPSE` — extremes of z[0],z[1] could yield 0-edge graphs. Cheapest, runs in seconds.
2. `IDENTITY_DECODER_CHECK` — assert `entropy(genome) > entropy(z) + log2(H!)/8` (graph capacity ≫ z capacity).
3. `LOCALITY_SPEARMAN` — z-distance vs edit-distance, expected ρ>0.5 by construction.
4. `CONTROL_PARITY` vs random-hash decoder.
5. `IRREGULAR_BASIN` — synthetic deceptive landscape (Agent A Clune 2011 regime test).

### 1.8 Compared to Gemini digest
**Agreement:** structured recipe, hierarchical sub-seeds as tie-breakers, provenance log. **Extension:** Gemini's "node groups, edge density, recurrence loop bias, inhibition/excitation ratio" list maps 1:1 to z fields; we make the mapping explicit and add irregularity slot. **Contradiction:** none. **Beyond Gemini:** explicit per-field anti-hash table (Gemini hand-waves "structured ≠ hash").

---

## Design 2: Motif-Mixture Decoder (MMD)

`z` is a continuous mixture over a small library of hand-specified motifs (Agent A Hornby–Pollack analog; Agent A finding 4 + cellular encoding). Counts of each motif are continuous in mixture weights, avoiding GE-style codon-to-rule discontinuity.

### 2.1 z specification (16 dims)
- `z[0..7] m[8]` softmax mixture over 8 motifs (chain, mirror_pair, fan_in, fan_out, mutual_inhibitor, recurrent_loop, lateral_block, sparse_random) — each `m[k]∈[0,1]`, post-softmax sums to 1.
- `z[8] total_motif_budget`  fraction of `H` covered by motifs → `[0.3, 0.95]`
- `z[9] glue_density`          extra inter-motif edges in `[0, 0.05]`
- `z[10] glue_recurrence_bias`
- `z[11] inhib_ratio`
- `z[12] thr_mean`
- `z[13] thr_spread`
- `z[14] irreg_amp`            ≤ 0.10
- `z[15] irreg_locality`       irregularity site clustering

### 2.2 Decoder algorithm
```python
def D_mmd(z, root_seed, H):
    weights = softmax(z[0:8])
    counts  = round_preserving_sum(weights * (z[8]*H/avg_motif_size))  # int counts
    G = empty_graph(H); cursor = 0
    rng = SubseedRNG(root_seed)
    for k, n_k in enumerate(counts):
        for i in range(n_k):
            site = pick_site(cursor, H, rng.fork(("motif",k,i)))
            apply_motif_template(G, MOTIFS[k], site)   # deterministic template
            cursor += motif_size(k)
    add_glue_edges(G, z[9], z[10], rng.fork("glue"))
    apply_irregular_overrides(G, z[14], z[15], budget=0.10*edge_count(G),
                              rng=rng.fork("irreg"))
    polarity = assign_polarity(H, z[11], rng.fork("pol"))
    thr      = sample_thresholds(H, z[12], z[13], rng.fork("thr"))
    return Genome(...)
```

### 2.3 Sub-seed scheme
Sub-seed selects *placement sites* of motifs and the identity of glue edges. **Critical:** counts come from continuous weights, not from `seed mod n_motifs`. **Field dominance:**
- `z[0..7]` set how many of each motif exist (count is `round(softmax(z)·budget)`).
- `z[8]` sets total budget.
- `z[9]` sets glue count.
- `z[14]` sets irregularity edge count.
- Seed only chooses *where* to place each motif and *which* node pairs become glue, not how many or which type.

### 2.4 Genome JSON schema
```json
{"version":"d9.0-mmd-1",
 "z_logged":[...16 floats...],
 "root_seed":"hex",
 "motif_counts":{"chain":4,"mirror_pair":2,...},
 "rule_trace":[{"rule_id":"motif_place","motif":"mutual_inhibitor",
                "site":[anchor_node],"subseed":"hex"},
               {"rule_id":"glue_edge","site":[i,j],"subseed":"hex"},
               {"rule_id":"irreg_override","site":[i,j],"op":"flip","subseed":"hex"}],
 "genome":{...},"validity_flag":true,"validity_reasons":[]}
```

### 2.5 Anti-random-hash argument
The motif-count vector is a **continuous function of z**: `n_k(z) = round(softmax(z[0:8])_k · z[8] · H / size_k)`. A perturbation `Δz[j]=ε` moves `n_k` by at most `±1` per motif (roundoff boundary), so the genome edit distance is bounded by `Σ_k size_k · 1 + glue_drift = O(max_motif_size)`, not `O(H²)`. Critically, this is **discontinuous only at integer rounding boundaries**, which occur on a measure-zero subset of z; locality holds almost everywhere. This is the precise contrast with Grammatical Evolution (Agent B / Rothlauf-Oetzel 2006): GE indexed rules by `codon mod n_rules`, producing modular-arithmetic discontinuities at *every* codon boundary; MMD has discontinuities only at count-rounding boundaries. We further smooth by using stochastic rounding seeded with `hash(z[0:8])` so neighbors do not both round in opposite directions on the same boundary.

### 2.6 Approximate inverse encoder
Subgraph isomorphism count of each motif in `g` → normalize → `z_hat[0:8]`; total motif coverage → `z_hat[8]`. NP-hard in general but Agent A flagged this as deferred. For D9.0 toy with `|MOTIF_LIB|=8` and motif size ≤4, simple counting suffices. Glue and irregularity fields default to 0.

### 2.7 Validation order
1. `LOCALITY_SPEARMAN` first — this is the high-risk axis (rounding boundaries) and the cheapest killer for MMD.
2. `DNP_VALIDITY_COLLAPSE` (motifs may overlap on small H).
3. `IDENTITY_DECODER_CHECK`.
4. `IRREGULAR_BASIN` — MMD is expected to do *better* than PIC on irregular basins because motif mixture provides coarse-grained discrete structure; this is the design's reason for existing.
5. `CONTROL_PARITY` vs shuffled-motif-library decoder.

### 2.8 Compared to Gemini digest
**Agreement:** "motif mix" appears in Gemini's recipe list. **Extension:** we make motif-count *continuous* via softmax, which Gemini does not specify (and which is the load-bearing escape from GE failure). **Contradiction:** Gemini implies motif selection by seed; we forbid that and require continuous weights.

---

## Design 3: Geometric Placement Field (GPF)

CPPN-flavored hand-coded geometry decoder (Agent A finding 2 + Stanley 2007 background). Each neuron has fixed coordinates; `z` parameterizes a deterministic scalar field over coordinate pairs, edges are top-K by field value.

### 3.1 z specification (10 dims)
- `z[0..3] f_coef[4]`  coefficients of polynomial kernel `f_z(x_i,x_j) = z[0]+z[1]·d+z[2]·d²+z[3]·sign(x_j−x_i)` where `d=‖x_i−x_j‖`
- `z[4] freq`            spatial frequency for sinusoidal modulation
- `z[5] phase`           phase
- `z[6] density`         fraction of edges retained → K = round(z[6]·H²)
- `z[7] inhib_ratio`
- `z[8] thr_mean`
- `z[9] irreg_amp`       ≤ 0.10

Neuron coordinates: deterministic grid `x_i = i/H` (1D). Trivially extends to 2D layouts inherited from VRAXION's `phi_dim` overlap.

### 3.2 Decoder algorithm
```python
def D_gpf(z, root_seed, H):
    coords = deterministic_grid(H)
    field  = lambda i,j: poly(coords[i],coords[j],z[0:4]) \
                       + sin(z[4]*(coords[i]+coords[j])+z[5])
    scores = [(field(i,j), i, j) for i in range(H) for j in range(H) if i!=j]
    K = round(z[6]*H*(H-1))
    edges = top_k(scores, K, tie_break=SubseedRNG(root_seed,"edges"))
    polarity = assign_polarity_by_coord(coords, z[7],
                                        rng=SubseedRNG(root_seed,"pol"))
    thr      = sample_thresholds(H, z[8], rng=SubseedRNG(root_seed,"thr"))
    apply_irregular_overrides(edges, z[9], budget=0.10*K, ...)
    return Genome(...)
```

### 3.3 Sub-seed scheme
Sub-seed used **only** for tie-breaking in `top_k` and in non-integer polarity counts. Field-by-field:
- `z[0..5]` *fully define* the score function `f_z`; seed never enters `f_z`.
- `z[6]` exactly sets edge count.
- `z[7]` exactly sets inhibitory count.

### 3.4 Genome JSON schema
```json
{"version":"d9.0-gpf-1",
 "z_logged":[...10 floats...],
 "root_seed":"hex",
 "field_coefs":[z0..z5],
 "rule_trace":[{"rule_id":"edge_place","site":[i,j],
                "score":0.43,"subseed":"hex"}, ...],
 "genome":{...},"validity_flag":true}
```

### 3.5 Anti-random-hash argument
`f_z` is Lipschitz-continuous in `z` with constant `L = max(|d|,|d²|,1)`. For `Δz=ε`, every score `f_z(i,j)` changes by at most `L·ε`. The top-K set changes only at score-crossings; the number of crossings between two top-K sets is bounded by `O(L·ε·H²)`. Therefore `edit_distance(D(z), D(z+ε)) = O(ε·H²)`, identical bound to PIC and strictly better than any seed-only scheme (which would have `Ω(H²)` distance for any ε). This is the same Lipschitz argument used in CPPN locality reasoning (Agent A Stanley 2007), here made concrete with a scalar Lipschitz constant.

### 3.6 Approximate inverse encoder
Fit `f_z` by least-squares regression of `1[edge(i,j)]` on the polynomial+sinusoid basis evaluated at neuron coordinates. Closed-form 6-parameter regression. `z_hat[6]` from edge density, `z_hat[7]` from inhib fraction. Irregularity fields default to 0.

### 3.7 Validation order
1. `IDENTITY_DECODER_CHECK` first — GPF's main risk is `f_z` degenerating to a constant (all coefs ≈ 0), producing trivial graphs. Cheapest to detect.
2. `DNP_VALIDITY_COLLAPSE`.
3. `LOCALITY_SPEARMAN` — expected ρ>0.6 by Lipschitz argument.
4. `IRREGULAR_BASIN` — GPF is at risk on irregular basins (Agent A Clune 2011 finding 1: smooth fields fail on irregular targets).
5. `CONTROL_PARITY` vs random-coef baseline.

### 3.8 Compared to Gemini digest
**Agreement:** "symmetry/mirror bias" maps to `z[3]` directional term. **Extension:** the polynomial+sinusoid kernel is more concrete than Gemini's verbal recipe. **Contradiction:** Gemini's "channel grouping" is harder to express geometrically; GPF defers it to a follow-up dimension if needed.

---

## Cross-design comparison

| Design | z dim | Validity risk | Locality (theory) | Anti-hash strength | Irregular-basin handling | Compute |
|--------|------:|---------------|-------------------|--------------------|--------------------------|---------|
| PIC    | 12    | low           | O(ε·H²) bound, smooth | strongest (closed-form per field) | weak (10% irreg slot only) | O(H²) |
| MMD    | 16    | medium (motif overlap) | piecewise-smooth, integer boundaries | strong (continuous mixture) | strong (motif library is discrete-irregular) | O(H²) |
| GPF    | 10    | low-medium (field collapse) | O(ε·H²) Lipschitz | strong (Lipschitz constant) | weak-medium (smooth field) | O(H²) |

---

## Recommended primary design and reasoning

**Primary: Design 1 — PIC.** Three reasons.
1. **Anti-hash by construction:** every z-field has a closed-form quantitative property it controls, with seed strictly relegated to tie-breaking. Strongest defense against the Rothlauf-Oetzel GE failure mode.
2. **Closest VRAXION precedent:** PIC is a continuous lift of `InitConfig` (Agent C). The Rust builder is already deterministic and validated; the toy compiler is a few hundred lines of Python.
3. **Lowest validation risk:** validity collapse is the cheapest gate, and PIC is engineered to pass it.

**Secondary (escalation if PIC fails irregular-basin test): Design 2 — MMD.** The motif library is an explicit discrete-structure injection that addresses Clune 2011 regime-dependency. Run only after PIC passes the first three gates and fails the irregular-basin landscape.

**Design 3 — GPF as alternative-geometry option** if D9.1 needs CPPN-style geometry alignment with Agent C `phi_dim` overlap. Not primary because field collapse risk is harder to detect early.

---

## Background only

- **Grammar VAE / JT-VAE / learned grammar decoders** (Agent A Findings 3, Hornby-Pollack analogy, Agent B Volz GAN): excluded by D9.0 hard constraint (no learned decoders). Resurface in D9.2+ if static grammars exhibit irregular-basin failure that motif libraries cannot fix.
- **nc-eNCE / edNCE graph grammar formalisms** (Gemini digest): too heavy; PIC and MMD subsume the needed expressiveness with simpler machinery.
- **NEAT speciation, novelty search, CMA-ME emitters** (Agent B): orthogonal to D(z) decoder design; belongs to D9.1+ scan strategy.
- **MAP-Elites cell-as-coordinate vs occupant** (Agent B): D9 chooses z-as-coordinate, an acknowledged untested commitment; not a decoder design issue.
- **BOP-Elites surrogate scan** (Agent B): D9.1+ when evaluation cost rises.
- **Codon→mod→rule schemes (Grammatical Evolution)**: explicitly excluded per Rothlauf-Oetzel locality failure.

### Critical Files for Implementation

- S:\Git\VRAXION\instnct-core\src\init.rs
- S:\Git\VRAXION\instnct-core\src\network.rs
- S:\Git\VRAXION\instnct-core\src\network\disk.rs
- S:\Git\VRAXION\instnct-core\examples\evolve_mutual_inhibition.rs
- S:\Git\VRAXION\docs\research\inbox\d9_gemini_genome_compiler_design.md
