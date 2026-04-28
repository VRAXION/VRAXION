# D9 Wave 1 — Agent D: Locality Measurement & Benchmark Theory

## Compared to Gemini digest

Gemini's design is accepted. This report adds: (1) concrete N and permutation counts for every recipe, (2) verified Mantel failure modes (spatial autocorrelation inflates Type I error), (3) Weinberger's τ = −1/ln(ρ(1)) as a second locality axis Gemini omitted, (4) identity-decoder pitfall Gemini missed, (5) Python-runnable landscape generators. Gemini's progressive-scan claim is independently supported by progressive gradient walk literature.

---

## Test recipes (concrete, runnable)

### Recipe 1: near-z → near-genome locality (Mantel test)

1. Sample N=500 pairs (z_i, z_j) uniformly from latent space. [theoretical: N synthesised from Quilodrán 2025 n=50–200 range scaled up]
2. Compute D_z: N×N Euclidean distance matrix over z vectors.
3. Compute D_g: N×N genome distance matrix (Hamming or normalised edit distance over genome bytes).
4. Run Mantel test: Pearson r_m over all n(n-1)/2 off-diagonal pairs; generate null by 999 permutations of row/column identities of D_g. [direct empirical: PMC3873175]
5. **Pass**: r_m > 0.3, p < 0.05 (two-tailed). Report r_m as effect size; p alone is insufficient.
6. **Failure mode**: if z is spatially autocorrelated (smooth grid scan), Mantel inflates Type I. Use random z samples, not grid. [direct empirical: PMC11696488]

D9.0 implication: Implement as `scipy.stats.pearsonr` on flattened upper-triangle pairs; use `numpy` permutation for null; 999 shuffles run in <5 s for N=500.

### Recipe 2: minimum decoder validation test set

Compose these five checks before plugging decoder into any live system. [theoretical: composition of verified pieces, no canonical single-paper protocol found]

1. **Validity rate**: generate 1000 networks; require ≥ 99% structurally valid (connected, no zero-degree nodes, no self-loops unless allowed). Threshold: DNP_VALIDITY_COLLAPSE at < 0.99.
2. **Locality z↔genome**: Recipe 1 above; r_m > 0.3, p < 0.05.
3. **Locality z↔behavior**: same Mantel protocol with D_b = behavior-fingerprint distance; r_m > 0.15, p < 0.05.
4. **FDC proxy**: sample 500 z points; compute fitness f(z) and distance d(z, z*) to best known z; require r_FDC > 0.15 to pass "not-deceptive". [unverified: exact threshold from Jones & Forrest 1995 ICGA; PDF failed to render but threshold cited in multiple search snippets]
5. **Control parity**: run all metrics on NCI_RANDOM_HASH_DECODER; real decoder must exceed hash on every metric by ≥ 0.20 with non-overlapping 95% bootstrap CIs.

### Recipe 3: random-hash negative control protocol

```python
# NCI_RANDOM_HASH_DECODER
import hashlib, numpy as np
def hash_decoder(z_bytes: bytes, genome_len: int) -> bytes:
    h = hashlib.sha256(z_bytes).digest()  # 32 bytes
    rng = np.random.default_rng(int.from_bytes(h[:8], 'big'))
    raw = rng.integers(0, 256, size=genome_len, dtype=np.uint8)
    return raw.tobytes()
```

Steps:
1. Run hash_decoder on all N=500 z samples.
2. Compute same D_z, D_g, D_b matrices; run same Mantel recipe.
3. Compute r_hash.
4. **Pass for real decoder**: r_real ≥ r_hash + 0.20, CI non-overlapping (bootstrap 1000 resamples).
5. If r_real < r_hash + 0.05: decoder is hash-like; reject (DNP_BEHAVIOR_HASHLIKE). [direct empirical: locality claim only valid if it beats uncorrelated baseline]

D9.0 implication: Identical code path for real decoder and hash decoder; only the decode function differs.

### Recipe 4: basin clustering quality

1. Run decoder on 500 z samples; compute behavior fingerprint B for each.
2. Cluster B into K basins (K-means or HDBSCAN).
3. **Silhouette score**: require mean silhouette > 0.5 for "reasonable" basin separation, > 0.7 for "strong". [direct empirical: sklearn documentation, Wikipedia Silhouette]
4. **ARI**: if ground-truth labels exist, require ARI > 0.4. Random labeling gives ARI ≈ 0. [direct empirical: sklearn adjusted_rand_score docs]
5. **Warning**: basin labels derived from the decoder itself make ARI circular — use independent fitness oracle labels when available.

D9.0 implication: `sklearn.metrics.silhouette_score(B, cluster_labels)` directly; for ARI use `adjusted_rand_score(true_labels, pred_labels)`.

### Recipe 5: progressive scan efficiency vs random

1. Define a 2D tile grid over z-space (e.g., 10×10 tiles).
2. Run two strategies: (a) tile-progressive (scan tiles in raster order), (b) uniform random z.
3. After each batch of 50 evaluations, record best-fitness-found and unique-basins-found.
4. **Pass**: progressive accumulates unique basins at ≥ 1.5× rate of random after 300 evaluations. [theoretical: supported by progressive gradient walk literature, ACM GECCO 2018]
5. If progressive ≤ random: DNP_SCAN_NO_GAIN; z-space has no exploitable structure.

D9.0 implication: One curve plot per strategy; gate is rate ratio, not absolute count.

---

## Toy landscape generators (Python-implementable)

### smooth basin
```python
def smooth(x, x_star): return -np.sum((x - x_star)**2)
```

### deceptive basin (fully deceptive k-bit trap, Whitley 1991 / Deb & Goldberg 1992)
```python
def trap_k(bits, k=4):  # operate on k-bit blocks
    blocks = [bits[i:i+k] for i in range(0, len(bits), k)]
    return sum(k if sum(b)==0 else sum(b)-1 for b in blocks)
# global optimum: all-zeros; schemata mislead GA toward all-ones
# [unverified: Whitley 1991 original; PDF not rendered]
```

### multi-basin (NK-style, K=3)
```python
def nk_fitness(bits, tables):  # tables[i] shape (2**K,)
    N, K = len(bits), len(tables[0].shape)  # precompute tables randomly
    return sum(tables[i][tuple(bits[i:i+K])] for i in range(N)) / N
```

### needle / high-variance basin
```python
def needle(bits, x_star): return 1.0 if np.array_equal(bits, x_star) else 1e-4*np.random.rand()
```

### random control
```python
_table = np.random.rand(2**16)
def random_landscape(bits): return _table[hash(bits.tobytes()) % len(_table)]
```

D9.0 implication: All five generators fit into `tools/analyze_phase_d9_latent_genome_toy.py`; run each with N=200 trajectories; autocorrelation τ = −1/ln(ρ(1)) should rank: smooth >> multi-basin >> deceptive >> needle ≈ random. [direct empirical: Weinberger formula rendered from ar5iv 0709.4011]

---

## Direct empirical findings (papers that ran these protocols)

Jones & Forrest 1995 (ICGA): introduced FDC; sampled 4000 points; r_FDC > 0.15 easy, |r| < 0.15 hard, r_FDC < −0.15 deceptive. [unverified: thresholds from search snippets, primary PDF failed to render]

Sokal / Mantel 1967 via PMC3873175: 4999 permutations for Mantel; p < 0.001 recommended for partial Mantel variant to avoid false positives.

Quilodrán et al. 2025 (Mol Ecol Resources): benchmarked Mantel variants; at n=50–200, Type I inflation occurs when autocorrelation parameter k > 0.2; recommend random z sampling not structured grids. [direct empirical: PMC11696488]

Chicano, Verel et al. 2012: combined LON + elementary landscape autocorrelation over 600 QAP instances; landscape τ correlates with search algorithm performance. [unverified: full paper not rendered]

---

## Theoretical results

Weinberger: τ = −1/ln(ρ(1)); assumes exponential decay of autocorrelation. [direct empirical: ar5iv 0709.4011 rendered]

Lipschitz: if decoder D is L-Lipschitz (||D(z1)−D(z2)|| ≤ L||z1−z2||), then locality is formally bounded. L can be estimated empirically as max over sampled pairs of genome_dist(z1,z2)/latent_dist(z1,z2). No closed-form certificate exists for rule-based compilers; empirical sampling is the practical approach. [theoretical]

LSH (Indyk & Motwani 1998): a (r,cr,p,q)-sensitive hash family guarantees P(collision) ≥ p when dist ≤ r and ≤ q when dist > cr. Does NOT preserve locality globally; only within designed (r,cr) buckets. Applied to D9: LSH is a valid indexing structure for basin search but not a locality *guarantee*. [direct empirical: original LSH definition recovered from Gionis/Indyk/Motwani Columbia slides]

---

## Failure modes of these metrics (what gives false confidence?)

1. **Mantel under autocorrelated sampling**: spatial/grid z-samples inflate r_m. Use random z. [direct empirical: PMC11696488]
2. **FDC counterexamples**: Altenberg 1997 (SFI 97-05-037) shows FDC can misclassify; use FDC alongside Mantel, not alone. [unverified: paper found, PDF not rendered]
3. **τ assumes exponential decay**: multi-scale landscapes have τ undefined or misleading. Compute ρ at multiple lags (1,2,4,8) not just lag-1. [theoretical]
4. **Silhouette/ARI circular**: if basin labels derived from same decoder, metrics validate the labeling, not the geometry. [theoretical]
5. **Identity decoder pitfall**: a decoder that maps z bytes directly to genome bytes passes all locality tests trivially. Add a non-triviality check: genome must differ structurally from z (graph must be a valid non-trivial network, not a byte dump). [speculative but directly actionable for D9.0]
6. **LSH false locality**: bucket collisions are not global proximity; two z vectors in the same bucket can be from opposite ends of space. [theoretical]

---

## Background only

Full nc-eNCE graph grammar formalisms, subgraph isomorphism, graph autoencoders, and inverse mapping theory are out of scope for D9.0 per Gemini digest consensus. Stadler's full GP-map topology theory (neutral networks, phenotype space connectivity) is foundational reading but has no direct D9.0 implementable consequence until D9.1 introduces real network evaluation.

## Sources

- [Mantel test in population genetics - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3873175/)
- [Benchmarking the Mantel test and derived methods - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11696488/)
- [Mantel test - Wikipedia](https://en.wikipedia.org/wiki/Mantel_test)
- [Fitness Distance Correlation as a Measure of Problem Difficulty - SFI](https://ideas.repec.org/p/wop/safiwp/95-02-022.html)
- [FDC Counterexample - Altenberg 1997 SFI](https://sfi-edu.s3.amazonaws.com/sfi-edu/production/uploads/sfi-com/dev/uploads/filer/93/95/9395a61b-aff6-4b1b-adbd-8426281449aa/97-05-037.pdf)
- [Measuring the Evolvability Landscape - ar5iv 0709.4011](https://ar5iv.labs.arxiv.org/html/0709.4011)
- [Local Optima Networks, Landscape Autocorrelation - arXiv 1210.4021](https://arxiv.org/abs/1210.4021)
- [Genotype-Phenotype Maps - Stadler Biological Theory 2006](https://link.springer.com/article/10.1162/biot.2006.1.3.268)
- [silhouette_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [adjusted_rand_score - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
- [Gionis, Indyk, Motwani LSH paper - Columbia](https://www.cs.columbia.edu/~verma/classes/uml/ref/nn_lsh_gionis_indyk_motwani.pdf)
- [NK model - Wikipedia](https://en.wikipedia.org/wiki/NK_model)
- [Correlated and uncorrelated fitness landscapes - Weinberger, Springer](https://link.springer.com/article/10.1007/BF00202749)
- [Progressive gradient walk for neural network landscape - ACM GECCO 2018](https://dl.acm.org/doi/10.1145/3205651.3208247)
