# D9 Wave 3 — Agent G: Adversarial Red Team Review

Skeptic-mode review. Citations point to file:section. "No defense found" means the issue is fatal-as-stated and a hardening is required before D9.0 begins.

## 1. Attacks on architecture designs

### 1.1 PIC — vulnerabilities

**A1. Validity collapse at low z[0].** `K = round(z[0]·H²)` with `z[0] ∈ [0.01, 0.15]` (E §1.1). At `z[0] ≈ 0.01` and `H = 64`, K = round(0.01·4096) = 41 edges across 64 nodes — high probability of orphan nodes. `valid_network_rate` will fail at extremes. **Defense**: post-sample connectivity repair (add minimum spanning forest before density top-K), or clamp z[0] to ≥ 0.03 for small H. E §1.7 lists this as the cheapest gate but doesn't specify the repair.

**A2. Locality collapse via top-K crossings (the "anti-hash bound is average, not uniform").** E §1.5 claims `‖D(z)−D(z+ε)‖_edit = O(ε·H²)`. **This bound is wrong at score-tie clusters.** When many edges have nearly identical scores (e.g. far from any modular boundary, `z[9]≈0`, `z[7]≈0.5`), the score function `s(i,j;z) = w0 + w1·rec + w2·sym + w3·block` is near-constant across `H²` candidates. A perturbation of 1e-3 in `z[9]` flips the *entire ranking* among tied scores; top-K now selects an arbitrary new permutation determined by the sub-seed tie-breaker. Genome edit distance scales `Θ(K)`, not `O(εH²)`. The `O(εH²)` bound holds *on average*, not *uniformly*. **Defense**: add the score function `s` an explicit non-degeneracy term (e.g. `+ 1e-6 · idx_hash`) so ties are broken by a continuous function of (i,j) rather than seed; *then* the Lipschitz bound holds. E §1.5 hand-waves this — fix it before implementation.

**A3. Anti-hash arg is not closed-form for `z[7], z[8], z[9]`.** E §1.3 says "z[7],z[8],z[9] reshape the score function s; seed never enters s." But the score reshapes a *combinatorial* selection (top-K). The closed-form moment (`Σ edge − Σ edge`) cited in E §1.5 has expectation monotone in z[7] only *integrated over a uniform edge density*. At low density (z[0] = 0.01), this monotonicity is dominated by sampling noise. **Defense**: for low z[0], increase the locality test sample to detect noise floor; gate at low-density edge counts.

**A4. Identity-decoder pitfall under small-H reduction.** For `H ≤ 8`, the score function has only 56 ordered pairs, and 12 z-fields fully determine the top-K outcome. There is a many-to-one mapping but at small H it is *injective on most of z space*: knowing genome ⟹ knowing z[0..7] up to a finite hash. PIC at small H *is* a hash, just deterministically computed. **Defense**: D9.0 toy must use `H ≥ 32` minimum and explicitly fail the test at H=8 to demonstrate the hashing regime exists.

**A5. Inverse encoder soundness — moments don't determine the graph.** E §1.6 claims moment matching: `z_hat[0] = density(g)`. But two graphs with identical density, inhibition, and recurrence ratios can have entirely different edge sets. Recovery is *up to the moment*, not *up to the graph*. The claim "z_hat" recovers structural fields is true, but reconstruction quality (`mean Hamming dist` of F §2.9) will be very high for random external networks. **Defense**: rename `inverse encoder` → `structural-fingerprint encoder`; do not claim genome-level inverse for non-D9 networks. Already approximately stated; tighten the language.

### 1.2 MMD — vulnerabilities

**B1. Validity collapse from motif overlap.** E §2.7 acknowledges this. With `total_motif_budget = z[8]·H` and motif sizes 2–6, `n_motifs ≈ 0.95·H/3 ≈ 0.32·H`. The `pick_site` cursor advances by `motif_size(k)`, but at the boundary (last few cursor positions), motifs *must* overlap or fall off the end. **Worst case**: `z[8] = 0.95` with all-`fan_in` (size 4) on `H = 12` requires 3 disjoint placements of size 4 = 12 nodes, leaving zero room for glue or irregularity. **Defense**: explicit overlap-or-truncate policy in `pick_site`; document that high-budget MMD has H-floor.

**B2. Locality collapse at rounding boundaries.** E §2.5 admits "discontinuous only at integer rounding boundaries." For 8 motifs, every boundary surface is a hyperplane in z[0..7]. Volume of the ε-tube around all 8 hyperplanes is `8·ε·(boundary surface area)`. For `ε = 0.01` and softmax outputs near 1/8 each (the high-entropy region), `O(8·H/budget)` of motif boundaries are within ε. Locality fails on a non-measure-zero subset for typical experimental z-distributions. The "stochastic rounding seeded with hash(z[0:8])" mitigation in §2.5 is a *seed-only* mechanism — it reintroduces the GE failure pattern (Agent B Rothlauf-Oetzel 2006, line 51): same z, hash decides count → discontinuous mapping. **Defense**: replace stochastic rounding with deterministic *fractional* motif application — e.g., partial motifs scaled by fractional weight. **No defense found that is both deterministic AND continuous; pick one.**

**B3. Anti-hash arg presumes motif library is rich.** With only 8 motifs, the inverse subgraph isomorphism count vector is 8-dimensional. Post-softmax z[0:8] is a 7-simplex. If the motif library has a near-degeneracy (e.g., `chain` and `fan_out` count similarly on dense graphs), the inverse map z_hat[0:8] is rank-deficient. **Defense**: orthogonalize motif library — e.g., require pairwise-low subgraph-frequency correlation across the training distribution. Untested in E.

**B4. Identity-decoder pitfall via motif library lookup.** If `MOTIF_LIB` is small enough (say `|LIB| = 4`) and motif size ≥ H/4, the decoder reduces to "z picks one of 4 layouts" — pure lookup. **Defense**: enforce `|LIB| ≥ 8 AND max_motif_size ≤ H/16`.

**B5. Stronger threat — unstable count rounding under bootstrap.** F's killer microtest does N=200 bootstrap. Across bootstrap resamples, motif counts *change* on the rounding boundary, inducing variance in the genome distance matrix that propagates to r_m. The bootstrap CI in F §3 will be *wider for MMD than PIC*, biasing the comparison. **Defense**: bootstrap z, not bootstrap genome — re-decode each resample.

### 1.3 GPF — vulnerabilities

**C1. Field collapse at `z[0..3] ≈ 0`.** E §3.7 admits this is the main risk. Worse than admitted: when `z[0..3] ≈ 0` and `z[4]` (frequency) is also small, the field `f_z` is near-constant across all (i,j) pairs. Top-K selection becomes pure sub-seed tie-breaking — *exactly* the seed-hash failure mode. The Lipschitz bound `L = max(|d|, |d²|, 1)` is bounded, but when the score function is near-zero everywhere, *any* ε-perturbation flips the entire top-K ordering. The bound `O(ε·H²)` is meaningless when the constant in front is `Ω(K)`. **Defense**: enforce `‖z[0:4]‖ ≥ δ` for some δ; reject z values below threshold in a "validity" test. **Or**: add a fixed non-degeneracy term to f_z (e.g., +0.01·d).

**C2. Locality collapse from top-K crossings — same as PIC A2 but worse.** Lipschitz of the *score function* does not imply Lipschitz of the *top-K selector* unless score gaps are bounded below. GPF has no guarantee of score gaps. **No defense found** as stated; needs same hardening as PIC A2.

**C3. The `O(L·ε·H²)` bound is on average; worst-case is `Ω(H²)`.** Same issue as PIC. **Defense same as A2.**

**C4. Identity-decoder pitfall via 1D coordinates.** With `coords = i/H` (1D grid, E §3.1), the polynomial+sinusoid in `(i/H, j/H)` is highly redundant. Two graphs differ only in score *gaps*, not in the underlying field. A small library of `f_z` shapes (`z` in 6-D) generates a small variety of edge patterns, dominated by "chain", "long-range", "modular." **Defense**: use 2D coordinates (already noted as "trivially extends"); make this a hard requirement in D9.0.

**C5. Inverse encoder soundness — least-squares regression of edge indicators on a 6-parameter basis is severely under-fit.** A typical graph has H² edge-indicator variables; 6 parameters cannot reconstruct anything beyond very low-frequency structure. `z_hat` recovers *only* the low-rank field, not the actual graph. The roundtrip claim in F §2.9 ("approximate inverse reconstruction quality") will look great for D9-generated graphs (because the basis matches the generator) and look terrible for any other graph. **Defense**: rename to "field-fingerprint encoder"; never claim reconstruction.

## 2. Attacks on falsification plan

### 2.1 Killer microtest weaknesses

**D1. Hamming on raw bytes does not match genome distance for any of the three designs.** F §0 lines 44-49 use `D = (A != A[i]).sum(axis=1)` over `GENOME_LEN = 256` bytes. PIC's genome (E §1.4) is `{edges, polarity, thr, channel}` — no byte layout fixed; serializing to bytes via JSON or Serde gives an arbitrary order-dependent distance. MMD's edge order depends on motif placement order. GPF's edge order depends on score sort. **A hash decoder with matching byte layout could artificially inflate `r_hash`, OR a real decoder with permuted byte order could deflate `r_real`.** Both directions break the comparison. **Defense**: use *graph edit distance* on the edge multiset, not byte Hamming. Increases compute — accept the cost. **Or**: canonical byte serialization specified per design.

**D2. N=200, 499 perms — bootstrap CI non-overlap is *not* automatic.** With N=200, the standard error of a Pearson r is ~0.07. Two `r` values 0.10 apart are barely 1.4σ — bootstrap CIs likely overlap. The killer test as written may **fail to reject hash-like decoders that are 0.10 above hash baseline**. **Defense**: require non-overlapping CIs explicitly in killer test (currently only required for full suite per F §3). Or raise threshold to `r_real - r_hash > 0.20` even at N=200, with an honest "may produce false negative if real decoder is 0.10–0.20 above hash" disclaimer.

**D3. The 0.10 vs 0.20 gap split is unjustified.** F §0 picks 0.10 for microtest, 0.20 for full suite. F's stated rationale is "CIs are wider at N=200." But CI width scales as `1/√N`; going from N=200 to N=500 narrows CI by factor √2.5 = 1.58 — which means the right gap for microtest should be `0.20·1.58 = 0.32`, not `0.10`. **The microtest threshold is set so low it likely auto-passes hash-like decoders.** **Defense**: use `r_real - r_hash > 0.20` even at N=200, and accept lower power; OR raise N to 500 in microtest (still under 60s budget on most laptops).

### 2.2 Benchmark false-positive risks

**D4. Smooth basin: Lipschitz-trivial decoder (`D(z) = z[:GENOME_LEN]`) passes Mantel trivially.** F §1.1 expects `r_m ∈ [0.50, 0.85]` for real decoder. Identity decoder achieves `r_m = 1.0`. The "structural non-triviality check" in F §5-step-3 is mentioned but not specified. **Defense**: explicit non-triviality test must include `genome graph ≠ raw z bytes`, `decoded graph satisfies validity rules from NetworkDiskV1`, AND `entropy(genome) > entropy(z) + log2(H!)/8` (E §1.7 mentions this for PIC; promote to a global gate).

**D5. F has not actually verified hash-decoder pass rate is 0% on `trap_k`.** F §1.2 claims "Hash pass-rate: 0% on r_m, ~50% on FDC alone." This claim is **stated, not measured**. If hash decoder happens to produce structured noise that correlates with bit count in the trap basin, hash could pass. **Defense**: F must include a "controls-baseline calibration" run *before* gate evaluation — measure actual hash pass-rate on each landscape and adjust thresholds accordingly. Currently this calibration is absent from the test plan.

**D6. Multi-basin silhouette artifact.** F §6.9 acknowledges silhouette+ARI artifact on random embeddings. F §1.3 hardening requires "ARI ≥ 0.4 against an independent fitness oracle." But for D9.0 *toy*, the only oracle is the synthetic landscape itself. If basin labels come from K-means on the landscape over z, then ARI between K-means-on-genome and K-means-on-z-fitness is *not* independent — both depend on z. **Defense**: derive oracle labels from the *fitness function itself* (e.g., basin = sign of fitness gradient region in z space), not from clustering.

### 2.3 Threshold justification gaps

**D7. r_m > 0.3 from Quilodrán 2025 transferred from population genetics with no validity argument.** Population genetics has biological autocorrelation (geographic distance ↔ genetic distance) that does not reflect graph grammar locality. The 0.3 threshold is *folklore*, not theory. Agent D §Recipe 1 cites this but provides no transfer argument. **Defense**: calibrate threshold against random hash baseline on each landscape — make the threshold *relative* to hash, not absolute.

**D8. The 0.20 hash gap has no empirical evidence.** F §3 claims "0.20 is the empirical 'real beats hash' floor." No citation. Agent D §Recipe 3 also cites no evidence. **Defense**: must be calibrated against negative controls in the run itself; don't fix in advance.

**D9. Progressive scan ≥ 1.5× rate ratio — no precedent cited.** Agent D §Recipe 5 cites "supported by progressive gradient walk literature, ACM GECCO 2018." That paper is on neural network loss landscapes, not graph grammar genomes. Whether 1.5× is achievable is unknown. The bar may be set so low that *any* spatially-coherent decoder passes, or so high that nothing passes. **Defense**: empirically derive ratio threshold from random vs. trivial-decoder run on smooth basin — set threshold at e.g., 80th percentile of hash-decoder distribution.

### 2.4 Logical consistency of DNP gates

**D10. DNP_CONTROL_PARITY is logically broken.** F §3 says "≥ 1 of {hash, nonlocal, shuffled-fitness, shuffled-labels, shuffled-rules, random-cells} passes a gate that real decoder also passes." But `shuffled-fitness` is *evaluation-side*, not decoder-side — it perturbs the fitness vector, not the decoder. To "pass a gate" the gate must measure decoder output against fitness. With shuffled fitness, the same decoder produces the same genomes; only the fitness-correlation gates change. So real and shuffled are scored against *different* fitness vectors, and the gate is comparing apples to oranges. **Defense**: split DNP_CONTROL_PARITY into *decoder-side* (hash, nonlocal, shuffled-rules) and *evaluation-side* (shuffled-fitness, shuffled-labels, random-cells). Decoder-side parity is meaningful; evaluation-side parity is "did the metric break when the labels are random?"

**D11. F §5 step 4 claims `~3 min` for Mantel + FDC at N=500, 999 perms.** Mantel with 999 perms × N=500 = 999 × (500²/2) Pearson computations = ~1.25e8 operations per Mantel test. For two decoders (real + hash) = 2.5e8 ops. Pure Python overhead + scipy.stats.pearsonr per permutation likely takes 3–5× the 3 min budget. **Defense**: vectorize Mantel using NumPy matrix-flatten and `scipy.stats.pearsonr` once per permutation block; cite measured wall-time before declaring 30 min total.

## 3. Wave-1 claim spot-checks

### 3.1 Agent A literature claims

**Spot-check A.1 (Clune 2011 "as regularity decreases, indirect encoding underperforms"):** The Wave-1 file at line 25 *quotes verbatim* "As the regularity of the problem decreases, the performance of the generative representation degrades to, and then underperforms, the direct encoding." This is the strongest claim and appears to be a direct quote from the paper. **Verdict: SUPPORTED if the quote is accurate.** I cannot verify the quote without the paper PDF. **Risk**: paper PDF not fetched in this red team session. The downstream D9.0 implication (must include irregular basin test) is correctly drawn IF the quote is accurate.

**Spot-check A.2 (Grammar VAE finding 3 contradicts unconditional grammar-first claim):** Line 35 says "directly contradicts the Gemini digest's implicit framing." This is *interpretation*, not a direct paper claim. Grammar VAE shows constrained learned decoder works; it does not show static decoder *fails*. The contradiction Agent A claims is overstated. **Verdict: SOFT.** Grammar VAE is positive evidence for grammar structure, not evidence against static compilers.

### 3.2 Agent B literature claims

**Spot-check B.1 (GE locality failure reading):** Wave-1 B line 51 says "the large majority of neighboring genotypes do not map to neighboring phenotypes." This is correctly cited as a *failure* by the standard threshold (most neighbors don't map locally). However, the distinction matters: was GE *worse than random*, or just *worse than alternatives*? Agent B line 15 says "Performance comparison... shows GE representation leads to measurably lower search performance" — lower than alternatives, not lower than random. **Implication**: GE was *not as bad as random hash*, just bad. This means the anti-hash argument's binary framing ("hash vs. structured") is too coarse: there are *intermediate* failure modes where a deterministic compiler is locality-poor without being random. **Defense**: D9.0's gates should distinguish "passes hash control AND passes locality threshold" — i.e., real decoder must beat both the random hash floor *and* an absolute Mantel threshold. F §3 already does this; good.

### 3.3 Agent C codebase claims

**Spot-check C.1 (PanelMetrics 8-D fingerprint):** Wave-1 C line 71-82 lists 8 fields. I cannot directly verify without reading `examples/evolve_mutual_inhibition.rs`. **Risk**: if the schema varies by phase or by experiment, F §2.3 "z↔behavior Spearman ρ" using this fingerprint is fragile. The D8 audit mentioned in Wave-1 C line 67 might use a *different* 8-feature set. **Defense**: D9.0 toy must define its own fingerprint, not import PanelMetrics directly. F §2.3 already provides a "synthetic 4-D fingerprint" fallback for toy-only — good, use that.

**Spot-check C.2 (psi prediction Spearman 0.64):** Wave-1 C line 106 cites this number. Validation method: "seed-held-out CV." This is *better* than no CV but is not *cross-experiment* CV. The claim does not necessarily transfer to D9 networks. **No D9.0 implication** — psi is not used in D9.0 toy.

### 3.4 Agent D threshold claims

**Spot-check D.1 (r_m > 0.3 as established threshold):** Wave-1 D line 16 cites Quilodrán 2025 "n=50–200 range scaled up." This is admitted in Wave-1 to be *synthesized*, not directly cited. Wave-1 D §Recipe 1 step 5 cites this as "Recipe 1 pass bar," but the number itself is folklore. **Verdict: WEAK — threshold is convention, not theorem.**

**Spot-check D.2 ("minimum decoder validation test set" canonical-protocol claim):** Wave-1 D §Recipe 2 admits "no canonical single-paper protocol found" (line 24). The protocol IS invented by composition. **Verdict: HONEST — invented by Agent D, claimed as composition not as citation. Acceptable.**

## 4. Anti-pattern audit (spot-check)

**P1. Random opaque hash:** PIC, MMD, GPF all claim anti-hash arguments. PIC's argument is closed-form per field (strongest). MMD's argument fails near rounding boundaries. GPF's argument fails when field collapses. **Status: Partial defense; needs hardening per A2, B2, C1.**

**P2. Too-early neural decoder:** D9.0 hard-constrained no torch/tf/jax/sklearn.neural_network (F §3 DNP_TOO_HEAVY). **Status: Addressed.**

**P3. Atlas cell inverse mapping without generative model:** None of E's three designs depends on D8 atlas cell inversion. Agent C's "psi prediction Spearman 0.64" is not used in D9.0 toy. **Status: Avoided.**

**P4. Geometry claim without locality test:** PIC, MMD, GPF all *propose* locality; F's plan tests it. **Status: Addressed in plan; risk is in execution (see D5, D6).**

**P5. Formal theorem without implementable consequence:** PIC's `O(εH²)` Lipschitz bound is *implementable* (test with empirical Mantel) but *non-uniform* (see A2). MMD's "discontinuous only at integer boundaries" is *measurable* (test density of boundary crossings) but not *bounded* (see B2). GPF's `O(Lε·H²)` is the same problem. **Status: All three Lipschitz claims are average-case, not worst-case. Hardening required.**

## 5. Adversarial decoder construction (D_advr that passes microtest)

**Construction**: `D_advr(z) = bytes_of(quantize(z, 8 bits per dim) + structured_noise)` where `structured_noise = SHA256(quantize(z))[:GENOME_LEN]`.

This decoder:
1. **Passes validity** (every output is bytes-formatted; can wrap in a graph builder that always produces a valid graph).
2. **Passes locality on smooth basin** (Mantel r_m on Hamming-byte-distance is `Θ(1)` because nearby z → similar quantization → similar output bytes; the SHA256 noise contributes `O(1)` Hamming on average but the quantized prefix dominates).
3. **Passes hash gap test** (the structured-quantization signal exceeds the SHA256 noise in raw byte Hamming).
4. **FAILS the actual D9 goal**: produces *no graph structure beyond what was already in z*; behavior is determined entirely by noise (since SHA256 is the only other signal). On a real fitness landscape, `D_advr` is no better than hash decoder, but on the *killer microtest* it passes.

**Why does it pass?** Because F's killer microtest measures *byte-distance* correlation, not *graph-structure* correlation. The microtest does not include a non-triviality check (step 3 in F §5 is listed at "1.7 min cumulative" but not specified for the killer microtest at §0).

**Implication**: The killer microtest IS NOT SUFFICIENT. It must be augmented with a non-triviality assertion: `entropy(genome) > entropy(z) + log2(H!)/8` AND `decoded graph satisfies NetworkDiskV1 validity rules`. Currently the entropy assertion is in E §1.7 for PIC only — promote it to F §0 mandatory.

**Defense**: add to F §0 killer microtest:
```python
H_GENOME = compute_entropy(g_real)
H_Z = compute_entropy(zb)
assert H_GENOME > H_Z + np.log2(math.factorial(H)) / 8, "identity-decoder pitfall"
```

Without this hardening, **D_advr passes the microtest and the project fails to detect it**.

## 6. Top 5 prioritized failure modes (likelihood × severity)

1. **(L=H, S=H) Top-K crossings break uniform Lipschitz bound (A2/B2/C1/C2/C3).** All three designs claim `O(εH²)` locality; all three fail uniformly when score gaps are small. **Test**: run Mantel at multiple z-density slices; if r_m drops > 0.2 between dense and sparse slices, Lipschitz claim is broken.

2. **(L=H, S=H) Killer microtest fails to catch D_advr (Section 5).** Microtest measures byte-Hamming, missing structural triviality. **Test**: implement D_advr in microtest as a third decoder; confirm it passes microtest but fails entropy check.

3. **(L=M, S=H) Hamming on raw bytes ≠ genome distance (D1).** Bootstrap and CIs computed on misleading metric. **Test**: implement graph edit distance and re-run microtest; report difference.

4. **(L=M, S=M) DNP_CONTROL_PARITY logically broken (D10).** Real decoder cannot "beat" shuffled fitness; gate is undefined. **Test**: split into decoder-side and evaluation-side parity; redefine gate per side.

5. **(L=M, S=M) MMD count rounding boundaries break locality on bootstrap (B2/B5).** MMD CIs will be artificially wide; comparison with PIC will favor PIC unfairly. **Test**: bootstrap z (re-decode each resample), not bootstrap genome; compare MMD CI width to PIC.

## 7. The cheapest test that kills D9 fastest

**Test name**: `IDENTITY_AUGMENTED_KILLER`. Modify F §0 microtest in three ways:
1. Add D_advr as third decoder (Section 5).
2. Replace byte Hamming with graph edit distance (D1).
3. Add entropy non-triviality assertion (Section 5 defense).

Wall: ~90 s (5 s per added decoder, ~30 s for graph edit distance over N=200).

If this fails on any of the three designs, D9.0 is dead; if it passes, Wave-1/2 architecture survives the toughest attack.

## 8. Verdict

- **Should D9.0 toy be implemented? YES, WITH-MODIFICATIONS.** The architecture is sound in principle; the falsification plan has implementable gaps that *must* be patched before code starts.

**Required hardenings before D9.0 starts:**

1. **(A2/B2/C1/C2)** All three designs must add a non-degeneracy term to their score/field/count function so worst-case Lipschitz bound holds. Score gaps must be bounded below. **Owner: design re-spec before any code.**
2. **(D1)** Replace byte Hamming with graph edit distance in *all* genome-distance metrics, not just the killer test. **Owner: F.**
3. **(Section 5)** Augment killer microtest with: third decoder D_advr + entropy non-triviality assertion + structural validity check. Promote E §1.7 entropy gate to F §0. **Owner: F.**
4. **(D2/D3)** Set killer microtest gap threshold to `r_real - r_hash ≥ 0.20` (same as full suite); accept lower power; OR raise N to 500. The current `0.10` lets D_advr-class decoders through. **Owner: F.**
5. **(D5/D8)** All hash-comparison thresholds must be calibrated against actual hash-decoder runs in the same study, not pre-fixed. **Owner: F.**
6. **(D10)** Split DNP_CONTROL_PARITY into decoder-side and evaluation-side gates with separate definitions. **Owner: F.**
7. **(B5)** Bootstrap re-decodes z, does not bootstrap genome. **Owner: F.**
8. **(D11)** Cite measured wall-time for full Mantel suite before claiming 30-min budget. If > 30 min, cut scope or vectorize. **Owner: F.**

**Single most important change**: hardening #3 (augmented killer test). Without it, the "60 s gate" is a smoke screen and the project may declare D9.0 success on a decoder that has no graph structure beyond byte echo of z.
