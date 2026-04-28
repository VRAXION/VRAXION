# D9 Wave 2 — Agent F: Falsification & Benchmark Plan

## 0. Killer microtest (cheapest single kill, ≤60 s)

**Goal**: in under 60 seconds, reject any decoder that is hash-like on locality. This is the single most cost-effective kill: locality is the *only* thing distinguishing D9 from a random byte oracle, so if we cannot beat hash on z↔genome Mantel, nothing else matters.

**Recipe** (executable, ~30–45 s wall):
```python
# killer_microtest.py
import numpy as np, hashlib, time
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

N = 200                              # smaller than Recipe 1's 500 — fits time budget
Z_DIM = 32                           # latent dim (placeholder; match D(z) input)
GENOME_LEN = 256                     # bytes
RNG = np.random.default_rng(20260428)

def hash_decoder(z_bytes, L=GENOME_LEN):
    h = hashlib.sha256(z_bytes).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8],'big'))
    return rng.integers(0,256,size=L,dtype=np.uint8).tobytes()

def real_decoder(z_bytes, L=GENOME_LEN):  # placeholder for D9.0 compiler
    raise NotImplementedError

def mantel_r(D1, D2, perms=499):       # 499 perms keeps under 60s budget
    iu = np.triu_indices_from(D1, k=1)
    r, _ = pearsonr(D1[iu], D2[iu])
    null = np.empty(perms)
    n = D1.shape[0]
    for p in range(perms):
        idx = RNG.permutation(n)
        D2p = D2[np.ix_(idx, idx)]
        null[p], _ = pearsonr(D1[iu], D2p[iu])
    p_val = (np.sum(np.abs(null) >= abs(r)) + 1) / (perms + 1)
    return r, p_val

# random (NOT grid — Quilodrán autocorrelation warning)
Z = RNG.standard_normal((N, Z_DIM)).astype(np.float32)
zb = [z.tobytes() for z in Z]
D_z = pdist(Z); from scipy.spatial.distance import squareform; D_z = squareform(D_z)

def hamming_dmat(genomes):
    A = np.frombuffer(b''.join(genomes), dtype=np.uint8).reshape(N, GENOME_LEN)
    D = np.zeros((N,N))
    for i in range(N):
        D[i] = (A != A[i]).sum(axis=1)
    return D

g_real = [real_decoder(z) for z in zb]
g_hash = [hash_decoder(z) for z in zb]
D_real = hamming_dmat(g_real); D_hash = hamming_dmat(g_hash)

t0 = time.time()
r_real, p_real = mantel_r(D_z, D_real)
r_hash, p_hash = mantel_r(D_z, D_hash)
print(f"r_real={r_real:.3f} p={p_real:.3f}  r_hash={r_hash:.3f} p={p_hash:.3f}  t={time.time()-t0:.1f}s")
```

**Kill thresholds** (justified):
- `r_real < 0.30` OR `p_real ≥ 0.05`  → emit `D9_DECODER_NO_LOCALITY` (Recipe 1 pass bar).
- `r_real < r_hash + 0.10`  → emit `D9_DECODER_HASHLIKE_BEHAVIOR` (Recipe 3 hardened: 0.10 chosen lower than the 0.20 full-suite bar because microtest uses N=200 and 499 perms, so CIs are wider; the full suite re-tests at 0.20).
- `r_real > 0.95` AND identity-check (genome-bytes ≠ z-bytes hash) fails → emit `D9_DECODER_VALIDITY_FAIL` (Agent D identity-decoder pitfall).

**Time budget**: N=200, 499 perms, two decoders → ~20–45 s on a laptop. If wall > 60 s, drop to N=150.

**Why this is the killer**: it touches every false-positive risk in one shot — hash baseline, autocorrelation (random z), identity triviality, Mantel effect-size floor. A decoder that survives this is *not* obviously hash; everything after is corroboration.

---

## 1. Benchmark suite

For all five: N=500 z samples (Recipe 1 / 2), 999 Mantel permutations, random (non-grid) z, both real decoder and `NCI_RANDOM_HASH_DECODER` always run side-by-side.

### 1.1 Smooth basin
- **Landscape**: `smooth(x, x*) = -‖x − x*‖²` (Agent D §Toy generators).
- **N**: 500 z; 200 trajectories for τ.
- **Expected ranges**: real decoder `r_m(z,genome) ∈ [0.50, 0.85]`, `r_FDC ∈ [0.40, 0.90]` (smooth → near-monotonic).
- **Pass**: `r_m ≥ 0.40` AND `r_FDC ≥ 0.30` AND `r_real ≥ r_hash + 0.20`.
- **Hash pass-rate**: ~0% (hash gives r_m ≈ 0). **False-positive risk: LOW.**
- **Hardening**: still require non-overlapping bootstrap 95% CIs (1000 resamples) — a hash decoder seeded with structured noise can occasionally cross the 0.30 floor.

### 1.2 Deceptive basin
- **Landscape**: `trap_k(bits, k=4)` (Whitley/Deb-Goldberg trap). Optimum all-zeros; gradient misleads to all-ones.
- **N**: 500 z; 1000 networks for validity.
- **Expected**: `r_m(z,genome) ∈ [0.30, 0.70]` (locality unaffected by deception), `r_FDC ∈ [-0.30, +0.10]` (deceptive landscape — Jones-Forrest negative regime).
- **Pass**: `r_m ≥ 0.30`. **`r_FDC` is a diagnostic, not a gate**, per Altenberg counterexample.
- **Hash pass-rate**: 0% on r_m, ~50% on FDC alone (random sign). **False-positive risk: MEDIUM if FDC standalone — REJECT FDC standalone.**
- **Hardening**: gate on Mantel; report FDC only as a deception-marker, never as a decoder-quality marker.

### 1.3 Multi-basin (NK)
- **Landscape**: Agent D's `nk_fitness`, K=3, N_bits=64, 16 random tables.
- **N**: 500 z; 300 evals for progressive scan (Recipe 5).
- **Expected**: `r_m(z,genome) ∈ [0.30, 0.70]`, basin silhouette `> 0.4` after K-means with K=8, ARI ≥ 0.4 vs. NK basin labels from independent oracle (NOT decoder labels — circularity warning).
- **Pass**: `r_m ≥ 0.30` AND `silhouette ≥ 0.4` AND `progressive/random ≥ 1.5×` (Recipe 5).
- **Hash pass-rate**: <5% on r_m, 0–10% on silhouette (HDBSCAN can find spurious clusters in random embeddings — known artifact). **False-positive risk: MEDIUM on silhouette.**
- **Hardening**: require BOTH silhouette ≥ 0.4 AND ARI ≥ 0.4 against an independent fitness oracle (not decoder-derived labels).

### 1.4 Needle / high-variance
- **Landscape**: Agent D's `needle(bits, x*) = 1 if bits==x* else 1e-4·rand()`.
- **N**: 500 z; 1000 evals for hit-rate.
- **Expected**: `r_m` near random (no exploitable gradient), target-hit-rate uniformly low.
- **Pass**: real decoder `target_hit_rate ≥ 2× hash` AND `progressive/random ≥ 1.5×` after 1000 evals.
- **Hash pass-rate**: by construction, hash baseline is the reference — they should match. **False-positive risk: HIGH that *both* fail and we declare D9 pass on a tie.**
- **Hardening**: needle benchmark is **diagnostic only**, not a pass gate. Used to confirm "decoder doesn't help when no signal exists" (i.e., decoder is not making up signal). Result `real ≈ hash` is *expected and acceptable* on needle.

### 1.5 Random control
- **Landscape**: Agent D's `random_landscape(bits) = table[hash(bits) mod L]`.
- **N**: 500 z.
- **Expected**: `r_m(z,fitness)` near 0 for both real and hash. Real decoder MUST NOT exceed hash by > 0.05.
- **Pass**: `|r_real − r_hash| < 0.10` (this is the **DNP_CONTROL_PARITY** anti-test: if real decoder appears to find structure in pure random, it is leaking metadata or overfitting the seed RNG).
- **Hash pass-rate**: 100% (this is the floor). **False-positive risk: this benchmark catches them, doesn't generate them.**

---

## 2. Metric definitions (formula-level)

1. **valid_network_rate** = (# decoded networks passing all of: connected, no zero-degree nodes, no self-loops [unless allowed], edge count ∈ [1, n²/2], polarity ∈ {−1,+1}, threshold ∈ [0, threshold_max]) / N_total. Source: Agent C `NetworkDiskV1` validation rules + Recipe 2.
2. **z↔genome Spearman ρ + Mantel**: ρ = Spearman of upper-triangle pairs (`D_z[iu]`, `D_g[iu]`); p-value from 999-permutation Mantel; report ρ as effect size.
3. **z↔behavior Spearman ρ**: same protocol with `D_b` = Euclidean distance in PanelMetrics 8-D space (Agent C `panel_timeseries.csv` columns: stable_rank, kernel_rank, separation_sp, collision_rate, f_active, unique_predictions, edges, accept_rate_window). For toy-only D9.0 without networks evaluated, replace with synthetic 4-D fingerprint (mean genome byte, std, byte-bigram entropy, edge density proxy).
4. **FDC** = Pearson r between `f(z_i)` and `d(z_i, z*)` where `z*` = best-fitness sample. Report alongside Mantel — never standalone (Altenberg).
5. **Basin clustering**: silhouette = `sklearn.metrics.silhouette_score(B, labels)`; ARI = `adjusted_rand_score(true_labels, pred_labels)` where `true_labels` come from independent fitness-oracle clustering (not decoder).
6. **Target cell hit rate** = (# decoded networks landing in named D8 cell or toy target tile) / N_evals.
7. **Progressive scan efficiency** = `unique_basins_progressive(t) / unique_basins_random(t)` after t=300 evals; gate at ≥ 1.5×.
8. **Exact roundtrip rate** = (# z where stored-z(g) == z) / N for D9-generated genomes; should be 1.0 (Gemini "exact inverse for own genomes only").
9. **Approximate inverse reconstruction quality** = mean Hamming distance between target genome and best `D(z̃)` over 1000 random z̃; reported, not gated.

---

## 3. DNP gate thresholds

| Gate | Threshold | Justification |
|------|-----------|---------------|
| `DNP_VALIDITY_COLLAPSE` | `valid_network_rate < 0.99` | Recipe 2 §1; Gemini digest line 158 |
| `DNP_LOCALITY_COLLAPSE` | `r_m(z, genome) ≤ r_hash + 0.20` (ε=0.20) with non-overlapping 95% bootstrap CIs (1000 resamples) | Recipe 3 §4: 0.20 is the empirical "real beats hash" floor; CI requirement guards against single-sample lucky permutations |
| `DNP_BEHAVIOR_HASHLIKE` | `r_m(z, behavior) ≤ r_hash + 0.10` (ε=0.10, looser than genome) | Recipe 2 §3 says r_m > 0.15 to pass; 0.10 above hash is the minimum nontrivial gap given behavior is downstream and noisier |
| `DNP_SCAN_NO_GAIN` | `progressive_scan_rate / random_scan_rate < 1.5` after 300 evals | Recipe 5 §4 |
| `DNP_CONTROL_PARITY` | ≥ 1 of {hash, nonlocal, shuffled-fitness, shuffled-labels, shuffled-rules, random-cells} passes a gate that real decoder also passes | Recipe 2 §5; Gemini line 168 |
| `DNP_TOO_HEAVY` | D9.0 implementation imports any of: torch, tensorflow, jax, sklearn.neural_network, transformers; or runtime > 30 min on the full toy suite | Gemini line 172 + Wave-1 budget |

---

## 4. Negative controls (≤10 lines each)

```python
# NCI_RANDOM_HASH_DECODER (Agent D Recipe 3, ready)
def hash_decoder(z, L=256):
    h = hashlib.sha256(z).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8],'big'))
    return rng.integers(0,256,size=L,dtype=np.uint8).tobytes()

# NCI_NONLOCAL_DECODER — same output for any permutation of z bytes
def nonlocal_decoder(z, L=256, real=real_decoder):
    return real(bytes(sorted(z)), L)   # canonical-sort destroys position info

# NCO_SHUFFLED_FITNESS
def shuffled_fitness(fitness_array, rng=RNG):
    return rng.permutation(fitness_array)

# NCO_SHUFFLED_BEHAVIOR_LABELS
def shuffled_behavior(behavior_matrix, rng=RNG):
    idx = rng.permutation(len(behavior_matrix)); return behavior_matrix[idx]

# NCO_RANDOM_CELL_ASSIGNMENT
def random_cells(N, n_cells=156, rng=RNG):
    return rng.integers(0, n_cells, size=N)

# NCO_RULE_LABEL_SHUFFLE — applies real grammar but with rule names permuted
def shuffled_rules(z, rule_table, rng=RNG, real=real_decoder):
    perm = rng.permutation(len(rule_table))
    shuffled_table = [rule_table[i] for i in perm]
    return real(z, rules=shuffled_table)
```

Acceptance: real decoder must beat ALL six on the gate it claims to pass.

---

## 5. Test execution order with time budget

| # | Test | Time | Cumulative | Verdict on fail |
|---|------|------|-----------|-----------------|
| 1 | **Killer microtest** (§0) | ≤60 s | 1 min | `D9_DECODER_NO_LOCALITY` / `D9_DECODER_HASHLIKE_BEHAVIOR` |
| 2 | Validity rate, 1000 networks | ~30 s | 1.5 min | `D9_DECODER_VALIDITY_FAIL` |
| 3 | Identity-decoder check (genome ≠ trivial-hash-of-z, structural non-triviality) | ~10 s | 1.7 min | `D9_DECODER_VALIDITY_FAIL` |
| 4 | Smooth + deceptive Mantel + FDC, N=500, 999 perms | ~3 min | 4.7 min | `D9_DECODER_NO_LOCALITY` |
| 5 | Multi-basin Mantel + silhouette + ARI | ~3 min | 7.7 min | `D9_DECODER_NO_LOCALITY` |
| 6 | Needle diagnostic + random control parity | ~3 min | 10.7 min | `D9_CONTROL_PARITY_FAIL` |
| 7 | Progressive scan (300 evals × 2 strategies × 5 landscapes) | ~6 min | 16.7 min | `D9_TILE_SCAN_NO_SIGNAL` |
| 8 | Six negative-control re-runs of step 4 + step 7 gates | ~10 min | 26.7 min | `D9_CONTROL_PARITY_FAIL` |
| 9 | Roundtrip + approximate inverse | ~2 min | 28.7 min | report-only |

**Total budget: <30 min** for full toy verdict. Fail-fast: any DNP gate triggered in steps 1–3 short-circuits to verdict immediately.

---

## 6. Failure-mode catalog (false-positive risks)

1. **High validity but no locality** → caught by step 4 (Mantel). Validity alone is *not* sufficient.
2. **High locality on smooth only, collapses on deceptive** → caught by §1.2 requiring `r_m ≥ 0.30` *also* on `trap_k`. Smooth-only locality is suspicious — likely a Lipschitz-trivial encoder.
3. **High Mantel r_m but FDC counterexample** → expected on deceptive; FDC reported as diagnostic, not gate. Altenberg-aware.
4. **Identity decoder triviality** (`D(z) = z`) → caught by step 3 structural non-triviality check + Agent D's identity-decoder warning. Genome must NOT be a hash of z, AND must encode a valid graph (edges/polarity/threshold), AND `genome_bytes ≠ z_bytes`.
5. **Progressive scan helped by autocorrelation, not real basin structure** → caught by random-z sampling in step 1, and by random-control benchmark §1.5 (if real beats hash on pure random, the "structure" is artifactual).
6. **Behavior labels biased by evaluator** (LSI Mario lesson) → cited Agent B; mitigation: ARI must use independent oracle, never decoder-derived labels (Recipe 4 §5 circularity warning).
7. **Mantel autocorrelation Type-I inflation** → mitigation: random z, not grid (Quilodrán/PMC11696488 in Agent D).
8. **Hash decoder lucky on small N** → mitigation: 1000-bootstrap 95% CIs required to be non-overlapping, not just point estimate gap.
9. **Silhouette artifact on random embeddings** → mitigation: silhouette + ARI (both required), independent oracle.
10. **Decoder leaks evaluation metadata** (e.g., uses fitness during decode) → mitigation: random control §1.5; if real beats hash on pure-random landscape, fail `DNP_CONTROL_PARITY`.
11. **Lucky permutation in 999-Mantel** → p < 0.05 alone insufficient; gate is *effect size* r_m ≥ 0.30 AND p < 0.05, both required.
12. **Behavior fingerprint coupled to z by construction** (e.g., D9.0 toy fingerprint = byte-mean of z) → mitigation: fingerprint must be derived from *evaluated* genome, not from z directly; verify via shuffled-behavior control NCO_SHUFFLED_BEHAVIOR_LABELS.

---

## 7. Compared to Gemini digest

**Aligned**: verdicts (`D9_LATENT_DECODER_TOY_PASS`, etc.) and control codes verbatim; DNP gates from Gemini lines 156–172; Python-first toy scope; locality + progressive scan + controls as the three load-bearing axes.

**Tightened beyond Gemini** (because Gemini said "reasonable values"):
- All thresholds now numeric with citations: r_m ≥ 0.30 (Recipe 1), ε=0.20 over hash (Recipe 3), silhouette ≥ 0.4 + ARI ≥ 0.4 (Recipe 4), progressive ≥ 1.5× (Recipe 5), validity ≥ 0.99 (Recipe 2).
- Bootstrap 95% CI non-overlap added to all hash-comparison gates — Gemini didn't specify CI requirement.
- Identity-decoder pitfall explicit (Agent D §Failure modes 5) — Gemini missed this.
- FDC demoted from gate to diagnostic (Altenberg) — Gemini implied it as a gate.
- Needle benchmark explicitly diagnostic-only — Gemini implied it as pass/fail; `real ≈ hash` on needle is *correct behavior*.
- Random control benchmark added as anti-leak detector (`|r_real − r_hash| < 0.10` on pure random). Gemini didn't specify a positive parity test.

**Deferred from Gemini** (per Wave-1 consensus): graph grammar formalisms, neural decoder, inverse mapping, live Rust integration. D9.0 is offline Python toy only.

**Killer microtest is the strongest single addition**: a 60-second test that touches every false-positive risk simultaneously. Gemini didn't propose a fail-fast.

### Critical Files for Implementation
- S:\Git\VRAXION\tools\analyze_phase_d9_latent_genome_toy.py (NEW; canonical D9.0 entry point per Gemini digest §Implementation Implications)
- S:\Git\VRAXION\docs\research\inbox\d9_claude_wave1_D_locality_benchmarks.md (Recipes 1, 3, 5; landscape generators; failure modes)
- S:\Git\VRAXION\docs\research\inbox\d9_gemini_genome_compiler_design.md (verdict names, control codes, DNP gate structure)
- S:\Git\VRAXION\docs\research\inbox\d9_claude_wave1_C_vraxion_context.md (PanelMetrics 8-D fingerprint columns; InitConfig structural-validity rules)
- S:\Git\VRAXION\instnct-core\src\network\disk.rs (NetworkDiskV1 validation rules — used to ground `valid_network_rate` checks at line 35)
