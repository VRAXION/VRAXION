# Research Digest: Fixed Sinusoidal Encoding for Byte-Level Models

**Date:** 2026-02-26
**Context:** Replacing `nn.Embedding(256, 8192)` with fixed sinusoidal vectors in INSTNCT v4
**Sources:** Kimi k2 Scholar CSVs + Grok deep research + Qwen deep research (PDF) + Claude deep research + Gemini deep research + deep-research agents on 6 key papers

---

## Executive Summary

All 5 research agents agree: **fixed zero-parameter input encoding is viable for byte-level models** and may reduce overfitting. Weight tying is a free regularizer. The novel combination (fixed encoding + byte-level + ring-buffer architecture) has no direct prior art.

**CRITICAL DISAGREEMENT (Claude Research vs others):** Sinusoidal encoding has a fundamental similarity problem for bytes. Adjacent byte values (97='a', 98='b') have ~0.97 cosine similarity — nearly identical vectors. Bytes are CATEGORICAL (not ordinal), so this ordinal bias is harmful. **Hadamard/random orthogonal encodings** provide perfect separation (cosine sim = 0 between all pairs) and are strictly superior. See "Claude Deep Research" section below.

**Updated consensus:** Fixed encoding YES, but **Hadamard > sin/cos** for byte-level tokens.

---

## Paper-by-Paper Findings

### 1. HTP — Harmonic Token Projection (Schmitz, 2025)

**arxiv:** 2511.20665
**What:** Vocabulary-free, training-free, deterministic, reversible embedding via coprime moduli on the unit circle.

**Core formula:**
```
r_i = N_t mod m_i                    # residue under coprime modulus m_i
E_i = [sin(2π r_i / m_i), cos(2π r_i / m_i)]
E(t) = [E_1, E_2, ..., E_k] ∈ ℝ^(2k)
```

Where `m_i` are pairwise coprime integers (primes > 256 work).

**Key insight for us:** Since byte values b ∈ [0,255] and all moduli m_i > 256, we get `b mod m_i = b` always. The formula simplifies to:
```
E_i(b) = [sin(2π b / m_i), cos(2π b / m_i)]
```
This is multi-frequency sinusoidal encoding with coprime integer frequencies.

**Results:** Beats Word2Vec, GloVe, and BERT-base on STS-B **with zero training**. Sentence-BERT still wins.

**Dimensionality ablation:** Plateau at dim ~512. For dim=8192 we'd use k=4096 coprime moduli.

**Limitations:** No polysemy handling (irrelevant for bytes), no published code, coprime moduli selection not specified.

---

### 2. FoNE — Fourier Number Embedding (Zhou et al., 2025)

**arxiv:** 2502.09741
**What:** Maps numbers into embeddings via Fourier features. Directly relevant — treats tokens as numeric values.

**Key question it asks:** "Are sinusoidal encodings truly sufficient?" — validates that Fourier-based encoding can capture numeric relationships.

*Note: Full paper details blocked by paywall. Abstract confirms approach works for number tokens.*

---

### 3. Press & Wolf — Weight Tying (2017, 898 citations)

**Paper:** "Using the Output Embedding to Improve Language Models" (EACL 2017)
**This is THE weight tying reference.**

**Method:** One line of code:
```python
self.decoder.weight = self.encoder.weight  # THE ONE LINE
```

**Hard constraint:** `embed_dim == hidden_dim`. Our case: both are 8192 → compatible.

**Results (PTB):**

| Model | Params | Test PPL |
|-------|--------|----------|
| Baseline (large) | 66M | 78.4 |
| + Weight Tying | 51M | 74.3 |
| + WT + Proj Reg | 51M | 73.2 |
| RHN + WT | 24M | 66.0 |

**On overfitting (direct quote):**
> "Weight tying significantly reduces perplexity on both the validation set and the test set, but not on the training set. This indicates less overfitting, as expected due to the reduction in the number of parameters."

**Critical nuance for us (vocab=256):**
- At vocab=256, embedding is only ~2M params. Parameter saving is negligible.
- BUT the gradient dynamics argument still holds: output gradients flow to ALL 256 embeddings every step, enriching rare bytes.
- It's a **free regularizer** — zero cost to implement, mild but consistent benefit.

**When it HURTS:** word2vec skip-gram (token repetition probability is high). Not an issue for our use case.

---

### 4. Pseudo-Inverse Tying — PIT (Gu et al., 2026!)

**arxiv:** 2602.04556 — Very fresh (Feb 2026)

**Problem with standard WT:** embedding and unembedding can drift apart during training.

**Solution:** Shared orthonormal memory M + learned SPD transform T (Cholesky-parameterized):
```
logits = M @ T @ hidden         # unembedding
embed  = T^-1 @ M^T @ one_hot  # embedding (via triangular solve)
```

**For our case (V=256, D=8192):**
- M is (256, 8192) — rectangular, orthonormal rows
- T is (8192, 8192) — 33.5M params from Cholesky factor
- T is too expensive for us (33.5M params > entire model)

**Verdict:** Interesting theoretically, but overkill for V=256. Standard WT is sufficient.

---

### 5. Byte Latent Transformer — BLT (Pagnoni et al., 2025, 123 citations)

**Paper:** ACL 2025 Outstanding Paper. State-of-art byte-level model.

**How BLT handles bytes:**
1. **Learned byte embeddings** — standard `nn.Embedding(256, h)` (0 FLOPs, tiny table)
2. **Hash n-gram enrichment** — THE key trick:
   ```
   e_i = x_i + Σ_{n=3..8} E_hash(Hash(g_{i,n}))
   ```
   - 6 n-gram tables (3-gram through 8-gram), 400K entries each
   - Rolling polynomial hash maps byte n-grams to table slots
   - Additive enrichment before any transformer layer

**Results:** BLT 8B matches/beats Llama 3 8B. **+8 points on noisy HellaSwag** (robustness win).

**Key insights for us:**
- "The 256-vocab bottleneck is solved by context, not vocab size"
- "You don't need many encoder layers if embeddings are rich"
- BLT uses LEARNED byte embeddings (not fixed) but enriches them with hashed context
- The n-gram trick is orthogonal to our sin/cos idea — could be combined later

---

### 6. Headless LM / Contrastive Weight Tying — CWT (Godey et al., 2023)

**arxiv:** 2309.08351

**What:** Remove the output projection entirely during pretraining. Train with contrastive loss instead:
- Positive: output hidden state close to correct token's input embedding
- Negatives: all other tokens in the batch

**Results:**
- +0.86 GLUE at equal compute, +1.65 at 2× budget
- 25% latency reduction per batch (no V×D projection)
- 20× more data-efficient

**For byte-level (V=256):**
- The efficiency win mostly disappears — V=256 projection is already cheap
- Standard weight tying is simpler and almost as effective
- CWT adds complexity (contrastive loss, fine-tuning phase for generation) without proportional benefit at small vocab

**Verdict:** Not worth it for V=256. Standard WT + fixed sin/cos is the better path.

---

## Additional Papers from Scholar CSVs

### Positional Encoding Helps RNNs (Morita, 2024, 9 cit.)
- Sinusoidal positional encoding helps RNNs handle large vocabulary
- Stabilizes learning for infrequent tokens
- **Relevant:** Our INSTNCT is RNN-like (recurrent over timesteps)

### Rethinking Embedding Coupling (Chung et al., 2020, 194 cit.)
- Shows **decoupled** embeddings can outperform tied embeddings in pre-trained LMs
- Counterpoint to Press & Wolf — suggests tying isn't always optimal
- **Takeaway:** Test both tied and decoupled in our ablation

### "By Tying You Assume Distributional Hypothesis" (Bertolotti & Cazzola, 2024, 5 cit.)
- Theoretical paper: weight tying implicitly assumes input/output distributions match
- For byte-level: input distribution (byte frequencies) ≈ output distribution (predicted bytes) → assumption holds

### Super Tiny Language Models (Hillier et al., 2024, 17 cit.)
- Tokenizer-free + weight tying for tiny models
- Confirms weight tying works well at small scale
- **Directly relevant** to our small-model + byte-level setup

### Memorization Without Overfitting (Tirumala et al., 2022, 430 cit.)
- Larger models memorize more before overfitting
- Fewer params = overfitting sooner
- **Supports our hypothesis:** reducing I/O params (fixed sin/cos) should reduce overfitting

### Anchor & Transform (Liang et al., 2020, 16 cit.)
- Sparse embeddings from anchors + transform
- Related concept: don't learn full embeddings, derive from a smaller set
- **Supports:** fixed base + learned transform approach

---

## Qwen Deep Research (24-page PDF, 242 references)

**Source:** `research/qwen_sincos_research.pdf`
**Title:** "Beyond Learned Embeddings: Evaluating Sinusoidal Encodings for Parameter-Efficient Byte-Level Recurrent Networks"

Qwen produced a comprehensive literature synthesis specifically targeting our exact use case. Key findings:

### 1. Final Performance — "Highly Probable" to Maintain Parity

- Fixed sinusoidal positional encoding in Transformers proves the principle: deterministic frequency-based signals CAN carry meaningful structural information without being learned [ref 18, 233]
- **Direct RNN evidence (Morita 2024):** LSTM with sinusoidal encoding successfully reversed sequences of length 64 with vocab 16,384, achieving >95% token accuracy. Vanilla LSTM failed as vocab grew [ref 199]
- For byte-level (vocab=256): the task is **disambiguation** (unique vector per byte), not semantic encoding. Sin/cos excels at this.
- The "autonomous nature" of fixed encodings (invariant to input values) helps the network discover general patterns rather than memorizing specific instances [ref 199]
- **Expressiveness loss is minimal** for 256 tokens in 8192 dimensions — vectors are distinct and decorrelated enough

### 2. Overfitting — ~79% Param Reduction Acts as Powerful Regularizer

- Removing 4.2M of 5.3M params = "classic regularization effect" [ref 207, 229]
- Fewer params → diminished capacity to memorize training data → forced to learn general patterns
- Effect analogous to dropout or weight decay — limits degrees of freedom
- **Risk:** If fixed encodings are TOO simplistic → underfitting. But for 256-token byte vocab this is unlikely.
- "Removing the embedding layer makes the optimization landscape smoother" [ref 187]
- **Key nuance:** Learned embeddings add gradient stochasticity that can help escape sharp minima. Removing this makes optimization smoother but potentially less exploratory.

### 3. Convergence Speed — "Most Confident Prediction"

- **ASR study finding:** Fixed sinusoidal encodings → **up to 8× faster training iteration rates** on single GPU vs learned embeddings [ref 198]
- Mechanism: stable, pre-computed signal → remaining params converge faster without co-adapting with changing input
- "The benefits stem from the static nature of the signal itself, not the subsequent processing mechanism" — directly applicable to our tanh-activated hidden state [ref 198]
- Simpler optimization problem → less noisy loss curves, potentially lower LR needed
- **This is the strongest prediction:** faster convergence is nearly certain.

### 4. Comparison Table (from Qwen's analysis)

| Strategy | I/O Params | Key Advantage | Key Risk |
|----------|-----------|---------------|----------|
| Baseline (learned) | 4.2M | Max flexibility | ~79% of model; slower convergence |
| Fixed sin/cos input | 2.1M | Param reduction; faster convergence; stable signal | Expressiveness loss (minimal for V=256) |
| Fixed + weight tied | 0M | Maximum param reduction | Highest expressiveness risk; relies on fixed table for both I/O |

### 5. Qwen's Strategic Recommendations

1. **Test 4 configurations side-by-side:**
   - Baseline (current learned embeddings)
   - Fixed sin/cos input only (learned output)
   - One-hot + linear projection (control)
   - Fixed sin/cos + weight tying (zero I/O params)

2. **Track (in priority order):**
   - Final validation loss/accuracy → parity check
   - Train/eval gap → overfitting measurement
   - Epochs/wall-time to threshold → convergence speed

3. **Acknowledge extrapolation:** Evidence is from transformers and standard RNNs, not ring-buffer pointer networks specifically.

### 6. Key Papers Qwen Found (not in our other sources)

- **Ref 115 (RoPE for character-level):** Exploring fixed theta values in RoPE on character-level models — directly relevant architecture
- **Ref 116 (Frozen Visual Unicode):** "Emergent Semantics Beyond Token Embeddings: Transformer LMs with Frozen Visual Unicode Representations" — LMs trained with FROZEN visual Unicode representations, closest analogue to our frozen sin/cos idea
- **Ref 198 (ASR Transformer vs RNN):** The 8× speedup finding — strongest convergence evidence
- **Ref 199 (Positional Encoding Helps RNNs):** LSTM + sinusoidal = 95% accuracy on reversal with vocab 16K — THE key RNN evidence
- **Ref 236 (Stable LM Pre-training):** "Stable Language Model Pre-training by Reducing Embedding" — reducing embedding instability for stable training

---

## Claude Deep Research — THE CRITICAL CONTRARIAN FINDING

**Source:** `research/claude_research_fixed_encodings.md`

Claude's research **agrees** with fixed encoding viability but **strongly disagrees** on the encoding type.

### The Sinusoidal Similarity Problem

Using standard PE formula with base=10000, adjacent bytes have ~0.97 cosine similarity:

| Byte distance | Cosine similarity | Problem |
|---|---|---|
| 0 (same) | 1.00 | Expected |
| 1 (adjacent, e.g. 'a'/'b') | ~0.97 | NEARLY IDENTICAL |
| 10 | ~0.85-0.90 | Still very high |
| 128 (half-range) | ~0.50 | Moderate |
| 255 (max) | ~0.40 | Still correlated |

**Why this is bad for bytes:**
- Bytes are CATEGORICAL, not ordinal
- 'a' (97) and 'b' (98) are completely different tokens, not "similar"
- '9' (57) and ':' (58) have no semantic relationship but ~0.97 similarity
- **Output projection catastrophe:** `hidden @ sincos_table.T` cannot produce sharp predictions because activating any byte co-activates its neighbors

**Compare random orthogonal vectors in R^8192:** ALL pairwise cosine similarities ~0 +/- 0.011

### Solution: Hadamard Matrix Encoding

- Take 256 rows from H_8192 Hadamard matrix (exists because 8192 = 2^13)
- Entries are +/-1/sqrt(8192)
- Cosine similarity between ANY two rows = exactly 0
- Perfect token separation with zero learnable parameters
- Fast Walsh-Hadamard Transform available: O(d log d)
- Johnson-Lindenstrauss: space is ~1000x overcomplete for 256 tokens

### Key Empirical Evidence (unique to Claude's research)

- **Bochkov (2025, TMLR):** Frozen non-semantic embeddings OUTPERFORMED learned on MMLU (23.81 vs 12.54)
- **Kumar et al. (AAAI 2022):** Random Gaussian vectors lose only 1-4 BLEU. Low-resource: random BEAT learned.
- **Das & Mali (2024):** Freezing RNN components IMPROVED PTB perplexity (123.5 -> 120.5)
- **Echo State Networks (Jaeger 2001):** 20+ years of fixed random projections in recurrent nets

### W_x Compensation Proof

In `h = tanh(W_x * x + W_h * h + b)`:
- For ANY injective fixed encoding producing 256 linearly independent vectors
- W_x can map them to ANY desired 256 target vectors
- The 256x8192 system is underdetermined (infinite solutions for any target)
- **W_x o f_fixed is mathematically equivalent to a learned embedding**

### Claude's Ranked Recommendations

1. **Hadamard input + learned output** (best — perfect separation, 0 params)
2. **Factorized embedding** (256->64->8192, ~540K params — less radical)
3. **Weight tying only** (simplest, most conservative)
4. **Sin/cos for both I/O** (RISKIEST — 0.97 adjacent similarity kills output)

---

## Gemini Deep Research

**Source:** `research/gemini_research_fixed_encodings.md`

Gemini produced a structured experimental design with 4 arms, closely aligned with Qwen's approach.

### Parameter Bloat Analysis

- Input: 2.1M params (256 × 8192) — 39.5% of model
- Output: 2.1M params (8192 × 256) — 39.5% of model
- **Total I/O: 79%** of model is just lookup/projection

### 4-Arm Experimental Design

| Arm | Input | Output | I/O Params |
|-----|-------|--------|------------|
| 1. Baseline | Learned | Learned | 4.2M |
| 2. Sinusoidal Input | Fixed Sin/Cos | Learned | 2.1M |
| 3. Weight Tying | Shared A | Shared A.T | 2.1M |
| 4. Combined | Fixed Sin/Cos | Sin/Cos.T | **0** |

### Key Arguments

- **"The first layer can learn any rotation"** — same W_x compensation as all other agents
- Fixed encodings place bytes onto a deterministic spiral manifold (ordinal structure)
- Fixed encodings are **immune to embedding overfitting** (cannot memorize)
- Higher initial loss expected, but may converge to same final performance

### Referenced Models

- ByteNet (2017) — established byte-level feasibility with learned embeddings
- ByT5 (2021) — learned embeddings scale but may not be efficient for small models
- MegaByte (2023) — byte patches, alternative approach

### Verdict

Optimistic about Combined (Fixed Sin/Cos + Sin/Cos.T). Recommends testing all 4 arms. **Does NOT address** the cosine similarity problem or Hadamard alternatives (same blind spot as Grok, Kimi, Qwen).

---

## Synthesis: What This Means for INSTNCT v4

### The UPDATED Plan (incorporating all 4 agents)

**Stage 1 — Hadamard input + learned output (Claude's #1 recommendation):**
- Replace `nn.Embedding(256, 8192)` with 256 rows from Hadamard H_8192 (0 learnable input params)
- Keep `nn.Linear(8192, 256)` for output
- W_x (first hidden layer) naturally compensates — no extra layer needed
- **Saves:** 2.1M params from input embedding
- **Risk:** Low — W_x compensation is proven (reservoir computing, 20+ years)
- Cosine similarity between ALL token pairs = 0 (perfect separation)

**Stage 1b — Sin/cos input (for comparison):**
- Same setup but with sinusoidal table instead of Hadamard
- Tests whether the 0.97 similarity actually hurts in practice
- If Hadamard > sin/cos: confirms Claude's finding
- If sin/cos ≈ Hadamard: the similarity was a non-issue (W_x compensated)

**Stage 2 — Hadamard input + weight-tied output:**
- Output: `logits = hidden @ hadamard_table.T` (no learnable output params)
- With Hadamard: cosine sim = 0, so sharp predictions ARE possible
- With sin/cos: cosine sim = 0.97, sharp predictions nearly impossible
- **Saves:** 4.2M total (79% of model)
- **Risk:** Medium — sin/cos must be well-conditioned for output projection too

### Fixed encoding table generation:

**Option A — Hadamard (Claude's #1, theoretically optimal):**
```python
from scipy.linalg import hadamard
H = hadamard(8192) / math.sqrt(8192)  # normalized, entries +/-1/sqrt(8192)
table = torch.tensor(H[:256], dtype=torch.float32)  # first 256 rows
# All pairwise cosine similarities = exactly 0
```

**Option B — Random orthogonal (equivalent quality):**
```python
Q, _ = torch.linalg.qr(torch.randn(8192, 8192))
table = Q[:256]  # first 256 rows of random orthogonal matrix
# Pairwise cosine similarities ~0 +/- 0.011
```

**Option C — Sinusoidal (other agents' recommendation, but 0.97 adjacent similarity):**
```python
freqs = 1.0 / (base ** (2i / dim))  # base ≈ 100-256
table[b, 2i]   = sin(b * freqs[i])
table[b, 2i+1] = cos(b * freqs[i])
# Adjacent bytes: cosine sim ~0.97 (problematic for output)
```

**Option D — HTP-style coprime moduli (reduces but doesn't eliminate similarity):**
```python
primes = first 4096 primes > 256
table[b, 2i]   = sin(2*pi * b / primes[i])
table[b, 2i+1] = cos(2*pi * b / primes[i])
```

### Key numbers:

- Current model params: ~5.3M (estimated)
- Input embedding: 2.1M (256 x 8192) -> 0 with fixed encoding
- Output projection: 2.1M (8192 x 256) -> 0 with weight tying
- Potential savings: 4.2M params (79%)
- Remaining: ~1.1M (hidden projections, read/write projections, ring mechanics)

---

## Agent Consensus Matrix

| Question | Grok | Kimi k2 | Qwen | Gemini | Claude |
|---|---|---|---|---|---|
| Fixed encoding viable? | YES | YES | YES | YES | YES |
| Reduces overfitting? | YES | YES | YES (79%) | YES (immune to memorization) | YES |
| Faster convergence? | Likely | Likely | YES (up to 8x) | YES (higher initial loss, same final) | YES |
| Sin/cos specifically? | Good | Good | Good | Good | **BAD** (0.97 similarity) |
| Hadamard/random? | N/A | N/A | N/A | N/A | **STRONGLY recommended** |
| Weight tying? | Yes | Yes (staged) | Yes | Yes (arm 3) | Yes |
| First-layer compensation? | Yes | Yes | Yes | Yes ("any rotation") | Yes (W_x proof) |

**The split:** 4 agents say sin/cos is fine (Grok, Kimi k2, Qwen, Gemini). 1 agent (Claude) says sin/cos has a fundamental flaw and Hadamard is strictly better. The resolution: **test both** (Stage 1 vs Stage 1b in the plan).

---

## References (by relevance)

1. **Bochkov 2025 (TMLR)** — Frozen non-semantic embeddings outperform learned (Claude)
2. **Kumar et al. 2022 (AAAI)** — Random Gaussian embeddings lose only 1-4 BLEU (Claude)
3. **HTP** — Schmitz 2025 — arxiv:2511.20665 — Deterministic sinusoidal (Kimi k2)
4. **Press & Wolf** — 2017 — EACL — Weight tying foundation, 898 cit. (all agents)
5. **BLT** — Pagnoni et al. 2025 — ACL — SOTA byte-level, 123 cit. (Kimi k2)
6. **Echo State Networks** — Jaeger 2001 — Fixed random projections in RNNs (Claude)
7. **Morita 2024** — Sin/cos helps RNNs, 95% accuracy on reversal (Qwen)
8. **Das & Mali 2024** — Freezing RNN components improves perplexity (Claude)
9. **FoNE** — Zhou et al. 2025 — arxiv:2502.09741 — Fourier number embedding (Kimi k2)
10. **PIT** — Gu et al. 2026 — arxiv:2602.04556 — Pseudo-inverse tying, too heavy (deep-research)
11. **CWT** — Godey et al. 2023 — arxiv:2309.08351 — Headless LMs (deep-research)
12. **Memorization** — Tirumala et al. 2022 — Fewer params = less overfitting, 430 cit. (Kimi k2)
