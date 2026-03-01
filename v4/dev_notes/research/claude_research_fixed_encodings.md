# Fixed encodings can replace learned byte embeddings - but don't use sinusoidal

**Source:** Claude Deep Research (2026-02-26)

**Fixed, zero-parameter input encodings can successfully replace learned embeddings for a 256-token byte vocabulary, with strong theoretical and empirical support - but sinusoidal encodings are the wrong choice.** The core problem: adjacent byte values (e.g., byte 97 and byte 98) share **~0.97 cosine similarity** in sinusoidal space, imposing a strong ordinal structure that doesn't match the mixed categorical nature of raw bytes. Random orthogonal or Hadamard-based encodings provide perfect separation (cosine similarity = 0 between all pairs) with zero learnable parameters, and are strictly superior.

## The sinusoidal similarity problem

Using standard transformer formula PE(pos, 2i) = sin(pos / 10000^(2i/d)):

| Byte distance | Approx. cosine similarity |
|---|---|
| 0 (same byte) | 1.00 |
| 1 (adjacent) | ~0.97 |
| 10 | ~0.85-0.90 |
| 50 | ~0.60-0.70 |
| 128 (half-range) | ~0.50 |
| 255 (maximum) | ~0.40 |

Compare: random unit vectors in R^8192 have all pairwise cosine similarities ~0 +/- 0.011.

**Output projection problem:** If logits = hidden @ sincos_table.T, the model CANNOT produce sharp probability distributions over individual bytes, because activating any byte necessarily co-activates its neighbors.

## Solution: Hadamard or random orthogonal encodings

- 256 rows from Hadamard H_8192 matrix: exactly 0 cosine similarity between ALL token pairs
- H_8192 exists (8192 = 2^13), entries are +/-1/sqrt(8192)
- Fast Walsh-Hadamard Transform: O(d log d)
- Johnson-Lindenstrauss: 256 points need only ~6,800 dims for eps=0.1, we have 8192 (1000x overcomplete)

## Key empirical evidence

- **Bochkov (2025, TMLR):** Frozen non-semantic embeddings (visual Unicode glyphs, binary token IDs). 0.3B model OUTPERFORMED trainable baseline on MMLU (23.81 vs 12.54).
- **Kumar et al. (AAAI 2022):** Random Gaussian vectors in NMT: only 1-4 BLEU drop. On low-resource: random OUTPERFORMED learned by 1.0 BLEU.
- **Das & Mali (2024):** Freezing RNN components IMPROVED PTB perplexity (123.5 -> 120.5).
- **Echo State Networks (Jaeger 2001, Jaeger & Haas Science 2004):** 20+ years of fixed random input projections in recurrent architectures.

## W_x compensation proof

In h = tanh(W_x * x + W_h * h + b), for ANY injective fixed encoding producing 256 linearly independent vectors, W_x can map them to any desired set of 256 target vectors. The 256x8192 system is underdetermined with infinitely many solutions.

## Ranked recommendations

1. **Hadamard input + learned output** (recommended first experiment)
2. **Factorized embedding** (256->64->8192, ~540K params)
3. **Weight tying only** (simplest, most conservative)
4. **Sin/cos for both I/O** (RISKIEST - cosine similarity problem)
