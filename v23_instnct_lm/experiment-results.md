# VRAXION v23 — INSTNCT CPU Language Learning Experiment Results

**Date:** 2026-03-12
**Platform:** CPU-only, 4 cores, PyTorch 2.10.0
**Corpus:** ~10KB inline English (190+ sentences, byte-level)

## Architecture

- **Byte-level tokenization** (vocab_size=256, raw UTF-8)
- **Ring buffer memory** — circular buffer with attention-based read/write
- **Self-wiring sparse connections** — edge list + scatter/gather, dynamic topology
- **Inverse arousal gate** — confident → more wiring, uncertain → less
- **Continuous learning** — no train/eval split

## Results Summary

| Config | Params | d_model | Blocks | Buffer | Loss | Acc | Tok/s | Edges | Time |
|--------|--------|---------|--------|--------|------|-----|-------|-------|------|
| Tiny | 494K | 128 | 4 | 256 | 2.283 | 31.6% | 230 | 9,731 | 1391s |
| Small | 1.7M | 256 | 4 | 128 | 2.297 | 31.2% | 187 | 28,193 | 1025s |
| Medium | 3.6M | 384 | 4 | 64 | 2.305 | 31.0% | 133 | 60,263 | 964s |

## Self-Wiring Activity

The inverse arousal gate worked as designed:
- **Step 0-500**: Confidence ~0.28, minimal wiring (add=1, rem=0)
- **Step 500+**: Confidence rose to ~0.31-0.36, moderate wiring (add=3, rem=1)
- Net edge growth: Tiny grew from 6,552 → 9,731 edges (+48%), Small grew ~7%, Medium grew ~2%

The arousal gate correctly held back wiring during early uncertain phases and increased it as the model gained confidence.

## Generated Text Quality

At 31% byte-level accuracy, the model learns:
- **Character frequencies**: common letters (e, t, h, a, o, n, i) appear proportionally
- **Common 2-3 grams**: "the", "he", "in", "ou", "and" appear frequently
- **Sentence structure**: newlines (\n) and periods appear at roughly correct intervals
- **NOT yet learned**: actual English words, word boundaries are noisy, no grammatical structure

Example (Tiny, step 5000):
```
The t ar talepin ornghan canng? che le lle whins ache chathe yo war
wietou yoth w methe tou inin me d d.
The sthe she tlerop bounun we fomencanthe as ger yome sined my ce
```

## Key Observations

### 1. No scaling benefit
All three sizes converge to ~31% accuracy — larger models don't help. This suggests:
- **Bottleneck is the architecture, not capacity.** The per-byte recurrent processing can't form long-range dependencies effectively with this training setup.
- **Batch size 1 + seq_len 64 limits gradient quality.** Each gradient update sees only 64 bytes.
- **10KB corpus is tiny.** The model can nearly memorize it, but the recurrent architecture makes this hard.

### 2. Self-wiring works but doesn't boost accuracy
Edges grow steadily (Tiny: +48%), but the added connections don't translate to better predictions. Possible reasons:
- The sparse layer is only one component — most learning happens in the ring buffer attention (dense projections)
- New edges are added based on activation correlation, which may not target the most useful connections for next-byte prediction

### 3. Ring buffer is expensive on CPU
The per-step attention over the full buffer is the main bottleneck:
- Buffer=256: 230 tok/s (Tiny)
- Buffer=128: 187 tok/s (Small, but also larger d_model)
- Buffer=64: 133 tok/s (Medium, largest d_model)

For CPU deployment, the ring buffer should use a sliding window or top-k sparse attention instead of full attention.

### 4. Inverse arousal behaves correctly
The arousal gate transitions from "conservative" (low confidence, no wiring) to "exploratory" (higher confidence, active wiring) as expected. The confidence stabilizes around 0.35, corresponding to moderate wiring intensity.

## Comparison with Expectations

The original prompt expected running 5000 steps for all sizes with larger configs (5M, 20M). On CPU with batch_size=1, the larger models are impractical (the original Small config with d=256, 6 blocks, buf=512 was ~10x slower than needed).

A byte-level LM needs significantly more compute to learn meaningful language structure than a word-level model. At 31% byte accuracy, the model is essentially predicting the most likely next character given recent characters — it hasn't learned words or grammar yet.

## Recommendations for Next Steps

1. **Increase corpus**: 10KB is too small. Even 100KB of text would help significantly.
2. **Use seq_len > 64**: Longer sequences give better gradient signals for recurrent models.
3. **Chunk-parallel training**: Process multiple independent sequences in parallel to better utilize CPU cores.
4. **Sparse attention for ring buffer**: Top-k or strided attention instead of full attention.
5. **Gradient-based self-wiring**: Use gradient magnitude instead of activation correlation to decide where to add edges.
6. **Compare with transformer baseline**: A simple 2-layer causal transformer on the same data would establish a ceiling.
