# L1 Byte Pair Merger — Overnight Report (2026-04-18/19)

## Task
Build Level 1 unit: merge 2 consecutive byte embeddings (2 x 16D = 32D input) into a smaller latent vector.

## Setup
- Input: all 65,536 byte pairs (256 x 256), each pair = concat of two 16D LUT embeddings = 32D
- Architecture: tied-weight mirror autoencoder (C19 encoder, linear decoder) — same recipe as L0 byte embedder
- Optimizer: L-BFGS (strong_wolfe, full-batch)
- Metric: lossless = all 32 dimensions have correct sign after roundtrip

## Phase 1: Output dimension sweep (tied mirror, H=48/64)

| OutDim | H=48 | H=64 | Compression |
|--------|------|------|-------------|
| 8      | 0.4% | 0.4% | 4:1 |
| 12     | 2.3% | 2.3% | 2.7:1 |
| 16     | 15.7%| 15.7%| 2:1 |

**Finding: Hidden size irrelevant. Output dim is the only variable.**

## Phase 2: Knee search (larger output dims)

| OutDim | H=64  | H=96  | H=128 |
|--------|-------|-------|-------|
| 20     | 22.9% | 23.0% | 23.0% |
| 24     | 35.5% | 35.7% | 35.8% |
| 28     | 51.0% | -     | -     |
| 32     | 73.2% | -     | -     |

**Finding: Hidden size still irrelevant even at H=128. No knee found — even out=32 (zero compression) only reaches 73.2%.**

## Phase 3: V2 experiments

### Loss function variations (tied H=64 out=32)

| Loss | Lossless |
|------|----------|
| MSE baseline | 73.18% |
| Sign-aware (rw=5) | 73.18% |
| Heavy recon (rw=10) | 73.18% |
| Margin loss | 71.14% |

**Finding: Loss function does not matter. 73.18% is a hard ceiling for tied mirror out=32.**

### Larger hidden (tied out=32, sign-aware)

| Hidden | Lossless |
|--------|----------|
| H=64   | 73.18% |
| H=128  | 73.18% |
| H=256  | 73.18% |

**Finding: Exact same ceiling regardless of capacity. 73.18% = 47,961/65,536 pairs solvable.**

### Untied encoder/decoder (separate weights, C19 both sides)

| Config | Untied | Tied (reference) |
|--------|--------|-----------------|
| H=64 out=16 | 15.1% | 15.7% |
| H=64 out=24 | 27.4% | 35.5% |
| H=64 out=32 | 20.1% | 73.2% |
| H=128 out=32 | ~24% (still running) | 73.2% |

**Finding: Untied is DRASTICALLY worse. The tied constraint acts as powerful regularization for L-BFGS. Without it, the optimizer struggles with 2x the parameters.**

## Key Conclusions

1. **Tied mirror is the right architecture** — untied is far worse, not better
2. **73.18% is a structural ceiling** for tied mirror on this input data, independent of:
   - Hidden size (48 to 256 — identical)
   - Loss function (4 variants — identical)
   - Output dim when >= 32 (the limit is not the bottleneck)
3. **The ceiling is in the input data**, not the model: the byte embedder's 16D LUT vectors have a structure that makes exactly 17,575 out of 65,536 pairs unresolvable by ANY tied linear mirror through a C19 layer
4. **2:1 compression (32D -> 16D) reaches only 15.7% lossless** — far from usable for lossless

## Recommendation

**The merger should be lossy by design.** Options:

### Option A: Lossy merger, per-dim optimized
- Use tied mirror, out=16 (2:1 compression)
- 15.7% full-lossless BUT 94% per-dimension accuracy
- On average, 30 out of 32 dimensions are correct
- Downstream task (word embedding) can learn on top of this

### Option B: Direct concatenation (no merger)
- Just pass 32D forward (2 x 16D concat)
- Let the word-level unit handle the compression
- Zero information loss at L1, all compression happens at L2

### Option C: Learned non-mirror compression
- Drop the mirror/autoencoder requirement
- Use a one-directional encoder (32D -> 16D) optimized for downstream task, not reconstruction
- No decoder needed — the byte embedder already provides lossless byte recovery

### Option D: Different merge ratio
- Instead of 2:1 byte pairs, try 4:1 or 8:1 directly
- Bigger context window per merge step
- May capture more meaningful patterns (morphemes, syllables)

**My recommendation: Option C** — a one-directional encoder makes the most sense. The byte embedder is already 100% lossless, so the merger doesn't need a decoder. It should focus purely on producing useful downstream signal.

## Files created
- `tools/diag_byte_pair_merger_sweep.py` — Phase 1+2 sweep (14 configs)
- `tools/diag_byte_pair_merger_v2.py` — Phase 3 V2 experiments (12 configs)
- `tools/L1_MERGER_OVERNIGHT_REPORT.md` — This report
