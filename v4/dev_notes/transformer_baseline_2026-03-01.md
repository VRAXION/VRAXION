# Transformer Baseline Comparison — 2026-03-01

## Goal

Fair A/B comparison: INSTNCT ring-buffer architecture vs standard GPT-style transformer,
matched at ~710K parameters, on byte-level WikiText-103 language modeling.

## Setup

Both models trained with **identical** settings:
- Data: WikiText-103 (521 MB, byte-level, masked supervision)
- Steps: 3000
- Batch: 128 × 64 bytes
- LR: 1e-3, cosine decay with 100-step warmup
- Optimizer: Adam
- Grad clip: 10.0
- Device: RTX 4070 Ti SUPER

### INSTNCT (Run 33E)
- hidden_dim=2048, slot_dim=64, M=1024, N=2, R=1, S=0.3, vshape
- bitlift input encoding, lowrank_c19 output encoding
- Sequential mode (state persists across sequences)
- 711K params

### TinyTransformer (Run 34)
- d_model=128, n_layers=3, n_heads=4, d_ff=576
- Sinusoidal positional encoding (0 extra params)
- Causal attention mask (autoregressive)
- No cross-sequence state (fresh each 64-token window)
- 710K params

## Results

```text
                    INSTNCT (ring)     Transformer        Delta
────────────────────────────────────────────────────────────────
Params              711,210            710,208            -0.1%
Peak accuracy       47.8% @s2960       52.9% @s2530       +5.1%
Final accuracy      46.9% @s3000       50.7% @s3000       +3.8%
Last 200 avg        46.6%              51.5%              +4.9%
BPB (final)         2.69               2.39               -0.30
Train time          2923s (49 min)     47s (< 1 min)      63× faster
Speed/step          1.0 s/step         0.015 s/step       66× faster
VRAM                5905 MB            ~2000 MB           3× less
```

## Learning Curves

### INSTNCT (Run 33E)
```text
Step  | Acc    | Phase gain
  500 | 35.2%  | +35.2%
 1000 | 41.0%  | +5.8%
 1500 | 42.6%  | +1.6%
 2000 | 44.4%  | +1.8%
 2500 | 47.3%  | +2.9%
 3000 | 46.9%  | -0.4% (LR exhausted)
```

### Transformer (Run 34)
```text
Step  | Acc    | Phase gain
  500 | 38.7%  | +38.7%
 1000 | 45.5%  | +6.8%
 1500 | 48.3%  | +2.7%
 2000 | 49.9%  | +1.6%
 2500 | 50.0%  | +0.1%
 3000 | 50.7%  | +0.7%
```

Transformer leads at every checkpoint. The gap widens after step 1000.

## Analysis

### Why Transformer Wins

1. **Parallel context**: Transformer sees all 64 tokens at once via full causal attention.
   INSTNCT processes byte-by-byte (recurrent) — at position t, it has only seen
   positions 0..t-1 through its hidden state, not directly.

2. **Efficient capacity use**: With d_model=128 and 3 layers, the transformer gets
   3 rounds of self-attention refinement over the entire window. INSTNCT gets one
   pass per token through hidden_dim=2048, but most of that capacity goes to the
   ring buffer compression (2048→64→2048) rather than direct pattern matching.

3. **Speed advantage compounds**: At 66× faster per step, the transformer can do
   200,000 steps in the same time INSTNCT does 3000. Fair wall-clock comparison
   would be even more lopsided.

### What INSTNCT Has (But Doesn't Help Here)

- **Cross-sequence memory**: Ring buffer persists across 64-token windows. But for
  WikiText byte prediction, local context (last 10-30 bytes) matters most. The
  benefit of remembering 1000+ bytes ago is marginal.

- **Sequential processing**: INSTNCT sees the data causally by construction (no
  mask needed). But the transformer achieves the same with a simple causal mask
  and gets parallelism for free.

## Conclusion

**On byte-level WikiText-103 at 710K params, the standard transformer is strictly
better than the INSTNCT ring architecture: +5% accuracy, 63× faster, 3× less VRAM.**

The ring buffer's cross-sequence memory doesn't compensate for the transformer's
within-window attention advantage on this task. For the ring architecture to shine,
it would need a task where very long-range memory (hundreds/thousands of tokens)
is essential and local context is insufficient.

## Files

- Transformer model: `model/tiny_transformer.py`
- INSTNCT results: `training_output/run33e_baseline_3000steps/`
- Transformer results: `training_output/run34_transformer_3000steps/`
- train.py flag: `--model transformer` (new)
