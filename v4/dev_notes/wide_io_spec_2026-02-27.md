# Wide I/O - Research-Backed Spec (2026-02-27)

## TL;DR

Wide I/O is promising, but the original draft is too optimistic about `K=8` as a default.
Literature consensus is:

- Raw bytes can work, but naive byte modeling often pays a speed tax.
- Strong byte models that scale well use hierarchy (global + local), patching, or downsampling.
- If we predict multiple bytes per step (`K>1`) without a local intra-frame decoder, quality can degrade.

For VRAXION v4, the safest first target is `K=2` or `K=4`, not `K=8` as an assumed sweet spot.

## Why This Revision

The previous spec assumed:

1. `K=8` is the sweet spot.
2. Intra-frame autoregressive loss is acceptable to drop.
3. Fewer steps alone should give better trade-off.

After paper review, these should be treated as hypotheses, not facts.

## External Evidence (Primary Sources)

### 1) ByT5 (token-free baseline)

Source: ByT5 paper (arXiv/TACL).

What matters for us:

- Byte-level models are viable and robust.
- But naive byte processing is slower in multiple settings.
- In ByT5 Table 9, pretrain throughput is about `0.75x` mT5 (`1232 vs 1646`, `576 vs 747`, `232 vs 306`, `70 vs 94`, `25 vs 33` seq/s).
- In ByT5 Table 10, slowdown is mild on some word tasks but large on sentence/doc tasks (roughly `3.7x` to `9.5x` slower on GEM-XSum/XNLI rows).

Takeaway:

- "Byte-level works" is true.
- "Byte-level is automatically faster" is false without architectural help.

### 2) CANINE

Source: CANINE paper (arXiv/TACL).

What matters for us:

- Tokenization-free encoder succeeds by combining character input with downsampling + deep stack.

Takeaway:

- Compression/downsampling is not optional decoration; it is a key efficiency mechanism.

### 3) Charformer (GBST)

Source: Charformer paper (arXiv/ICLR).

What matters for us:

- Learns latent subword blocks from byte/char stream.
- Reports speed gains (`28%` to `100%`) over byte/subword baselines with competitive quality.

Takeaway:

- Learned local block structure beats flat byte processing.

### 4) MEGABYTE

Source: MEGABYTE paper (arXiv).

What matters for us:

- Uses multiscale modeling: local model inside patches + global model between patches.
- Claims better cost/quality scaling for long byte sequences.

Takeaway:

- If we widen input into patches, we should preserve local autoregressive structure somehow.

### 5) BLT (Byte Latent Transformer)

Source: BLT (ACL/OpenReview).

What matters for us:

- Dynamically sized patches based on byte entropy.
- Reports improved scaling under fixed inference cost.

Takeaway:

- Adaptive patching can outperform fixed-token assumptions, but only with explicit patch machinery.

## Implication for VRAXION Wide I/O

## What still makes sense

- Increase bytes processed per model step (`K>1`) to reduce recurrent step count.
- Reallocate parameter budget away from large discrete I/O tables when possible.

## What must be corrected

- `K=8` is not a guaranteed sweet spot.
- Predicting `K` bytes in parallel with one shared hidden state removes intra-frame dependency.
- Fair comparison must be done at equal bytes seen, not equal steps.

## Revised Design Direction

### Stage A (safe baseline)

- Keep current `K=1` run as baseline.
- Lock eval protocol and datasets.

### Stage B (first real wide-I/O probe)

- Test `K=2` and `K=4` first.
- Use low-rank output head (or tied/shared head with explicit position signal).
- Keep model core unchanged.

Rationale:

- This isolates whether wider framing helps before adding complex local decoders.
- Lower risk of destroying byte-level next-token behavior.

### Stage C (only if Stage B is positive)

- Try `K=8` with a small local intra-frame decoder.
- Example pattern: global ring state predicts a frame context, then a tiny local AR head predicts bytes within the frame.

Rationale:

- Closer to MEGABYTE/BLT style global-local decomposition.

## Evaluation Protocol (must-have)

Every K experiment must report:

1. Train/eval on the same data mix and same split policy.
2. Compare at equal bytes seen (`tokens_seen * K`) and equal wall-clock windows.
3. Metrics:
   - `loss_per_byte`
   - `masked_acc` (if used)
   - `bytes/sec`
   - `steps/sec`
   - peak VRAM
4. At least one fixed held-out eval set that is unchanged across runs.

Without this, `K` conclusions are not trustworthy.

## Parameter Math (keep, but mark as provisional)

Given hidden size `H=2048`:

- Input projection from bits: `Linear(8K, H)` -> params `8K*H + H`.
- Naive output: `Linear(H, 256K)` -> params `H*256K + 256K` (usually too large as K grows).
- Practical output options:
  - low-rank factorization,
  - shared head + per-position adapter,
  - tiny local decoder over frame positions.

These are engineering options, not guaranteed quality winners.

## Updated Recommendation

For immediate iteration on your GPU:

1. Treat `K=4` as first candidate knee.
2. Keep `K=8` as a second-stage experiment only.
3. Do not declare success/failure from step-level curves alone; use equal-byte comparisons.
4. If `K=4` is neutral or better on quality with higher bytes/sec, keep going.
5. If `K=4` drops quality materially, add local intra-frame decoder before trying larger K.

## Notes For Claude (message-style)

- The wide-I/O idea is directionally valid.
- The critical missing piece is fair evaluation and intra-frame structure handling.
- Please avoid hard-coding `K=8 sweet spot` in docs/config until `K=2/4` evidence is in.
- If we want "patching", we should do it in a global+local way (MEGABYTE/BLT style), not only flat parallel byte logits.

## References

- ByT5: https://arxiv.org/abs/2105.13626
- CANINE: https://arxiv.org/abs/2103.06874
- Charformer: https://arxiv.org/abs/2106.12672
- MEGABYTE: https://arxiv.org/abs/2305.07185
- BLT (OpenReview): https://openreview.net/forum?id=UZ3J8XeRLw
