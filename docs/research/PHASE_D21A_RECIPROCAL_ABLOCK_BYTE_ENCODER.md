# Phase D21A Reciprocal A-Block Byte Encoder

Date: 2026-05-02

## Summary

D21A built the base A-block as a tiny sparse reciprocal byte autoencoder:

```text
byte -> 8 visible bits -> 16D abstract code -> 8 bit logits -> byte logits
```

The decoder is always `encoder.T`, so every visible-code edge is reciprocal by
construction. This implements the intended "microphone and speaker" behavior:
the same block can encode incoming bytes and decode internal state back to byte
answers.

Implementation:

```text
tools/_scratch/d21a_reciprocal_byte_ablock.py
```

## Main Result

Main atlas:

```text
verdict: D21A_RECIPROCAL_ABLOCK_PASS
candidates scanned: 29,718
gate-pass candidates: 1,618
best family: redundant_copy_2x
code_dim: 16
edge_count: 16
exact_byte_acc: 1.0
bit_acc: 1.0
byte_argmax_acc: 1.0
byte_margin_min: +4.0
hidden_collisions: 0
reciprocity_error: 0
single_edge_drop_mean_bit: 1.0
```

Crystallize:

```text
start: 16D redundant_copy_2x, 16 reciprocal edges
compact gate-pass candidate: 14 reciprocal edges
exact_byte_acc: 1.0
bit_acc: 1.0
byte_margin_min: +2.0
single_edge_drop_mean_bit: 0.991071
```

Interpretation: the 16-edge redundant A-block is the strongest stable base
gate. It can be crystallized to 14 edges while preserving exact reconstruction
and the D21A gate, but the 16-edge version retains the best margin and edge-drop
robustness.

## Quality Interpretation

D21A is already at the accuracy ceiling for byte reconstruction:

```text
exact byte reconstruction = 100%
bit accuracy = 100%
byte argmax accuracy = 100%
```

Future improvements are therefore not accuracy improvements. They are quality
improvements:

```text
larger normalized byte distance
larger margin under noise
better edge-drop and bit-flip robustness
less trivial duplicate-bit structure
more context-ready abstract geometry
```

Compared to the older local `byte_autoencoder.log` trace, which ended around:

```text
best bit accuracy: 58.5%
best byte accuracy: 7.2%
```

D21A is a much cleaner base byte gate. This comparison is not a full fair
dense-vs-sparse benchmark because the old run used a different training shape,
but it is enough to show that the reciprocal A-block solves the byte round-trip
surface cleanly.

## Next Step

D21B should turn this into an error-correcting A-block:

```text
8 visible bits -> 16D/32D abstract code
maximize normalized pairwise byte separation
preserve 100% reconstruction
preserve reciprocal decoder
measure noise, bit-flip, and edge-drop robustness
```

If D21B passes, D21C should add a small context lane:

```text
16D byte lane + 4/8D context lane
```

and test whether context can be carried without degrading byte reconstruction.

## Status

Verdict: `D21A_RECIPROCAL_ABLOCK_PASS`

Release relevance: component-level progress only. It is not a release
checkpoint, but it gives a clean byte-to-abstract-to-byte gate for later D20/D21
context experiments.
