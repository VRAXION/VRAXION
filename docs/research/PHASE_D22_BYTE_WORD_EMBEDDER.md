# Phase D22 Byte-to-Word Embedder

Date: 2026-05-02

## Summary

D22 composes the D21 A-block into a fixed 8-byte window adapter:

```text
8 bytes -> 8 parallel D21A A-block codes -> 128D word-ish code -> 8 bytes
```

This is not semantic word understanding. It is the byte-to-window substrate that
can sit below deeper B/C blocks.

Implementation:

```text
tools/_scratch/d22_byte_word_embedder.py
```

## Result

Verdict:

```text
D22_COMPACT_WORD_EMBEDDER_PASS
```

Confirm run:

```text
window_bytes: 8
eval_windows: 65536
tested widths: 32,64,96,128
```

Pack heatmap:

```text
D22 pack heatmap: brighter = score, P=pass F=fail
width   cell  exact  margin  drop_bit  verdict
   32    F    0.002    0.00  0.735   D22_WORD_EMBEDDER_FAIL
   64   @P    1.000    2.00  0.992   D22_COMPACT_WORD_EMBEDDER_PASS
   96   @P    1.000    2.00  0.997   D22_WORD_EMBEDDER_PASS
  128   @P    1.000    4.00  1.000   D22_WORD_EMBEDDER_PASS
```

Best robust reference:

```text
pack: full_128_ablock
word_width: 128
window_exact_acc: 1.0
byte_exact_acc: 1.0
bit_acc: 1.0
byte_margin_min: +4.0
sample_hidden_collisions: 0
single_dim_drop_mean_window_exact: 1.0
single_dim_drop_mean_bit: 1.0
position_shuffle_window_exact_acc: 0.000122
random_code_window_exact_acc: 0.0
```

Best compact candidate:

```text
pack: compact_64_visible_bits
word_width: 64
window_exact_acc: 1.0
byte_exact_acc: 1.0
bit_acc: 1.0
byte_margin_min: +2.0
sample_hidden_collisions: 0
single_dim_drop_mean_window_exact: 0.513782
single_dim_drop_mean_bit: 0.991972
position_shuffle_window_exact_acc: 0.000122
random_code_window_exact_acc: 0.0
```

## Size And Packability

The 8-byte window has two useful deployment surfaces:

```text
128D robust reference:
  16 dims per byte
  int8 window vector: 128 bytes
  byte LUT: 4096 bytes
  single-dim-drop exact: 100%

64D compact baseline:
  8 dims per byte
  int8 window vector: 64 bytes
  byte LUT: 2048 bytes
  exact reconstruction: 100%
  less robust to single-dim drops
```

Meaning: `128D` is the safer research/default width; `64D` is the compact exact
deployment candidate.

## Why This Matters

D21 proved that one A-block can do byte IO, context write, recurrent state,
marker memory, and multi-slot memory. D22 shows that many A-blocks can sit side
by side as a clean byte-window surface:

```text
A-block = one byte IO/memory cell
D22     = 8-byte window embedder
```

This supports the next deeper block:

```text
B-block:
  operate over 8-byte / 64D-128D window embeddings
  apply small rules or transformations
  use D21E memory when needed
```

## Commands

Smoke:

```powershell
python tools\_scratch\d22_byte_word_embedder.py --mode baseline-check --window-bytes 8 --widths 32,64,96,128 --eval-windows 4096 --out output\phase_d22_byte_word_embedder_20260502\smoke
```

Confirm:

```powershell
python tools\_scratch\d22_byte_word_embedder.py --mode confirm --window-bytes 8 --widths 32,64,96,128 --eval-windows 65536 --out output\phase_d22_byte_word_embedder_20260502\confirm
```

## Tests

Passed:

```text
python -m py_compile tools\_scratch\d22_byte_word_embedder.py
D22 smoke
D22 confirm
```

Generated `output/` remains uncommitted.

## Next Step

D22B / B-block should operate on this window surface:

```text
input:  8-byte window embedding
task:   recall/transform/copy small byte facts
output: byte or byte-window answer
```

Default recommendation:

```text
research width: 128D
compact deploy width: 64D
```
