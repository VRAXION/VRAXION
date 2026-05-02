# Phase D23 B-Block Word Abstraction

Date: 2026-05-02

## Summary

D23 adds the first reciprocal B-block probe on top of the D22 A-window surface:

```text
8 bytes
  -> D22 128D A-window code
  -> B-block latent
  -> reciprocal B-block inverse
  -> D22 128D A-window code
  -> 8 bytes
```

This is not a reasoning block yet. It proves whether the next abstraction layer
can compress an exact byte-window representation while preserving reversibility
and useful local geometry.

Implementation:

```text
tools/_scratch/d23_bblock_reciprocal_word_abstraction.py
```

## Result

Verdict:

```text
D23_BBLOCK_64D_PASS
```

Main run:

```text
window_bytes: 8
eval_windows: 65536
geometry_pairs: 200000
tested latent widths: 32,48,64,96,128
```

Best candidate:

```text
family: B0_block_average
latent_width: 64
window_exact_acc: 1.0
byte_exact_acc: 1.0
bit_acc: 1.0
byte_margin_min: +2.0
collision_count: 0
encoder_weight_count: 64
reciprocity_error: 0.0
```

Geometry:

```text
hamming_distance_correlation: 0.266428
one_byte_neighbor_closer_rate: 0.999845
prefix_neighbor_closer_rate: 0.999845
random_far_margin: +0.872942
```

Controls:

```text
position_shuffle_control: 0.000122
random_code_control: 0.0
random_projection_control: 0.0
```

## Width Sweep

```text
D23 B-block heatmap: brighter = B_score, P=pass R=recon-only F=fail
latent family                  cell exact geom1 geomP corr verdict
    32 B0_block_average        %F 0.002 1.000 1.000 0.185 D23_BBLOCK_TOO_COMPRESSED
    48 B0_block_average        %F 0.008 1.000 1.000 0.225 D23_BBLOCK_TOO_COMPRESSED
    64 B0_block_average        @P 1.000 1.000 1.000 0.266 D23_BBLOCK_64D_PASS
    96 B0_block_average        @F 1.000 1.000 1.000 0.254 D23_BBLOCK_FAIL
   128 B0_identity_128         @P 1.000 1.000 1.000 0.252 D23_BBLOCK_128D_REFERENCE_PASS
```

Interpretation:

```text
32D / 48D:
  too compressed; reconstruction fails.

64D:
  primary pass; exact, reciprocal, collision-free, and controls clean.

96D:
  exact reconstruction, but random_projection_control reaches about 9.8%.
  This makes it a weaker proof target than 64D, not the preferred candidate.

128D:
  expected full-width reference; exact and reciprocal, but not compact.
```

## What The 64D B-Block Means

The D22 128D surface stores each byte as a redundant 16D A-code. D23 shows that
the B-block can keep the visible 8 bit lanes per byte and still reconstruct the
whole 8-byte window exactly:

```text
D22 research surface:
  8 bytes * 16 dims = 128D

D23 B latent:
  8 bytes * 8 visible-bit dims = 64D
```

This is a real reciprocal abstraction layer in the current prototype sense:

```text
128D A-window abstract <-> 64D B-window abstract
```

It is still structural byte/window abstraction, not semantic word understanding.

## Why This Matters

D21 proved small state/memory primitives. D22 proved a clean 8-byte A-window
surface. D23 proves the first compact B-level reciprocal window abstraction:

```text
A-block:
  byte <-> byte abstract

D22:
  8 bytes <-> 128D A-window abstract

D23:
  128D A-window abstract <-> 64D B-window abstract
```

The next useful phase is D24: run small transformations over the 64D B latent
and decode them back to bytes.

## Commands

Smoke:

```powershell
python tools\_scratch\d23_bblock_reciprocal_word_abstraction.py --mode smoke --window-bytes 8 --latent-widths 64,128 --eval-windows 4096 --geometry-pairs 20000 --out output\phase_d23_bblock_word_abstraction_20260502\smoke
```

Main:

```powershell
python tools\_scratch\d23_bblock_reciprocal_word_abstraction.py --mode width-sweep --window-bytes 8 --latent-widths 32,48,64,96,128 --eval-windows 65536 --geometry-pairs 200000 --out output\phase_d23_bblock_word_abstraction_20260502\main
```

## Tests

Passed:

```text
python -m py_compile tools\_scratch\d23_bblock_reciprocal_word_abstraction.py
python tools\check_public_surface.py
D23 smoke
D23 main width sweep
```

Generated `output/` remains uncommitted.

## Next Step

D24 should use the `64D` B latent as the primary surface:

```text
input:  8-byte window encoded into 64D B latent
task:   tiny reversible byte-window transformations
output: decoded 8-byte window
```

Recommended first D24 tasks:

```text
copy window
reverse window
rotate-left / rotate-right
marker-copy inside the 8-byte window
```
