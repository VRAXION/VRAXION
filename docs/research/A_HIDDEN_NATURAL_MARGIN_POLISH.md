# A-HiddenNaturalMarginPolish

Date: 2026-05-03

## Summary

Goal: improve the decode safety margin of `A-HiddenNatural16` without losing
its non-copy byte geometry.

Result:

```text
A_HIDDEN_NATURAL_MARGIN_STRONG_PASS
```

The polished candidate keeps exact byte roundtrip and remains non-copy, while
raising margin from `+2.5` to `+3.516`.

## Before / After

```text
Before: A-HiddenNatural16
  exact roundtrip: 100%
  margin:          +2.500
  geometry:        0.777
  copy penalty:    0.00

After: A-HiddenNaturalMarginPolish
  exact roundtrip: 100%
  margin:          +3.516
  geometry:        0.770
  copy penalty:    0.00
  direct edges:    0
```

## Confirmed Candidate

```text
hidden_dim:             8
direct_edge_count:      0
hidden_in_edge_count:   8
hidden_out_edge_count:  30
effective_edge_count:   30
hidden_collisions:      0
reciprocity_error:      0.0
single_edge_drop_mean_bit: 0.998958
```

Final `A-GeometryAuditRevival` ranking:

```text
name                exact margin geom  rank pca95 avgCos copy audit
------------------- ----- ------ ----- ---- ----- ------ ---- ------
A-HiddenNatural16   1.000  3.516 0.770  7.8     8  0.310 0.00  29.15
A-NaturalSparse16   1.000  2.500 0.764  7.8     8  0.305 0.00  28.85
A-HiddenBitGain16   1.000  4.000 0.731  7.9     8  0.313 0.94  27.17
A-StableCopy16      1.000  4.000 0.669  8.0     8  0.270 1.00  26.49
```

## Decision

```text
New A_v2 strong candidate:
  A-HiddenNaturalMarginPolish

Still shipped/default:
  A-StableCopy16
```

Reason:

```text
A-HiddenNaturalMarginPolish is stronger than the previous A_v2 lead,
but AB-WindowCodec64 is still built around the stable A path.
Do not silently replace the shipped A-block until AB rebuild + compatibility
tests pass.
```

## Generated Evidence

Generated output remains uncommitted:

```text
output/phase_a_hidden_natural_margin_polish_20260503/main/
output/phase_a_hidden_natural_margin_polish_20260503/confirm/
output/phase_a_hidden_natural_margin_polish_20260503/geometry_audit/
```

Code:

```text
tools/_scratch/a_hidden_natural_margin_polish.py
```

## Next Step

```text
A-v2 compatibility check:
  rebuild AB-WindowCodec64 using A-HiddenNaturalMarginPolish
  verify 8-byte window roundtrip remains exact
  verify B64 transforms/memory/router still pass or document required migration
```
