# A-GeometryAuditRevival

Date: 2026-05-03

## Summary

This revives the older byte-embedding geometry audit idea from the
`byte-embed-dim` notes and the `geometry_sweep` / `geometry_explain` history.
The old question was:

```text
Is the byte embedding only a reversible identity code,
or do related bytes land close to each other?
```

For the current A-block candidates, after margin polish, the answer is:

```text
A-HiddenNaturalMarginPolish has the best natural byte geometry.
A-StableCopy16 remains the safer shipped/default A-block because its decode margin is higher.
```

## Main Result

```text
name                exact margin geom  rank pca95 avgCos copy audit
------------------- ----- ------ ----- ---- ----- ------ ---- ------
A-HiddenNatural16   1.000  3.516 0.770  7.8     8  0.310 0.00  29.15
A-NaturalSparse16   1.000  2.500 0.764  7.8     8  0.305 0.00  28.85
A-HiddenBitGain16   1.000  4.000 0.731  7.9     8  0.313 0.94  27.17
A-StableCopy16      1.000  4.000 0.669  8.0     8  0.270 1.00  26.49
```

Interpretation:

```text
A-HiddenNaturalMarginPolish:
  best near/far byte geometry
  zero copy penalty
  still exact roundtrip
  stronger margin than the earlier A-HiddenNatural16 lead

A-StableCopy16:
  strongest simple safety/margin
  still the shipped/default A-block
  geometry is more identity/Hamming-like
```

## What The Audit Measures

The audit computes full 256-byte distance/similarity matrices and checks:

```text
ASCII neighbor closer rate:
  A-B should be closer than A-Z style far pairs.

case pair closer rate:
  A-a should be closer than A-7.

digit neighbor closer rate:
  1-2 should be closer than 1-Z.

class cluster score:
  digits, uppercase, lowercase, punctuation, space should separate.

nearest-neighbor sanity:
  the nearest bytes should look explainable, not arbitrary.

effective rank / cosine overlap:
  the code should use the available space without collapsing into low-rank mush.
```

## Concrete Example

Nearest neighbors around selected bytes:

```text
A-StableCopy16:
  A -> 0x01, Q, a, E, C, @ ...
  This is mostly Hamming-bit identity behavior.

A-HiddenNaturalMarginPolish:
  A -> @, E, C, 0x01, a, ...
  a -> `, c, e, !, A, ...
  0 -> 1, 2, 4, ...
  , -> -, (, ., ...
  This is closer to the intended natural ASCII geometry.
```

## Files

Generated evidence is intentionally under uncommitted `output/`:

```text
output/phase_a_geometry_audit_revival_20260503/main/a_geometry_audit_summary.csv
output/phase_a_geometry_audit_revival_20260503/main/a_nearest_neighbors.csv
output/phase_a_geometry_audit_revival_20260503/main/a_group_cluster_scores.csv
output/phase_a_geometry_audit_revival_20260503/main/a_ascii_distance_heatmap.txt
output/phase_a_geometry_audit_revival_20260503/main/A_GEOMETRY_AUDIT_REVIVAL_REPORT.md
```

Code:

```text
tools/_scratch/a_geometry_audit_revival.py
```

## Decision

```text
Ship/default:
  A-StableCopy16

A_v2 research lead:
  A-HiddenNaturalMarginPolish

Next A-block work:
  AB compatibility testing with A-HiddenNaturalMarginPolish.
```

The important result is not “more hidden neurons are always better.” The result
is narrower:

```text
Hidden link locations can create better byte geometry and now a much stronger
margin, but the shipped AB artifact still needs compatibility testing before replacement.
```
