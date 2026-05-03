# Phase D21H: A-Block Champion Compare

D21H compares the currently relevant byte-to-abstract A candidates on one shared deterministic table:

```text
legacy_int4_c19_h24
binary_c19_h16
d21a_redundant_copy_2x
d21g_natural_sparse
```

Metrics:

```text
exact byte reconstruction
bit accuracy
minimum byte margin
unique hidden/code count
ASCII geometry
edge count
architecture cleanliness
```

This phase is a decision aid. It does not automatically replace the current AB pipeline artifact.

Current D21H decision:

```text
D21H_ABLOCK_CHAMPION_CURRENT_PIPELINE

Production/default AB A-block:
  d21a_redundant_copy_2x
  exact_byte_acc = 1.0
  byte_margin_min = 4.0
  edge_count = 16

Research/natural lead:
  d21g_natural_sparse
  exact_byte_acc = 1.0
  byte_margin_min = 2.5
  edge_count = 28
  better geometry, but fractional/research-clean only

Legacy extra-hidden byte codec:
  legacy_int4_c19_h24
  exact_byte_acc = 1.0
  byte_margin_min = 10.5368
  edge_count = 416
  strongest raw margin, but not the frozen reciprocal AB interface
```

Decision: keep D21A as the shipped AB surface for now. Keep D21G as the natural-geometry research branch. Keep C19 as a historical/reference byte codec, not as the current A-block replacement.
