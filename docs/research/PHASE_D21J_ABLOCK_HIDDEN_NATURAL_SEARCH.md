# Phase D21J: Hidden Natural A-Block Search

D21J follows D21I with a stricter question:

```text
Can hidden link-location neurons carry a non-copy A manifold?
```

D21I found an exact hidden-only A-block, but the best map was still mostly
copy-like bit amplification. D21J forbids direct `visible->A16` edges and
strongly penalizes copy-like effective maps.

Model shape:

```text
visible byte bits -> hidden link neurons -> A16

effective_encoder = code_from_hidden @ hidden_from_visible
decode = effective_encoder.T
```

Full-pass criteria:

```text
exact_byte_acc == 1.0
bit_acc == 1.0
byte_margin_min > 0
hidden_collisions == 0
direct_edge_count == 0
effective_copy_penalty <= 0.45
ASCII geometry beats D21A
```

Interpretation:

```text
D21J_HIDDEN_NATURAL_A_PASS:
  hidden-only can carry a non-copy exact A manifold

D21J_HIDDEN_NATURAL_WEAK_PASS:
  hidden-only non-copy exists, but margin/geometry is not production-grade

D21J_NO_NONCOPY_HIDDEN_A:
  hidden-only exact candidates collapse back to copy-like maps
```

Generated output remains under `output/` and is not committed.

Current bounded main result:

```text
D21J_HIDDEN_NATURAL_A_PASS

best candidate:
  arm: factor_d21g_h8
  hidden_dim: 8
  direct_edge_count: 0
  hidden_in_edge_count: 8
  hidden_out_edge_count: 29
  effective_edge_count: 29
  exact_byte_acc: 1.0
  byte_margin_min: +2.5
  ascii_class_geometry: 0.7767
  effective_copy_penalty: 0.0
```

Interpretation: hidden neurons can carry a non-copy exact A manifold. H8 is
already sufficient for the D21G-style natural A map; H16/H32 did not provide a
clear extra benefit in this bounded run. Compared with D21I, this is not just a
bit-gain trick, but a hidden factorization of the natural sparse A candidate.

Practical status:

```text
D21A remains the minimal shipped A-block.
D21I is the hidden bit-gain lead.
D21J is the hidden natural/non-copy A_v2 lead.
```
