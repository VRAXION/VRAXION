# Phase D21I: A-Block Hidden Sparse Sweep

D21I tests whether the A-block improves when fixed byte IO is kept but extra
hidden link-location neurons are added.

Compared structures:

```text
io_only:
  visible bits -> A16

hidden_only:
  visible bits -> hidden -> A16
  no direct visible->A16 highway

overlay_locked:
  D21A direct highway + hidden residual paths
```

The proof uses a tied linear effective encoder:

```text
effective_encoder = direct + code_from_hidden @ hidden_from_visible
decode = effective_encoder.T
```

This keeps the test focused on hidden sparse topology. It does not yet test a
nonlinear hidden activation.

Decision criteria:

```text
exact byte reconstruction
minimum byte margin
ASCII geometry
direct edge count
hidden in/out edge count
effective copy penalty
single-edge-drop robustness
```

Current intended interpretation:

```text
D21I_HIDDEN_SPARSE_A_PASS:
  hidden-only exact candidate beats IO-only geometry without losing margin

D21I_HIDDEN_BIT_GAIN_PASS:
  hidden-only exact candidate beats IO-only geometry, but mostly by
  copy-like bit amplification rather than by a fully non-copy manifold

D21I_HIDDEN_REPLICATES_IO_ONLY:
  hidden-only can reproduce D21A through hidden link locations, but does not
  improve the geometry

D21I_OVERLAY_HIDDEN_GEOMETRY_LEAD:
  hidden residual paths improve geometry, but still need the D21A direct highway

D21I_IO_ONLY_STILL_BEST:
  extra hidden locations do not beat the current D21A A-block
```

Generated output remains under `output/` and is not committed.

Current bounded main result:

```text
D21I_HIDDEN_BIT_GAIN_PASS

best hidden candidate:
  arm: hidden_only_h16
  exact_byte_acc: 1.0
  byte_margin_min: +4.0
  ascii_class_geometry: 0.7308
  direct_edge_count: 0
  hidden_in_edge_count: 9
  hidden_out_edge_count: 17
  structural_edge_count: 26

baseline D21A:
  exact_byte_acc: 1.0
  byte_margin_min: +4.0
  ascii_class_geometry: 0.6686
  direct_edge_count: 16
```

Interpretation: extra hidden link locations can produce an exact no-direct
A-block with better ASCII geometry. The current best improvement is still
copy-like: it mostly amplifies one ASCII-relevant bit through a hidden path.
This is useful as a hidden-architecture lead, but not yet a fully natural
non-copy A manifold.
