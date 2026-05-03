# Phase D21F A-Block Natural Geometry

Date: 2026-05-03

## Summary

D21F tested whether the A-block can become a more natural sparse character
space instead of only a lossless byte copy codec.

Question:

```text
Can byte -> A16 -> byte stay exact
while A16 also makes ASCII-near things closer
and reduces identity/copy structure?
```

Implementation:

```text
tools/_scratch/d21f_ablock_natural_geometry.py
docs/playground/ablock_circuit.html
```

## Main Verdict

```text
D21F_A_NATURAL_WEAK_PASS
```

The run found natural-geometry signals, but not a deploy-safe A_v2 replacement.

Best practical candidate:

```text
arm: overlay_locked
exact_byte_acc: 1.0
bit_acc: 1.0
byte_margin_min: +2.0
ascii_class_geometry: 0.700994
identity_copy_penalty: 0.391667
single_edge_drop_mean_bit: 0.994466
```

Best geometry lead:

```text
arm: no_prefill
exact_byte_acc: 1.0
bit_acc: 1.0
byte_margin_min: +0.125
ascii_class_geometry: 0.802174
identity_copy_penalty: 0.0
```

Baseline:

```text
arm: baseline16
exact_byte_acc: 1.0
bit_acc: 1.0
byte_margin_min: +4.0
ascii_class_geometry: 0.668637
identity_copy_penalty: 1.0
single_edge_drop_mean_bit: 1.0
```

## Interpretation

D21F proves the A-block can be made less copy-like while still reconstructing
all 256 bytes exactly. It does not prove that the natural geometry is robust
enough to replace the deploy A-block.

The current decision:

```text
Deploy A remains A_v1 redundant_copy_2x.
D21F candidates are research leads only.
```

Reason:

```text
overlay_locked improves geometry and copy penalty,
but margin drops from +4.0 to +2.0 and edge-drop exactness drops.

no_prefill is much more natural/non-copy,
but byte margin is only +0.125.
```

## Visual Playground

The A-block circuit playground now exposes candidate switching:

```text
A_v1 redundant copy baseline
D21F overlay_locked weak candidate
D21F no_prefill geometry lead
```

URL after GitHub Pages refresh:

```text
https://vraxion.github.io/VRAXION/playground/ablock_circuit.html
```

## Next Step

Do not replace AB codec v1 yet.

Next useful step is D21G:

```text
margin-aware natural A search

objective:
  exact roundtrip
  byte_margin_min >= +2.0, preferably +4.0
  geometry > baseline
  identity_copy_penalty lower
  edge-drop robustness close to baseline
```

If D21G fails, keep A as a robust byte codec and push natural/semantic geometry
into C/D layers instead of forcing it into the byte codec.
