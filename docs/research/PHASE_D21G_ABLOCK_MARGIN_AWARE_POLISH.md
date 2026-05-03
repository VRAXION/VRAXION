# Phase D21G A-Block Margin-Aware Natural Polish

Date: 2026-05-03

## Summary

D21G targeted the exact weakness in the D21F `no_prefill` A-block candidate:

```text
D21F no_prefill:
  exact byte roundtrip: 100%
  natural/non-copy geometry: strong
  byte_margin_min: +0.125
```

The goal was to keep the non-copy sparse character while raising the output
margin.

Implementation:

```text
tools/_scratch/d21g_ablock_margin_aware_polish.py
```

## Verdict

```text
D21G_MARGIN_NATURAL_PASS
```

Best natural candidate:

```text
arm: balanced_energy_2
exact_byte_acc: 1.0
bit_acc: 1.0
byte_argmax_acc: 1.0
byte_margin_min: +2.5
ascii_class_geometry: 0.764149
identity_copy_penalty: 0.0
ordered_copy_penalty: 0.0
edge_count: 28
single_edge_drop_mean_bit: 0.997768
single_edge_drop_mean_exact: 0.982143
```

Comparison:

```text
D21F no_prefill:
  margin:       +0.125
  geometry:      0.802174
  copy penalty:  0.0
  edge count:    9

D21G balanced_energy_2:
  margin:       +2.5
  geometry:      0.764149
  copy penalty:  0.0
  edge count:    28

A_v1 deploy baseline:
  margin:       +4.0
  geometry:      0.668637
  copy penalty:  1.0
  edge count:    16
```

## Interpretation

D21G shows that the natural-looking `no_prefill` path was not a dead end. We
can push the output margin from thin `+0.125` to a much safer `+2.5` while
keeping zero identity-copy penalty.

This is a stronger A_v2 research candidate than D21F. It is still not an
automatic deploy replacement for A_v1 because the A_v1 baseline remains simpler
and has the strongest margin/edge-drop behavior.

Current decision:

```text
Deploy AB codec remains A_v1.
D21G candidate becomes the latest mutated A-block research candidate.
```

The playground now defaults to:

```text
LATEST mutated: D21G margin-natural pass
```

## Next Step

The next useful A-block step is not more blind mutation. It is a focused output
curve view:

```text
for selected input byte:
  show top-N byte scores
  show correct-vs-runner-up margin
  show semantic probes: A/B, A/a, A/Z, A/7
```

That will make the margin and geometry tradeoff visible instead of only numeric.
