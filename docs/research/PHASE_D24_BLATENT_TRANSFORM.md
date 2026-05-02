# Phase D24 B-Latent Transformation Probe

Date: 2026-05-03

## Summary

D24 tests whether the stable AB codec's `64D` B latent can be used as a
working surface, not only as a reversible codec:

```text
B64 input -> sparse transform core -> B64 output -> decoded 8-byte window
```

Implementation:

```text
tools/_scratch/d24_blatent_transform_probe.py
```

Stable substrate:

```text
tools/ab_window_codec.py
tools/ab_window_codec_v1.json
```

## Result

Verdict:

```text
D24_BLATENT_TRANSFORM_PASS
```

Main run:

```text
eval_windows: 65536
control_repeats: 8
tasks: copy, reverse, rotate_left, rotate_right, bit_not
```

Primary task result:

```text
copy:         exact 100%, margin +2.0, random control 0%
reverse:      exact 100%, margin +2.0, random control 0%
rotate_left:  exact 100%, margin +2.0, random control 0%
rotate_right: exact 100%, margin +2.0, random control 0%
bit_not:      exact 100%, margin +2.0, random control 0%
```

Shared metrics:

```text
byte_exact_acc: 100%
bit_acc: 100%
b_output_collision_count: 0
transform_edge_count: 64
input_output_hamming_consistency: 1.0
position_shuffle_control_acc: 0.0001220703125
random_sparse_control_acc: 0.0
```

## Robustness Note

The exact transform is a minimal sparse signed/permutation core with one edge
per B lane:

```text
transform_edge_count: 64
```

Single-edge drop is intentionally fragile:

```text
copy/reverse/rotate single_edge_drop_mean_exact: ~0.486
bit_not single_edge_drop_mean_exact: ~0.514
single_edge_drop_mean_bit: ~0.992
```

Interpretation: D24 proves exact symbolic transforms over B64, but the v1
transform core is not redundant. Redundancy can be added later if the transform
artifact needs fault tolerance.

## What This Means

D23/AB established:

```text
8 bytes <-> B64
```

D24 establishes:

```text
B64 -> exact sparse transform -> B64
```

So the B layer is now a usable working surface for exact stateless transforms:

```text
A-block: byte IO
AB codec: 8-byte window <-> B64
D24: B64 can perform exact byte-window transforms
```

This is still component-level progress, not a release model.

## Commands

Smoke:

```powershell
python tools\_scratch\d24_blatent_transform_probe.py --mode smoke --tasks copy,reverse,rotate_left,rotate_right --eval-windows 4096 --out output\phase_d24_blatent_transform_probe_20260503\smoke
```

Main:

```powershell
python tools\_scratch\d24_blatent_transform_probe.py --mode main --tasks copy,reverse,rotate_left,rotate_right,bit_not --eval-windows 65536 --control-repeats 8 --out output\phase_d24_blatent_transform_probe_20260503\main
```

Crystallize/export:

```powershell
python tools\_scratch\d24_blatent_transform_probe.py --mode crystallize --tasks copy,reverse,rotate_left,rotate_right --eval-windows 65536 --out output\phase_d24_blatent_transform_probe_20260503\crystallize
```

## Tests

Passed:

```text
python -m py_compile tools\_scratch\d24_blatent_transform_probe.py
python -m py_compile tools\ab_window_codec.py
python tools\ab_window_codec.py --verify-artifact tools\ab_window_codec_v1.json
python tools\check_public_surface.py
D24 smoke
D24 main
D24 crystallize/export
```

Generated `output/` remains uncommitted.

## Next Step

D24B/D25 should add state/query semantics on top of B64:

```text
marker-copy within an 8-byte B-window
key-value recall over B64 windows
integrate D21E multi-slot memory with B-level transforms
```
