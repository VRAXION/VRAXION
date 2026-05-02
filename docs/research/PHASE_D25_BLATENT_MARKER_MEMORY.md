# Phase D25 B-Latent Marker Memory Probe

Date: 2026-05-03

## Summary

D25 lifts the D21E multi-slot memory idea from byte payloads to full B64
8-byte-window payloads:

```text
STORE_SLOT_A, PAYLOAD_WINDOW_A
STORE_SLOT_B, PAYLOAD_WINDOW_B
distractor windows...
QUERY_SLOT_B
  -> PAYLOAD_WINDOW_B
```

This is the first B-level state/query probe on top of the stable AB codec.

Implementation:

```text
tools/_scratch/d25_blatent_marker_memory_probe.py
```

Stable substrate:

```text
tools/ab_window_codec.py
tools/ab_window_codec_v1.json
```

## Result

Verdict:

```text
D25_BLATENT_MEMORY_PASS
```

Confirm run:

```text
slot_counts: 2,4
distractor_lengths: 1,2,4,8,16,32
eval_sequences: 32768
```

Slot results:

```text
2-slot:
  query_window_exact_acc: 100%
  query_byte_exact_acc: 100%
  query_bit_acc: 100%
  query_byte_margin_min: +2.0
  state_dim: 128
  memory_edge_count: 128
  wrong_slot_recall_rate: 0%

4-slot:
  query_window_exact_acc: 100%
  query_byte_exact_acc: 100%
  query_bit_acc: 100%
  query_byte_margin_min: +2.0
  state_dim: 256
  memory_edge_count: 256
  wrong_slot_recall_rate: 0%
```

Controls:

```text
reset_state_control: 0% exact
time_shuffle_control: 0% exact
random_state_control: 0% exact
marker_shuffle_control: 0% exact
query_shuffle_control: 0% exact
```

Baseline:

```text
prev_window_baseline:
  2-slot exact: 50%
  4-slot exact: 25%
```

Interpretation: the previous-window baseline only hits the fixed-slot chance
rate. It is not query-addressed memory.

## Important Detail

D25's exact B64 memory is minimal, not redundant:

```text
one state lane per B64 payload lane per slot
2 slots -> 128 state lanes / edges
4 slots -> 256 state lanes / edges
```

This is expected for exact full-window recall. Any future compression is a
separate D25b/D26 question.

## What This Means

D24 established exact stateless B64 transforms:

```text
B64 -> sparse transform -> B64
```

D25 establishes addressed B64 memory:

```text
B64 payload window -> slot state -> query-selected B64 output
```

The B layer is now both:

```text
stateless transform surface
stateful key-value memory surface
```

This is still component-level progress, not a release model.

## Commands

Smoke:

```powershell
python tools\_scratch\d25_blatent_marker_memory_probe.py --mode smoke --slot-counts 2 --distractor-lengths 1,2 --eval-sequences 4096 --out output\phase_d25_blatent_marker_memory_probe_20260503\smoke
```

Oracle:

```powershell
python tools\_scratch\d25_blatent_marker_memory_probe.py --mode oracle --slot-counts 2,4 --distractor-lengths 1,2,4,8,16 --eval-sequences 65536 --out output\phase_d25_blatent_marker_memory_probe_20260503\oracle
```

Confirm:

```powershell
python tools\_scratch\d25_blatent_marker_memory_probe.py --mode confirm --candidates output\phase_d25_blatent_marker_memory_probe_20260503\crystallize\memory_candidates.csv --top-k 16 --slot-counts 2,4 --distractor-lengths 1,2,4,8,16,32 --eval-sequences 32768 --out output\phase_d25_blatent_marker_memory_probe_20260503\confirm
```

## Tests

Passed:

```text
python -m py_compile tools\_scratch\d25_blatent_marker_memory_probe.py
python -m py_compile tools\ab_window_codec.py
python tools\ab_window_codec.py --verify-artifact tools\ab_window_codec_v1.json
python tools\check_public_surface.py
D25 smoke
D25 oracle
D25 crystallize-memory
D25 confirm
```

Generated `output/` remains uncommitted.

## Next Step

D26 should combine D24 transforms with D25 memory:

```text
store payload window X
query asks for reverse(X), rotate(X), or bit_not(X)
```

That moves from "remember a window" to "remember and transform a queried
window."
