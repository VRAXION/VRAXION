# Phase D21C Tiny Recurrent A-Block Core

Date: 2026-05-02

## Summary

D21C attaches a tiny state/core block behind the D21A/D21B A-block shell.

Previous state:

```text
D21A: byte -> abstract -> byte is exact
D21B: context lane can steer byte output; fake context does not
```

D21C task:

```text
input sequence:  [A, B]
target output:   previous byte A when B arrives
```

This is deliberately small, but it requires state. Reset, time-shuffled state,
or random state should not solve it.

Implementation:

```text
tools/_scratch/d21c_tiny_recurrent_ablock_core.py
```

## Result

Verdict:

```text
D21C_PREV_BYTE_CORE_PASS
```

Best high-margin confirmed core:

```text
state_dim: 16
core_edges: 16
prev_byte_exact_acc: 1.0
prev_byte_bit_acc: 1.0
prev_byte_margin_min: +12.0
long_sequence_exact_acc: 1.0
state_collision_count: 0

reset_each_token_acc: 0.003906
time_shuffle_state_acc: 0.004105
random_state_acc: 0.003967
current_byte_cheat_rate: 0.0
zero_context_byte_reconstruction_acc: 1.0
```

Confirmed high-margin core entries:

```text
0:0:1 1:1:1 2:2:1 3:3:1 4:4:1 5:5:1 6:6:1 7:7:1
8:0:-1 9:1:-1 10:2:-1 11:3:-1 12:4:-1 13:5:-1 14:6:-1 15:7:-1
```

Crystallized compact core:

```text
state_dim: 16
core_edges: 8
prev_byte_exact_acc: 1.0
long_sequence_exact_acc: 1.0
prev_byte_margin_min: +4.0
reset_each_token_acc: 0.003906
time_shuffle_state_acc: 0.003738
random_state_acc: 0.003769
current_byte_cheat_rate: 0.0
zero_context_byte_reconstruction_acc: 1.0
```

The compact 8-edge core is enough for exact previous-byte memory; the 16-edge
core keeps a wider output margin.

## Atlas

Bounded atlas:

```text
D21C core heatmap: brighter = pair_exact * long_sequence_exact
legend: PASS=P WEAK=W ARTIFACT=F NONE=.
state\edge     8    16    24    32
         8     P     P     P     P
        16     P     P     P     P
        32     P     P     P     P
```

Interpretation: previous-byte state storage is locally easy once the D21A/D21B
A-block shell exists. The core does not need a large hidden space for this
first memory primitive.

## Why This Matters

D21C proves the first complete micro-path:

```text
byte input
-> fixed reciprocal byte lane
-> tiny recurrent state
-> context lane
-> previous-byte output
```

This is not a release model. It is the smallest validated state-carrying
component behind the A-block. It proves that the context lane can be driven by a
core, not only by an externally supplied oracle context.

## Commands

Baseline:

```powershell
python tools\_scratch\d21c_tiny_recurrent_ablock_core.py --mode baseline-check --out output\phase_d21c_tiny_recurrent_ablock_core\baseline
```

Oracle:

```powershell
python tools\_scratch\d21c_tiny_recurrent_ablock_core.py --mode prev-byte-oracle --eval-pairs 65536 --sequence-len 16 --out output\phase_d21c_tiny_recurrent_ablock_core\oracle
```

Bounded atlas:

```powershell
python tools\_scratch\d21c_tiny_recurrent_ablock_core.py --mode core-atlas --state-dims 8,16,32 --core-edge-budgets 8,16,24,32 --samples 128 --eval-pairs 4096 --sequence-len 16 --sequence-count 256 --workers 10 --out output\phase_d21c_tiny_recurrent_ablock_core\atlas
```

Crystallize:

```powershell
python tools\_scratch\d21c_tiny_recurrent_ablock_core.py --mode crystallize-core --state-dim 16 --start-family oracle_prev_byte --max-steps 5000 --workers 10 --eval-pairs 65536 --sequence-len 16 --sequence-count 512 --out output\phase_d21c_tiny_recurrent_ablock_core\crystallize
```

Confirm:

```powershell
python tools\_scratch\d21c_tiny_recurrent_ablock_core.py --mode confirm --candidates output\phase_d21c_tiny_recurrent_ablock_core\crystallize\core_candidates.csv --top-k 16 --eval-pairs 65536 --sequence-len 32 --sequence-count 4096 --out output\phase_d21c_tiny_recurrent_ablock_core\confirm
```

Compact path confirm:

```powershell
python tools\_scratch\d21c_tiny_recurrent_ablock_core.py --mode confirm --candidates output\phase_d21c_tiny_recurrent_ablock_core\crystallize\core_paths.csv --top-k 16 --eval-pairs 65536 --sequence-len 32 --sequence-count 4096 --out output\phase_d21c_tiny_recurrent_ablock_core\confirm_path
```

## Tests

Passed:

```text
python -m py_compile tools\_scratch\d21c_tiny_recurrent_ablock_core.py
python tools\check_public_surface.py
D21C baseline-check
D21C prev-byte-oracle
D21C bounded core-atlas
D21C crystallize-core
D21C full-pair confirm
D21C compact path confirm
```

Generated `output/` remains uncommitted.

## Next Step

D21D should move from one-step memory to a marker-memory task:

```text
input:  marker byte M, payload byte P, distractors...
target: recall P when query marker appears
```

D21D should keep the same adversarial controls:

```text
reset state fails
time-shuffled state fails
random state fails
current-byte echo fails
zero-context byte reconstruction remains exact
```
