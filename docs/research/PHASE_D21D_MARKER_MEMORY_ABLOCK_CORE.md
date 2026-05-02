# Phase D21D Marker-Memory A-Block Core

Date: 2026-05-02

## Summary

D21D extends the D21A/D21B/D21C A-block line from one-step memory to delayed
marker memory:

```text
input:  MARKER, PAYLOAD, distractor..., QUERY
target: PAYLOAD at QUERY
```

This is the first A-block microtask where the core must store a payload after a
marker, keep it through unrelated distractors, and emit it only when the query
byte appears.

Implementation:

```text
tools/_scratch/d21d_marker_memory_ablock_core.py
```

Fixed task constants:

```text
marker_byte = 0xF0
query_byte  = 0x0F
payload bytes exclude marker/query
strict distractors exclude marker/query/payload
```

## Result

Verdict:

```text
D21D_MARKER_MEMORY_PASS
```

Fresh oracle check:

```text
mode: marker-memory-oracle
eval_sequences: 65536
distractor_lengths: 1,2,4,8,16
query_payload_exact_acc: 1.0
query_payload_bit_acc: 1.0
query_payload_margin_min: +12.0
payload_state_collision_count: 0
non_query_byte_reconstruction_acc: 1.0
reset_state_acc: 0.0
time_shuffle_state_acc: 0.0
marker_shuffle_acc: 0.0
query_removed_false_positive_rate: 0.0
prev_byte_baseline_acc: 0.0
current_byte_cheat_rate: 0.0
```

Bounded confirm:

```text
source: output/phase_d21d_marker_memory_ablock_core/atlas/memory_candidates.csv
top_k: 16
eval_sequences: 32768
distractor_lengths: 1,2,4,8,16,32
pass_count: 12 / 16
best verdict: D21D_MARKER_MEMORY_PASS
```

Best high-margin confirmed core:

```text
state_dim: 32
memory_edge_count: 16
query_payload_exact_acc: 1.0
query_payload_bit_acc: 1.0
query_payload_margin_min: +12.0
all_distractor_lengths_pass: true
long_sequence_payload_acc: 1.0
payload_state_collision_count: 0
non_query_byte_reconstruction_acc: 1.0
zero_context_byte_reconstruction_acc: 1.0

reset_state_acc: 0.0
time_shuffle_state_acc: 0.0
random_state_acc: 0.003082
marker_shuffle_acc: 0.0
query_removed_false_positive_rate: 0.0
prev_byte_baseline_acc: 0.0
current_byte_cheat_rate: 0.0
```

High-margin memory entries:

```text
0:0:1 1:1:1 2:2:1 3:3:1 4:4:1 5:5:1 6:6:1 7:7:1
8:0:-1 9:1:-1 10:2:-1 11:3:-1 12:4:-1 13:5:-1 14:6:-1 15:7:-1
```

Compact confirmed family:

```text
identity_memory, 8 edges
query_payload_exact_acc: 1.0
non_query_byte_reconstruction_acc: 1.0
random-state control: ~0.0034 to 0.0041
query_payload_margin_min: +4.0
verdict: D21D_MARKER_MEMORY_PASS
```

## Atlas And Rescope

The original full atlas/crystallize shape was too slow for the current machine.
The run was intentionally rescoped rather than left to spin:

```text
atlas:
  state_dims: 16,32,64
  memory_edge_budgets: 16,32
  samples: 8
  eval_sequences: 2048

confirm:
  eval_sequences: 32768
  distractor_lengths: 1,2,4,8,16,32
```

Atlas heatmap:

```text
D21D memory heatmap: brighter = query_exact * (1-control_max)
legend: PASS=P WEAK=W ARTIFACT=F NONE=.
state\edge    16    32
     16 :   P =   P
     32 =   P     P
     64 @   P =   P
```

Crystallize note:

```text
low-budget crystallize found 7-edge/8-edge path candidates in the scout loop,
but full confirm rejected the 7/8/9-edge path candidates.

confirmed crystallize-path passes start at 10+ edges; the separately generated
8-edge identity-memory family from the atlas/confirm does pass.
```

## Why This Matters

D21D proves a second memory primitive behind the A-block shell:

```text
D21A: exact byte IO
D21B: context write lane
D21C: one-step recurrent state
D21D: marker/payload delayed recall
```

Plain-language meaning:

```text
The core can hear "remember this byte",
carry it through noise,
and answer it later when queried.
```

This is still scratch/prototype work, not a release checkpoint. It does not
unlock H512/H8192 by itself. It does show that the A-block line can support a
real delayed-recall primitive with adversarial controls.

## Commands

Baseline:

```powershell
python tools\_scratch\d21d_marker_memory_ablock_core.py --mode baseline-check --out output\phase_d21d_marker_memory_ablock_core\baseline
```

Fresh oracle:

```powershell
python tools\_scratch\d21d_marker_memory_ablock_core.py --mode marker-memory-oracle --distractor-lengths 1,2,4,8,16 --eval-sequences 65536 --out output\phase_d21d_marker_memory_ablock_core\oracle_final
```

Bounded atlas:

```powershell
python tools\_scratch\d21d_marker_memory_ablock_core.py --mode memory-atlas --state-dims 16,32,64 --memory-edge-budgets 16,32 --distractor-lengths 1,2,4,8,16 --samples 8 --eval-sequences 2048 --workers 10 --out output\phase_d21d_marker_memory_ablock_core\atlas
```

Bounded confirm:

```powershell
python tools\_scratch\d21d_marker_memory_ablock_core.py --mode confirm --candidates output\phase_d21d_marker_memory_ablock_core\atlas\memory_candidates.csv --top-k 16 --distractor-lengths 1,2,4,8,16,32 --eval-sequences 32768 --out output\phase_d21d_marker_memory_ablock_core\confirm
```

Crystallize-path confirm:

```powershell
python tools\_scratch\d21d_marker_memory_ablock_core.py --mode confirm --candidates output\phase_d21d_marker_memory_ablock_core\crystallize\memory_paths.csv --top-k 16 --distractor-lengths 1,2,4,8,16,32 --eval-sequences 32768 --out output\phase_d21d_marker_memory_ablock_core\confirm_path
```

## Tests

Passed:

```text
python -m py_compile tools\_scratch\d21d_marker_memory_ablock_core.py
python tools\check_public_surface.py
git diff --check
D21D baseline-check
D21D marker-memory-oracle
D21D bounded memory-atlas
D21D bounded confirm
D21D crystallize-path confirm
```

Generated `output/` remains uncommitted.

## Next Step

D21E should test whether this remains clean when the task needs more than one
stored item:

```text
multiple slots or key-value memory
MARKER_A, PAYLOAD_A, MARKER_B, PAYLOAD_B, ..., QUERY_A -> PAYLOAD_A
```

Acceptance should keep the same adversarial controls:

```text
reset state fails
time-shuffled state fails
random state fails
marker-shuffled sequence fails
query-removed false positives stay near zero
non-query byte reconstruction remains exact
```
