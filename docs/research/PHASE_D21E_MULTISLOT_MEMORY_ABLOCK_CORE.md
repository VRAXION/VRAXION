# Phase D21E Multi-Slot Memory A-Block Core

Date: 2026-05-02

## Summary

D21E extends D21D from one delayed memory slot to slot-addressed key-value
memory:

```text
MARKER_A -> PAYLOAD_A
MARKER_B -> PAYLOAD_B
MARKER_C -> PAYLOAD_C
MARKER_D -> PAYLOAD_D
distractor...
QUERY_B -> PAYLOAD_B
QUERY_A -> PAYLOAD_A
```

Implementation:

```text
tools/_scratch/d21e_multislot_memory_ablock_core.py
tools/_scratch/d21_medium_horizon_autopilot.py
```

D21E still uses the fixed D21 A-block shell:

```text
D21A byte IO
D21B context write lane
D21C recurrent state path
D21D marker-memory recall
```

## Result

Verdict:

```text
D21E_MULTISLOT_MEMORY_PASS
```

Oracle:

```text
slot_counts: 2,4
distractor_lengths: 1,2,4,8,16
eval_sequences: 65536
state_dim: 64
memory_edges: 64
query_payload_exact_acc: 1.0
query_shuffle_acc: 0.0
wrong_slot_recall_rate: 0.0
verdict: D21E_MULTISLOT_MEMORY_PASS
```

Bounded atlas:

```text
state_dims: 32,64,128
slot_counts: 2,4
memory_edge_budgets: 16,32,64,96
samples: 8
eval_sequences: 2048
```

Atlas result:

```text
D21E memory heatmap: brighter = query_exact * (1-control_max)
legend: PASS=P 2SLOT=2 WEAK=W ARTIFACT=F NONE=.
state\edge    16    32    64    96
        32     .     .     .     .
        64     . @   P @   P @   P
       128     . @   P @   P @   P
```

Bounded confirm:

```text
source: atlas/memory_candidates.csv
top_k: 16
slot_counts: 2,4
distractor_lengths: 1,2,4,8,16,32
eval_sequences: 32768
pass_count: 10 / 16
best verdict: D21E_MULTISLOT_MEMORY_PASS
```

Best high-margin confirmed core:

```text
state_dim: 64
slot_count: 4
memory_edge_count: 64
query_payload_exact_acc: 1.0
query_payload_bit_acc: 1.0
query_payload_margin_min: +12.0
all_slot_counts_pass: true
all_distractor_lengths_pass: true
long_sequence_payload_acc: 1.0
slot_state_collision_count: 0
non_query_byte_reconstruction_acc: 1.0
zero_context_byte_reconstruction_acc: 1.0

reset_state_acc: 0.0
time_shuffle_state_acc: 0.0
random_state_acc: 0.001862
marker_shuffle_acc: 0.0
query_shuffle_acc: 0.0
query_removed_false_positive_rate: 0.0
prev_byte_baseline_acc: 0.0
current_byte_cheat_rate: 0.0
wrong_slot_recall_rate: 0.0
```

## Crystallize

A bounded crystallize pass started from the 64-edge oracle core and pruned while
preserving all D21E gates.

Result:

```text
start: 64 edges
confirmed compact path: 32 edges
query_payload_exact_acc: 1.0
query_payload_margin_min: +4.0
verdict: D21E_MULTISLOT_MEMORY_PASS
```

The 32-edge core is the compact confirmed candidate. The 64-edge core remains
the high-margin reference.

## Rescope Note

The original atlas target was `samples=256` and the original confirm target was
`eval_sequences=131072`. The full atlas was too slow on the current machine, so
the run was intentionally rescoped:

```text
atlas: samples=8, eval_sequences=2048
confirm: eval_sequences=32768
```

This is bounded component evidence, not a release checkpoint and not an
H512/H8192 unlock.

## Why This Matters

D21E proves that the A-block line can hold multiple addressed memories:

```text
one-slot recall -> multi-slot key-value recall
```

Plain-language meaning:

```text
The core can remember several named byte values at once,
ignore distractors,
and return the value belonging to the queried name.
```

This is the first clean A-block working-memory primitive. The next step is D21F:
use multi-slot memory inside a tiny sequence reasoning task.

## Commands

Smoke:

```powershell
python tools\_scratch\d21e_multislot_memory_ablock_core.py --mode multislot-oracle --slot-counts 2 --distractor-lengths 1,2 --eval-sequences 4096 --out output\phase_d21e_multislot_memory_ablock_core_20260502\smoke
```

Oracle:

```powershell
python tools\_scratch\d21e_multislot_memory_ablock_core.py --mode multislot-oracle --slot-counts 2,4 --distractor-lengths 1,2,4,8,16 --eval-sequences 65536 --state-dim 64 --memory-edge-budget 64 --out output\phase_d21e_multislot_memory_ablock_core_20260502\oracle
```

Bounded atlas:

```powershell
python tools\_scratch\d21e_multislot_memory_ablock_core.py --mode memory-atlas --state-dims 32,64,128 --slot-counts 2,4 --memory-edge-budgets 16,32,64,96 --distractor-lengths 1,2,4,8,16 --samples 8 --eval-sequences 2048 --workers 10 --out output\phase_d21e_multislot_memory_ablock_core_20260502\atlas
```

Bounded confirm:

```powershell
python tools\_scratch\d21e_multislot_memory_ablock_core.py --mode confirm --candidates output\phase_d21e_multislot_memory_ablock_core_20260502\atlas\memory_candidates.csv --top-k 16 --slot-counts 2,4 --distractor-lengths 1,2,4,8,16,32 --eval-sequences 32768 --out output\phase_d21e_multislot_memory_ablock_core_20260502\confirm
```

Crystallize:

```powershell
python tools\_scratch\d21e_multislot_memory_ablock_core.py --mode crystallize-memory --state-dim 64 --slot-counts 2,4 --memory-edge-budget 64 --distractor-lengths 1,2,4,8,16 --max-steps 5000 --eval-sequences 2048 --out output\phase_d21e_multislot_memory_ablock_core_20260502\crystallize
```

Crystallize-path confirm:

```powershell
python tools\_scratch\d21e_multislot_memory_ablock_core.py --mode confirm --candidates output\phase_d21e_multislot_memory_ablock_core_20260502\crystallize\memory_paths.csv --top-k 16 --slot-counts 2,4 --distractor-lengths 1,2,4,8,16,32 --eval-sequences 32768 --out output\phase_d21e_multislot_memory_ablock_core_20260502\confirm_path
```

Autopilot smoke:

```powershell
python tools\_scratch\d21_medium_horizon_autopilot.py --max-phases 2 --eval-sequences 1024 --heartbeat-s 60 --out output\phase_d21_medium_horizon_autopilot_20260502\smoke
```

## Tests

Passed:

```text
python -m py_compile tools\_scratch\d21e_multislot_memory_ablock_core.py
python -m py_compile tools\_scratch\d21_medium_horizon_autopilot.py
python tools\check_public_surface.py
git diff --check
D21E smoke oracle
D21E oracle
D21E bounded atlas
D21E bounded confirm
D21E bounded crystallize
D21E crystallize-path confirm
D21 medium-horizon autopilot smoke
```

Generated `output/` remains uncommitted.

## Next Step

D21F should use the D21E multi-slot memory primitive in a tiny sequence
reasoning task:

```text
store multiple byte facts
query one fact
apply one simple rule or transform
return the transformed payload
```

The same controls should stay mandatory:

```text
reset state fails
time-shuffled state fails
marker/query shuffle fails
wrong-slot recall fails
query-removed false positives stay zero
non-query byte reconstruction remains exact
```
