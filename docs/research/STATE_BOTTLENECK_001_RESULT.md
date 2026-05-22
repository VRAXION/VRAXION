# STATE_BOTTLENECK_001 Result

## Goal

Test whether a hard-audited predicted-state bottleneck improves story state tracking over direct answer paths.

## Run

Valid slice command:

```powershell
python scripts\probes\run_state_bottleneck_probe.py ^
  --out target\pilot_wave\state_bottleneck_001\valid_slice_2026_05_15 ^
  --seeds 2026-2027 ^
  --arms EXPLICIT_LEDGER_ORACLE,ORACLE_STATE_VISIBLE,BAG_OF_TOKENS_MLP,STATIC_POSITION_MLP,GRU_DIRECT_ANSWER,GRU_STATE_BOTTLENECK,NEURAL_SLOT_BOTTLENECK,SHUFFLED_STATE_BOTTLENECK ^
  --supervision-modes final_state_only,per_event_state_supervision ^
  --train-examples 2048 ^
  --eval-examples 2048 ^
  --epochs 80 ^
  --jobs 4 ^
  --heartbeat-seconds 15
```

This is a valid decision slice, not the full 10-seed run. The earlier long full attempt is discarded because it did not satisfy the no-black-box-run rule after crashing before writing metrics.

No-black-box audit for this run:

```text
metrics.jsonl rows: 22
progress.jsonl rows: 149
job_progress files: one per started job
parent heartbeat cadence: 15 seconds while jobs were pending
worker progress: epoch rows plus heartbeat batch rows
```

## Arm Summary

| Arm/Mode | Det Answer | Soft | Hard | Same-token | Order | State Slot | Count Slot | Soft-Hard Gap | Det Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `BAG_OF_TOKENS_MLP/baseline` | `0.654` | `nan` | `nan` | `0.500` | `0.500` | `nan` | `nan` | `nan` | `nan` |
| `EXPLICIT_LEDGER_ORACLE/baseline` | `1.000` | `nan` | `nan` | `1.000` | `1.000` | `1.000` | `1.000` | `nan` | `0.000` |
| `GRU_DIRECT_ANSWER/baseline` | `0.718` | `nan` | `nan` | `0.500` | `0.500` | `nan` | `nan` | `nan` | `nan` |
| `GRU_STATE_BOTTLENECK/final_state_only` | `0.700` | `0.740` | `0.591` | `0.750` | `0.750` | `0.872` | `0.727` | `0.149` | `0.040` |
| `GRU_STATE_BOTTLENECK/per_event_state_supervision` | `0.703` | `0.727` | `0.637` | `0.750` | `0.750` | `0.923` | `0.879` | `0.089` | `0.024` |
| `NEURAL_SLOT_BOTTLENECK/final_state_only` | `0.671` | `0.713` | `0.610` | `0.500` | `0.500` | `0.856` | `0.716` | `0.104` | `0.042` |
| `NEURAL_SLOT_BOTTLENECK/per_event_state_supervision` | `0.655` | `0.695` | `0.617` | `0.500` | `0.500` | `0.911` | `0.831` | `0.077` | `0.040` |
| `ORACLE_STATE_VISIBLE/baseline` | `1.000` | `nan` | `nan` | `1.000` | `1.000` | `1.000` | `1.000` | `nan` | `0.000` |
| `SHUFFLED_STATE_BOTTLENECK/final_state_only` | `0.195` | `0.643` | `0.541` | `0.250` | `0.250` | `0.429` | `0.554` | `0.102` | `0.448` |
| `SHUFFLED_STATE_BOTTLENECK/per_event_state_supervision` | `0.199` | `0.659` | `0.511` | `0.250` | `0.250` | `0.437` | `0.583` | `0.148` | `0.460` |
| `STATIC_POSITION_MLP/baseline` | `0.586` | `nan` | `nan` | `0.750` | `0.750` | `nan` | `nan` | `nan` | `nan` |

## Verdict

```json
[
  "BOTTLENECK_PARTIAL"
]
```

## Interpretation

Read bottleneck results through the hard and deterministic decoders. A soft-state answer win is not sufficient because soft probabilities can act as a covert channel.

The bottleneck helped with state structure but did not yet solve the task:

```text
GRU_DIRECT_ANSWER answer accuracy: 0.718
GRU_STATE_BOTTLENECK deterministic answer: 0.700-0.703
GRU_STATE_BOTTLENECK state slot accuracy: 0.872-0.923
GRU_STATE_BOTTLENECK count slot accuracy: 0.727-0.879
GRU_STATE_BOTTLENECK same-token/order controls: 0.750
SHUFFLED deterministic answer: ~0.20
```

So the state bottleneck is causally meaningful, because the shuffled state teacher collapses under deterministic decoding. But it is not yet a strong positive, because deterministic answer accuracy does not beat direct GRU by the required margin.

Per-event state supervision is the useful variant so far: it raises state slot accuracy and count slot accuracy, but the learned state is still not clean enough to turn into a better answer than the direct route.

Main diagnosis:

```text
The state schema is sufficient: oracle state visible = 1.000.
The learned bottleneck is not yet sufficient: count slots remain imperfect.
The next blocker is learned state extraction, not the deterministic decoder.
```

## Claim Boundary

Controlled toy state-update domain only. This does not prove open-ended natural-language grounding or consciousness.
