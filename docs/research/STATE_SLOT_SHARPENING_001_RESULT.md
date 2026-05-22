# STATE_SLOT_SHARPENING_001 Result

## Goal

Find whether sharper predicted state slots can make the hard/deterministic bottleneck beat direct GRU.

## Run

Valid slice:

```text
seeds=2026,2027
train_examples=1024
eval_examples=1024
epochs=30
jobs=19
heartbeat_sec=15
```

No-black-box audit:

```text
metrics.jsonl rows: 28
progress.jsonl rows: 75
job_progress files: 28
```

This is a valid slice, not the full 10-seed run.

## Arm Summary

| Arm | Answer | Det | Soft | Hard | Count Slot | State Slot | Same-token | Order |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `COUNT_BY_TYPE_ONLY` | `0.502` | `0.502` | `0.228` | `0.250` | `0.708` | `0.740` | `0.250` | `0.250` |
| `COUNT_WEIGHT_0P5` | `0.474` | `0.474` | `0.472` | `0.462` | `0.693` | `0.844` | `0.500` | `0.500` |
| `COUNT_WEIGHT_1P0` | `0.508` | `0.508` | `0.487` | `0.486` | `0.700` | `0.837` | `0.750` | `0.750` |
| `COUNT_WEIGHT_2P0` | `0.565` | `0.565` | `0.525` | `0.535` | `0.713` | `0.804` | `0.750` | `0.750` |
| `COUNT_WEIGHT_4P0` | `0.592` | `0.592` | `0.533` | `0.560` | `0.720` | `0.799` | `0.750` | `0.750` |
| `COUNT_WEIGHT_8P0` | `0.572` | `0.572` | `0.531` | `0.551` | `0.733` | `0.794` | `1.000` | `1.000` |
| `DISCRETE_SLOT_PRESSURE` | `0.500` | `0.500` | `0.487` | `0.480` | `0.699` | `0.836` | `0.750` | `0.750` |
| `FULL_STATE_STRONG` | `0.495` | `0.495` | `0.444` | `0.448` | `0.724` | `0.780` | `0.500` | `0.500` |
| `GRU_DIRECT_ANSWER` | `0.686` | `nan` | `nan` | `nan` | `nan` | `nan` | `1.000` | `1.000` |
| `LIFECYCLE_ONLY` | `0.221` | `0.221` | `0.302` | `0.269` | `0.182` | `0.674` | `0.250` | `0.250` |
| `ORACLE_STATE_VISIBLE` | `1.000` | `1.000` | `nan` | `nan` | `1.000` | `1.000` | `1.000` | `1.000` |
| `PER_EVENT_STATE_SUPERVISION` | `0.532` | `0.532` | `0.481` | `0.466` | `0.704` | `0.855` | `0.500` | `0.500` |
| `SHUFFLED_STATE_STRONG` | `0.114` | `0.114` | `0.445` | `0.445` | `0.638` | `0.466` | `0.000` | `0.000` |
| `STATE_BOTTLENECK_BASE` | `0.508` | `0.508` | `0.487` | `0.486` | `0.700` | `0.837` | `0.750` | `0.750` |

## Verdict

```json
[
  "STILL_EXPLICIT_LEDGER_REQUIRED"
]
```

## Diagnosis

Best deterministic bottleneck arm:

```text
COUNT_WEIGHT_4P0 = 0.592
```

Baseline:

```text
GRU_DIRECT_ANSWER = 0.686
```

So count-weight sharpening helped relative to the base bottleneck, but it did not make the hard/deterministic state path competitive with direct GRU.

Important observations:

```text
Oracle state visible still passes at 1.000.
Shuffled strong state collapses to 0.114 deterministic accuracy.
Count slots remain around 0.70-0.73, far below the 0.92 gate.
Per-event supervision improves state/flag quality but not enough answer quality.
Entropy/discrete pressure did not fix the count slot.
```

Main blocker:

```text
raw text -> clean count/state slots is still not solved.
```

This means the next step should not be `MIN_PRISMION_CELL_001` yet. The bottleneck state itself is still too noisy.

## Claim Boundary

Controlled toy state-update domain only. This does not prove open-ended grounding or consciousness.

