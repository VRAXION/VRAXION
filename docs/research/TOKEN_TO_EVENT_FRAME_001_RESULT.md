# TOKEN_TO_EVENT_FRAME_001 Result

## Goal

Test whether raw event clauses can be converted into hard event frames that feed an identity-aware deterministic ledger.

## Run

Smoke / valid slice:

```text
seeds=2026,2027
train_examples=1024
eval_examples=1024
epochs=30
jobs=20
heartbeat_sec=15
```

No-black-box audit:

```text
metrics.jsonl rows: 14
progress.jsonl rows: 37
job_progress files: 14
```

This is not the full 10-seed run. It is enough to diagnose the v1 setup because all learned frame arms nearly saturate and the shortcut controls saturate too.

## Arm Summary

| Arm | Frame Exact | Ledger | Event | Entity | Ref | Qty | Illegal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BAG_OF_TOKENS_EVENT_FRAME` | `0.999` | `0.999` | `1.000` | `0.999` | `1.000` | `1.000` | `0.000` |
| `DIRECT_STORY_GRU_ANSWER` | `nan` | `0.652` | `nan` | `nan` | `nan` | `nan` | `nan` |
| `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.999` | `0.998` | `1.000` | `0.999` | `1.000` | `1.000` | `0.000` |
| `SHUFFLED_EVENT_FRAME_TEACHER` | `0.078` | `0.270` | `0.545` | `0.131` | `0.545` | `0.575` | `0.000` |
| `STATIC_POSITION_EVENT_FRAME` | `0.999` | `0.997` | `1.000` | `0.999` | `1.000` | `1.000` | `0.000` |
| `STORY_TO_EVENT_FRAMES_GRU` | `0.999` | `0.997` | `1.000` | `0.999` | `1.000` | `1.000` | `0.000` |

## Verdict

```json
[
  "EVENT_FRAME_POSITIVE",
  "EVENT_FRAME_LEXICAL_SHORTCUT"
]
```

## Diagnosis

The event-frame route works mechanically:

```text
SEGMENTED_EVENT_FRAME_CLASSIFIER frame exact: 0.999
SEGMENTED_EVENT_FRAME_CLASSIFIER ledger answer: 0.998
STORY_TO_EVENT_FRAMES_GRU ledger answer: 0.997
SHUFFLED_EVENT_FRAME_TEACHER ledger answer: 0.270
DIRECT_STORY_GRU_ANSWER ledger answer: 0.652
```

This means hard predicted event frames can feed the deterministic identity ledger, and the shuffled teacher control collapses.

But the v1 dataset is too lexical/template-solvable:

```text
BAG_OF_TOKENS_EVENT_FRAME ledger answer: 0.999
STATIC_POSITION_EVENT_FRAME ledger answer: 0.997
```

So this is a positive mechanism check, not a strong parser/grounding claim.

Recommended next step:

```text
TOKEN_TO_EVENT_FRAME_002_HARD_CONTRASTS
```

Make clause-level bag/static insufficient by adding stricter order, role, and context contrasts before moving to a PrismionCell state updater.

## Claim Boundary

V1 assumes gold clause segmentation for the segmented arm. This is not full natural-language segmentation, symbol grounding, or a PrismionCell test.
