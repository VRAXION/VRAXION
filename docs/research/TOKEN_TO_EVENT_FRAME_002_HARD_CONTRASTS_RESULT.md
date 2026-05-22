# TOKEN_TO_EVENT_FRAME_002_HARD_CONTRASTS Result

## Goal

Test whether raw event clauses can be converted into hard event frames under contrast cases that should break bag/static shortcuts.

## Run

```text
seeds=2026,2027
train_examples=1024
eval_examples=1024
epochs=30
jobs=20
completed_jobs=14
heartbeat_sec=15
```

## Arm Summary

| Arm | Frame Exact | Ledger | Role Swap | Affected | Noop/Neg | Mention | Ref |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BAG_OF_TOKENS_EVENT_FRAME` | `0.999` | `0.998` | `0.500` | `0.833` | `1.000` | `1.000` | `1.000` |
| `DIRECT_STORY_GRU_ANSWER` | `nan` | `0.688` | `0.500` | `0.750` | `1.000` | `1.000` | `nan` |
| `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` |
| `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.999` | `0.998` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` |
| `SHUFFLED_EVENT_FRAME_TEACHER` | `0.137` | `0.248` | `0.250` | `0.417` | `0.000` | `0.000` | `0.543` |
| `STATIC_POSITION_EVENT_FRAME` | `0.999` | `0.999` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` |
| `STORY_TO_EVENT_FRAMES_GRU` | `0.998` | `0.996` | `1.000` | `0.833` | `1.000` | `1.000` | `1.000` |

## Verdict

```json
[
  "EXPLICIT_EVENT_PARSER_REQUIRED_FOR_NOW",
  "EVENT_FRAME_LEXICAL_SHORTCUT"
]
```

## Interpretation

This is a useful negative smoke, not a hard-positive result.

What worked:

- `EVENT_FRAME_ORACLE` reached `1.000`, so the identity ledger and hard-frame answer route are valid.
- `SHUFFLED_EVENT_FRAME_TEACHER` collapsed to `0.248` ledger accuracy and `0.137` frame exact accuracy, so correct frame labels matter.
- `SEGMENTED_EVENT_FRAME_CLASSIFIER` and `STORY_TO_EVENT_FRAMES_GRU` still solve the task when hard frames are predicted correctly.

What failed:

- `STATIC_POSITION_EVENT_FRAME` matched or beat the segmented parser on the hard metrics (`0.999` ledger, `1.000` role swap, `1.000` affected entity).
- `BAG_OF_TOKENS_EVENT_FRAME` was weakened on same-token role swaps (`0.500`) but still had `0.998` ledger accuracy overall.

So the v2 track is still too template/position-solvable. This does not justify moving to PrismionCell yet. The next useful work is a stricter split where the hard contrast templates are not present in train, hard cases are not diluted by easy random cases, and static-position parsing cannot memorize fixed phrasing.

Run hygiene check:

- `metrics.jsonl`: 14 completed job records
- `progress.jsonl`: 37 progress/heartbeat records
- `job_progress/*.jsonl`: 14 per-job progress files

## Claim Boundary

V2 still assumes gold clause segmentation for the segmented arm. This is not full natural-language segmentation, symbol grounding, or a PrismionCell test.
