# TOKEN_TO_EVENT_FRAME_003_ANTI_STATIC Result

## Run

```text
seeds=2026,2027
train_examples=1024
eval_examples=1024
epochs=30
jobs=20
completed_jobs=16
final_round=4
adaptive_hardening=True
```

## Arm Summary

| Arm | Frame | Ledger | Hard | In-Dist | Heldout | Pair | Ref |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BAG_OF_TOKENS_EVENT_FRAME` | `0.770` | `0.671` | `0.599` | `0.684` | `0.659` | `0.661` | `1.000` |
| `DIRECT_STORY_GRU_ANSWER` | `nan` | `0.851` | `0.800` | `0.996` | `0.706` | `0.794` | `nan` |
| `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` |
| `POSITION_ONLY_EVENT_FRAME` | `0.190` | `0.520` | `0.277` | `0.581` | `0.459` | `0.485` | `0.996` |
| `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.865` | `0.821` | `0.695` | `1.000` | `0.643` | `0.754` | `1.000` |
| `SHUFFLED_EVENT_FRAME_TEACHER` | `0.001` | `0.484` | `0.356` | `0.433` | `0.535` | `0.493` | `0.448` |
| `STATIC_POSITION_EVENT_FRAME` | `0.879` | `0.754` | `0.806` | `1.000` | `0.508` | `0.651` | `0.989` |
| `STORY_TO_EVENT_FRAMES_GRU` | `0.903` | `0.766` | `0.695` | `1.000` | `0.531` | `0.680` | `1.000` |

## Verdict

```json
[
  "TEMPLATE_GENERALIZATION_WEAK",
  "PARSER_WEAK_UNDER_SHUFFLE",
  "NEGATION_MODAL_BOTTLENECK"
]
```

## Adaptive Rounds

| Round | SEGMENTED hard | STATIC hard | BAG hard | POSITION_ONLY hard | Verdict |
|---:|---:|---:|---:|---:|---|
| 0 | 0.830 | 0.858 | 0.576 | 0.571 | `POSITION_LEAK_WARNING`, `TEMPLATE_GENERALIZATION_WEAK` |
| 1 | 0.864 | 0.887 | 0.673 | 0.407 | `POSITION_LEAK_WARNING`, `TEMPLATE_GENERALIZATION_WEAK`, `ROLE_BINDING_BOTTLENECK` |
| 2 | 0.840 | 0.829 | 0.728 | 0.521 | `TEMPLATE_GENERALIZATION_WEAK`, `ROLE_BINDING_BOTTLENECK` |
| 3 | 0.799 | 0.868 | 0.655 | 0.445 | `POSITION_LEAK_WARNING`, `TEMPLATE_GENERALIZATION_WEAK`, `ROLE_BINDING_BOTTLENECK`, `NEGATION_MODAL_BOTTLENECK` |
| 4 | 0.695 | 0.806 | 0.599 | 0.277 | `TEMPLATE_GENERALIZATION_WEAK`, `PARSER_WEAK_UNDER_SHUFFLE`, `NEGATION_MODAL_BOTTLENECK` |

## Interpretation

This is a useful negative. The anti-static pressure did reduce the shortcut baselines, but it also pushed the learned segmented parser below the required hard gate.

What worked:

- `EVENT_FRAME_ORACLE` stayed at `1.000`, so the deterministic ledger and frame schema are valid.
- `POSITION_ONLY_EVENT_FRAME` fell to `0.277` hard score by round 4, so pure position metadata is not enough.
- `BAG_OF_TOKENS_EVENT_FRAME` fell to `0.599` hard score by round 4, so the syntactic shuffle does weaken bag shortcuts.
- `SHUFFLED_EVENT_FRAME_TEACHER` collapsed to `0.001` frame exact / `0.484` ledger, so correct frame supervision still matters.

What failed:

- `SEGMENTED_EVENT_FRAME_CLASSIFIER` ended at `0.695` hard score, below the `0.85` gate.
- `heldout_template_accuracy` for the segmented parser was only `0.643`.
- `STATIC_POSITION_EVENT_FRAME` remained too high at `0.806` hard score, even after hardening.
- No-op/near-miss/modal/mention handling is the current stress point (`NEGATION_MODAL_BOTTLENECK`).

Full run was not launched because the smoke failed the required gate:

```text
STATIC hard_contrast_score <= 0.70
SEGMENTED hard_contrast_score >= 0.85
```

Run hygiene check:

- `metrics.jsonl`: 80 completed job records
- `progress.jsonl`: 187 progress/heartbeat records
- `hardening_rounds.jsonl`: 5 round records
- `job_progress/*.jsonl`: 80 per-job progress files
- raw `target/` outputs are not tracked by git

## Claim Boundary

This is a controlled anti-static toy parser probe. It is not full natural-language segmentation, symbol grounding, or a PrismionCell test.
