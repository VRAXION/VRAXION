# TOKEN_TO_EVENT_FRAME_004_CURRICULUM_FACTORS Result

## Run

```text
seeds=2026,2027
suite=all
train_examples=1024
eval_examples=1024
epochs=30
jobs=20
completed_jobs=160
first_collapse_round=0
```

## Verdict

```json
[
  "NEGATION_MODAL_BOTTLENECK",
  "PARSER_WEAK_UNDER_CURRICULUM",
  "ROLE_BINDING_BOTTLENECK",
  "STATIC_SHORTCUT_RECOVERED",
  "TEMPLATE_COLLAPSE_AT_ROUND_0"
]
```

## Interpretation

This is a useful isolation result, not a positive pass.

The deterministic ledger is still valid: `EVENT_FRAME_ORACLE` is `1.000` ledger accuracy across the smoke. The blocker remains before the ledger, inside `raw clause -> event frame`.

Main signals:

- `role_binding`: `SEGMENTED` reaches `0.905` frame / `0.908` ledger / `0.856` hard, but pair accuracy is only `0.767` and `STATIC` remains close (`0.813` hard). This is not a clean parser win.
- `negation_modal`: `SEGMENTED` falls to `0.809` frame / `0.632` ledger / `0.501` hard. Plain `not stolen` is solved (`1.000`), but heldout near-miss/modal/mention forms fail (`almost`: `0.080`, modal: `0.250`, mention: `0.114`).
- `template_curriculum`: first collapse is round `0` by heldout transfer, because round-0 `SEGMENTED` has high hard score (`0.907`) but heldout accuracy only `0.684`. Some later rounds recover in-distribution, but heldout transfer is unstable.
- `combined_recheck`: `SEGMENTED` is `0.864` frame / `0.793` ledger / `0.597` hard, confirming that skill interaction is worse than isolated role binding.

Next implication:

```text
Do not build PrismionCell yet.
Next isolate no-op/inhibition parsing:
  EVENT_FRAME_NEGATION_MODAL_CURRICULUM_001
```

## Suite Summary

| Suite | Round | Arm | Frame | Ledger | Hard | Heldout |
|---|---:|---|---:|---:|---:|---:|
| `combined_recheck` | `0` | `BAG_OF_TOKENS_EVENT_FRAME` | `0.763` | `0.588` | `0.470` | `0.542` |
| `combined_recheck` | `0` | `DIRECT_STORY_GRU_ANSWER` | `nan` | `0.907` | `0.727` | `0.814` |
| `combined_recheck` | `0` | `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` |
| `combined_recheck` | `0` | `POSITION_ONLY_EVENT_FRAME` | `0.164` | `0.284` | `0.421` | `0.313` |
| `combined_recheck` | `0` | `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.864` | `0.793` | `0.597` | `0.586` |
| `combined_recheck` | `0` | `SHUFFLED_EVENT_FRAME_TEACHER` | `0.000` | `0.597` | `0.349` | `0.554` |
| `combined_recheck` | `0` | `STATIC_POSITION_EVENT_FRAME` | `0.891` | `0.775` | `0.731` | `0.551` |
| `combined_recheck` | `0` | `STORY_TO_EVENT_FRAMES_GRU` | `0.929` | `0.782` | `0.643` | `0.563` |
| `negation_modal` | `0` | `BAG_OF_TOKENS_EVENT_FRAME` | `0.790` | `0.597` | `0.454` | `0.194` |
| `negation_modal` | `0` | `DIRECT_STORY_GRU_ANSWER` | `nan` | `0.875` | `0.722` | `0.749` |
| `negation_modal` | `0` | `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` |
| `negation_modal` | `0` | `POSITION_ONLY_EVENT_FRAME` | `0.163` | `0.501` | `0.463` | `0.500` |
| `negation_modal` | `0` | `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.809` | `0.632` | `0.501` | `0.265` |
| `negation_modal` | `0` | `SHUFFLED_EVENT_FRAME_TEACHER` | `0.076` | `0.262` | `0.265` | `0.262` |
| `negation_modal` | `0` | `STATIC_POSITION_EVENT_FRAME` | `0.917` | `0.941` | `0.903` | `0.882` |
| `negation_modal` | `0` | `STORY_TO_EVENT_FRAMES_GRU` | `0.914` | `0.788` | `0.684` | `0.576` |
| `role_binding` | `0` | `BAG_OF_TOKENS_EVENT_FRAME` | `0.750` | `0.708` | `0.678` | `0.917` |
| `role_binding` | `0` | `DIRECT_STORY_GRU_ANSWER` | `nan` | `1.000` | `1.000` | `1.000` |
| `role_binding` | `0` | `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` |
| `role_binding` | `0` | `POSITION_ONLY_EVENT_FRAME` | `0.179` | `0.500` | `0.586` | `1.000` |
| `role_binding` | `0` | `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.905` | `0.908` | `0.856` | `0.815` |
| `role_binding` | `0` | `SHUFFLED_EVENT_FRAME_TEACHER` | `0.000` | `0.810` | `0.805` | `0.803` |
| `role_binding` | `0` | `STATIC_POSITION_EVENT_FRAME` | `0.974` | `0.897` | `0.813` | `0.794` |
| `role_binding` | `0` | `STORY_TO_EVENT_FRAMES_GRU` | `0.891` | `0.818` | `0.721` | `0.637` |
| `template_curriculum` | `0` | `BAG_OF_TOKENS_EVENT_FRAME` | `0.758` | `0.736` | `0.802` | `0.850` |
| `template_curriculum` | `0` | `DIRECT_STORY_GRU_ANSWER` | `nan` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `0` | `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `0` | `POSITION_ONLY_EVENT_FRAME` | `0.167` | `0.434` | `0.396` | `0.500` |
| `template_curriculum` | `0` | `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.800` | `0.842` | `0.907` | `0.684` |
| `template_curriculum` | `0` | `SHUFFLED_EVENT_FRAME_TEACHER` | `0.003` | `0.249` | `0.282` | `0.200` |
| `template_curriculum` | `0` | `STATIC_POSITION_EVENT_FRAME` | `0.860` | `0.533` | `0.727` | `0.066` |
| `template_curriculum` | `0` | `STORY_TO_EVENT_FRAMES_GRU` | `0.869` | `0.700` | `0.825` | `0.400` |
| `template_curriculum` | `1` | `BAG_OF_TOKENS_EVENT_FRAME` | `0.767` | `0.768` | `0.821` | `0.934` |
| `template_curriculum` | `1` | `DIRECT_STORY_GRU_ANSWER` | `nan` | `0.933` | `0.956` | `0.866` |
| `template_curriculum` | `1` | `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `1` | `POSITION_ONLY_EVENT_FRAME` | `0.166` | `0.102` | `0.291` | `0.000` |
| `template_curriculum` | `1` | `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.872` | `0.916` | `0.952` | `0.832` |
| `template_curriculum` | `1` | `SHUFFLED_EVENT_FRAME_TEACHER` | `0.001` | `0.401` | `0.318` | `0.383` |
| `template_curriculum` | `1` | `STATIC_POSITION_EVENT_FRAME` | `0.989` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `1` | `STORY_TO_EVENT_FRAMES_GRU` | `0.966` | `0.983` | `0.991` | `0.967` |
| `template_curriculum` | `2` | `BAG_OF_TOKENS_EVENT_FRAME` | `0.777` | `0.750` | `0.813` | `0.916` |
| `template_curriculum` | `2` | `DIRECT_STORY_GRU_ANSWER` | `nan` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `2` | `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `2` | `POSITION_ONLY_EVENT_FRAME` | `0.166` | `0.461` | `0.470` | `0.500` |
| `template_curriculum` | `2` | `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.892` | `0.925` | `0.958` | `0.851` |
| `template_curriculum` | `2` | `SHUFFLED_EVENT_FRAME_TEACHER` | `0.011` | `0.570` | `0.455` | `0.649` |
| `template_curriculum` | `2` | `STATIC_POSITION_EVENT_FRAME` | `0.874` | `0.508` | `0.725` | `0.017` |
| `template_curriculum` | `2` | `STORY_TO_EVENT_FRAMES_GRU` | `0.872` | `0.873` | `0.914` | `0.749` |
| `template_curriculum` | `3` | `BAG_OF_TOKENS_EVENT_FRAME` | `0.766` | `0.663` | `0.765` | `0.751` |
| `template_curriculum` | `3` | `DIRECT_STORY_GRU_ANSWER` | `nan` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `3` | `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `3` | `POSITION_ONLY_EVENT_FRAME` | `0.157` | `0.574` | `0.555` | `0.500` |
| `template_curriculum` | `3` | `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.947` | `0.966` | `0.981` | `0.933` |
| `template_curriculum` | `3` | `SHUFFLED_EVENT_FRAME_TEACHER` | `0.000` | `0.230` | `0.234` | `0.249` |
| `template_curriculum` | `3` | `STATIC_POSITION_EVENT_FRAME` | `0.841` | `0.542` | `0.748` | `0.083` |
| `template_curriculum` | `3` | `STORY_TO_EVENT_FRAMES_GRU` | `0.960` | `0.975` | `0.986` | `0.950` |
| `template_curriculum` | `4` | `BAG_OF_TOKENS_EVENT_FRAME` | `0.792` | `0.725` | `0.833` | `0.887` |
| `template_curriculum` | `4` | `DIRECT_STORY_GRU_ANSWER` | `nan` | `0.832` | `0.888` | `0.664` |
| `template_curriculum` | `4` | `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `4` | `POSITION_ONLY_EVENT_FRAME` | `0.166` | `0.062` | `0.282` | `0.000` |
| `template_curriculum` | `4` | `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.902` | `0.878` | `0.922` | `0.757` |
| `template_curriculum` | `4` | `SHUFFLED_EVENT_FRAME_TEACHER` | `0.000` | `0.737` | `0.498` | `0.792` |
| `template_curriculum` | `4` | `STATIC_POSITION_EVENT_FRAME` | `0.915` | `0.667` | `0.734` | `0.333` |
| `template_curriculum` | `4` | `STORY_TO_EVENT_FRAMES_GRU` | `0.828` | `0.902` | `0.902` | `0.809` |
| `template_curriculum` | `5` | `BAG_OF_TOKENS_EVENT_FRAME` | `0.755` | `0.557` | `0.473` | `0.568` |
| `template_curriculum` | `5` | `DIRECT_STORY_GRU_ANSWER` | `nan` | `0.858` | `0.619` | `0.716` |
| `template_curriculum` | `5` | `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `5` | `POSITION_ONLY_EVENT_FRAME` | `0.164` | `0.491` | `0.496` | `0.480` |
| `template_curriculum` | `5` | `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.860` | `0.797` | `0.565` | `0.595` |
| `template_curriculum` | `5` | `SHUFFLED_EVENT_FRAME_TEACHER` | `0.040` | `0.488` | `0.359` | `0.538` |
| `template_curriculum` | `5` | `STATIC_POSITION_EVENT_FRAME` | `0.912` | `0.734` | `0.800` | `0.643` |
| `template_curriculum` | `5` | `STORY_TO_EVENT_FRAMES_GRU` | `0.830` | `0.776` | `0.450` | `0.572` |
| `template_curriculum` | `6` | `BAG_OF_TOKENS_EVENT_FRAME` | `0.790` | `0.645` | `0.648` | `0.684` |
| `template_curriculum` | `6` | `DIRECT_STORY_GRU_ANSWER` | `nan` | `0.987` | `0.974` | `0.975` |
| `template_curriculum` | `6` | `EVENT_FRAME_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` |
| `template_curriculum` | `6` | `POSITION_ONLY_EVENT_FRAME` | `0.167` | `0.687` | `0.371` | `0.566` |
| `template_curriculum` | `6` | `SEGMENTED_EVENT_FRAME_CLASSIFIER` | `0.878` | `0.828` | `0.733` | `0.655` |
| `template_curriculum` | `6` | `SHUFFLED_EVENT_FRAME_TEACHER` | `0.000` | `0.528` | `0.368` | `0.492` |
| `template_curriculum` | `6` | `STATIC_POSITION_EVENT_FRAME` | `0.936` | `0.762` | `0.865` | `0.703` |
| `template_curriculum` | `6` | `STORY_TO_EVENT_FRAMES_GRU` | `0.864` | `0.799` | `0.664` | `0.612` |

## Claim Boundary

This is a controlled factor-isolation toy parser probe. It is not full natural-language segmentation, symbol grounding, or a PrismionCell test.
