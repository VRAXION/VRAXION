# Modular Skill Controller Probe

## Goal

Test whether frozen primitive modules can be composed by a learned controller without primitive skill drift.

## Results

| Arm | Primitive Before | Primitive After | Drift | Composition | Program Acc |
|---|---:|---:|---:|---:|---:|
| `shared_no_replay` | `1.000000` | `0.316035` | `0.683965` | `0.846647` | `null` |
| `shared_with_replay` | `1.000000` | `1.000000` | `0.000000` | `0.784840` | `null` |
| `frozen_hand_composition` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `null` |
| `frozen_learned_controller` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `1.000000` |

## Verdict

```json
{
  "learned_controller_preserves_primitives": true,
  "learned_controller_composes_successfully": true,
  "learned_controller_program_selection_successful": true,
  "shared_no_replay_still_forgets": true,
  "modular_controller_hypothesis_supported": true
}
```

## Interpretation

This is a controller sanity check. It uses a small fixed candidate-program set, so it does not prove open-ended program discovery. It does test whether composition can be learned above frozen primitives without overwriting them.

Claim boundary: toy evidence only; no consciousness, biology, or production claim.
