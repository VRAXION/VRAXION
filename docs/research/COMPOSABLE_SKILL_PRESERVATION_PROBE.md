# Composable Skill Preservation Probe

## Goal

Quickly test whether composition training in a shared end-to-end network overwrites primitive skills, while frozen primitive modules preserve skill identity under composition.

Tasks use arithmetic modulo 7:

- primitives: `add`, `mul`
- composites: `add_then_mul`, `mul_then_add`

## Results

| Arm | Primitive Before | Primitive After | Drift | Composition |
|---|---:|---:|---:|---:|
| `shared_no_replay` | `1.000000` | `0.316035` | `0.683965` | `0.846647` |
| `shared_with_replay` | `1.000000` | `1.000000` | `0.000000` | `0.784840` |
| `frozen_modules` | `1.000000` | `1.000000` | `0.000000` | `1.000000` |

## Verdict

```json
{
  "shared_no_replay_forgets_primitives": true,
  "replay_reduces_forgetting": true,
  "frozen_modules_preserve_primitives": true,
  "frozen_modules_compose_successfully": true,
  "logic_controller_hypothesis_supported": true
}
```

## Interpretation

This is a quick toy check, not a final architecture claim. The frozen-module arm is an existence/reference check: it uses the intended composition policy and does not prove that the controller can discover that policy. A positive pattern means the next serious probe should compare shared-loss training against a modular controller with frozen primitive regression tests and explicit route/edge survival rules.

Claim boundary: toy evidence only; no consciousness, biology, or production validation claim.
