# PILOT_SENSOR_GUARD_COMPENSATION_001 Result

## Goal

Test whether stricter strength/margin guard thresholds can compensate for factor-heldout learned sensor errors.

## Top Threshold Settings

| Setting | Score | Action | False Commit | Missed Execute | Known | Weak/Amb | Neg | Corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scope_stack_char_ngram_mlp_factor|s=0.75|m=0.30` | `0.404` | `0.745` | `0.170` | `0.000` | `1.000` | `0.750` | `0.500` | `0.500` |
| `scope_stack_char_ngram_mlp_factor|s=0.75|m=0.45` | `0.404` | `0.745` | `0.170` | `0.000` | `1.000` | `0.750` | `0.500` | `0.500` |
| `scope_stack_char_ngram_mlp_factor|s=0.75|m=0.60` | `0.404` | `0.745` | `0.170` | `0.000` | `1.000` | `0.750` | `0.500` | `0.500` |
| `scope_stack_char_ngram_mlp_factor|s=0.85|m=0.30` | `0.404` | `0.745` | `0.170` | `0.000` | `1.000` | `0.750` | `0.500` | `0.500` |
| `scope_stack_char_ngram_mlp_factor|s=0.85|m=0.45` | `0.404` | `0.745` | `0.170` | `0.000` | `1.000` | `0.750` | `0.500` | `0.500` |
| `scope_stack_char_ngram_mlp_factor|s=0.85|m=0.60` | `0.404` | `0.745` | `0.170` | `0.000` | `1.000` | `0.750` | `0.500` | `0.500` |
| `direct_evidence_char_ngram_mlp_factor|s=0.95|m=0.30` | `0.149` | `0.574` | `0.000` | `0.426` | `0.000` | `1.000` | `0.500` | `0.000` |
| `direct_evidence_char_ngram_mlp_factor|s=0.95|m=0.45` | `0.149` | `0.574` | `0.000` | `0.426` | `0.000` | `1.000` | `0.500` | `0.000` |
| `direct_evidence_char_ngram_mlp_factor|s=0.95|m=0.60` | `0.149` | `0.574` | `0.000` | `0.426` | `0.000` | `1.000` | `0.500` | `0.000` |
| `scope_stack_char_ngram_mlp_factor|s=0.95|m=0.30` | `0.149` | `0.574` | `0.000` | `0.426` | `0.000` | `1.000` | `0.500` | `0.000` |
| `scope_stack_char_ngram_mlp_factor|s=0.95|m=0.45` | `0.149` | `0.574` | `0.000` | `0.426` | `0.000` | `1.000` | `0.500` | `0.000` |
| `scope_stack_char_ngram_mlp_factor|s=0.95|m=0.60` | `0.149` | `0.574` | `0.000` | `0.426` | `0.000` | `1.000` | `0.500` | `0.000` |

## Verdict

```json
{
  "best_by_score": "scope_stack_char_ngram_mlp_factor|s=0.75|m=0.30",
  "best_metrics": {
    "action_accuracy": 0.7446808510638298,
    "correction_accuracy": 0.5,
    "false_commit_rate": 0.1702127659574468,
    "known_accuracy": 1.0,
    "margin_threshold": 0.3,
    "missed_execute_rate": 0.0,
    "negation_accuracy": 0.5,
    "score": 0.40425531914893614,
    "seed_count": 10.0,
    "strength_threshold": 0.75,
    "weak_ambiguous_accuracy": 0.75
  },
  "global": [
    "GUARD_CALIBRATION_CANNOT_FIX_SENSOR_SCOPE_ERRORS"
  ],
  "viable_count": 0
}
```

## Interpretation

If no threshold setting reaches high action accuracy with low false commits, the blocker is sensor evidence quality rather than guard calibration.
False commits from strong but wrong evidence, such as negation/correction scope errors, cannot be reliably fixed downstream by thresholds.

## Claim Boundary

No general NLU, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.
