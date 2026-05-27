# D38 Learned Conditioning Router Field Proof Result

Local recovery file for the D38 learned conditioning router field proof.

The runnable evidence is produced by:

```bash
python scripts/probes/run_d38_learned_conditioning_router_field_proof.py --out target/pilot_wave/d38_learned_conditioning_router_field_proof/smoke
python scripts/probes/run_d38_learned_conditioning_router_field_proof_check.py --check-only --out target/pilot_wave/d38_learned_conditioning_router_field_proof/smoke
```

## Result Status

Recovered locally from current `origin/main` on branch `codex/d39-learned-router-scale-confirm`.

- artifact path: `target/pilot_wave/d38_learned_conditioning_router_field_proof/smoke`
- decision: `learned_conditioning_router_field_confirmed`
- verdict: `D38_LEARNED_CONDITIONING_ROUTER_FIELD_CONFIRMED`
- next: `D39_ROUTER_LAYER_SCALE_CONFIRM`

## Dataset Invariants

- duplicate_target_pocket_rate = 0.0
- missing_target_pocket_rate = 0.0
- expected_selected_points_to_target_rate = 1.0

## OOD Rule Invariance

- known_rule_oracle_test_accuracy = 1.0
- known_rule_oracle_ood_accuracy = 1.0
- ood_label_rule_changed = false

## Metrics

| Arm | train | test | ood |
| --- | ---: | ---: | ---: |
| MONOLITHIC_FORMULA_BASELINE | 0.3520 | 0.3196 | 0.3216 |
| ORACLE_GATED_RULE_FORMULA_UPPER_BOUND | 1.0000 | 1.0000 | 1.0000 |
| MUTABLE_LEARNED_ROUTER_GATE | 1.0000 | 1.0000 | 1.0000 |
| SHUFFLED_GATE_CONTROL | 0.1192 | 0.1044 | 0.1088 |
| NO_FAMILY_INPUT_CONTROL | 0.3520 | 0.3196 | 0.3216 |
| EXPLICIT_TARGET_STATE_UPPER_BOUND | 1.0000 | 1.0000 | 1.0000 |

## Deltas

- monolithic_vs_learned_test_delta = 0.6804
- learned_vs_shuffled_test_delta = 0.8956
- learned_vs_no_family_test_delta = 0.6804

## Boundary

A positive D38 proves only that a small mutable learned router gate can learn formula binding on this controlled known-rule symbolic pocket task. It does not prove hidden-rule Raven reasoning, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.
