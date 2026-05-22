# ANCHOR-MINI-010 Result

## Verdict

```text
ANCHOR_MINI_010_SERIALIZATION_STRONG_POSITIVE
```

ANCHOR-MINI-010 tested whether process-first sparse routing survives
held-out byte serialization variants under a schema-aware decoder.

## Run

```bash
python S:/Git/VRAXION_anchorwiki/tools/anchorweave/run_anchor_mini010_serialization_robustness.py --out target/anchorweave/anchor_mini010_serialization_robustness/full_2026_05_10 --seeds 2026-2125 --jobs 24 --budget-hours 8 --skip-build
```

Runtime:

```text
3293.13 seconds
```

## Summary

```text
completed_jobs: 7000
valid_jobs: 4200
valid_seed_count: 100
blocked_jobs: 0
```

| carrier | answer_eval | trap_rate | invalid_reject | policy_bit | plan_exact | consistency | byte_ok | decode_ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `BYTE_DIRECT_ANSWER` | 0.100 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| `BYTE_AUX_PLAN_DIRECT_ANSWER` | 0.100 | 1.000 | 1.000 | 0.984 | 0.937 | 0.112 | 1.000 | 1.000 |
| `BYTE_PLAN_FIRST` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `BYTE_PLAN_FIRST_HYBRID` | 1.000 | 0.000 | 1.000 | 0.999 | 0.997 | 1.000 | 1.000 | 1.000 |
| `BYTE_SHUFFLED_TEACHER` | 0.000 | 0.333 | 0.667 | 0.500 | 0.000 | 1.000 | 1.000 | 1.000 |
| `BYTE_SHORTCUT_TEACHER` | 0.100 | 1.000 | 0.000 | 0.349 | 0.008 | 1.000 | 1.000 | 1.000 |
| `BYTE_ORACLE_PLAN_VISIBLE` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Format Curve

| eval_format | BYTE_PLAN_FIRST acc | trap | policy_bit | plan_exact | decode_ok | pass |
|---|---:|---:|---:|---:|---:|---|
| `canonical_fixed` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | True |
| `field_order_swap` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | True |
| `slot_order_perm` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | True |
| `alias_long` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | True |
| `noise_fields` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | True |

## Interpretation

A positive result means the schema-aware byte carrier closes the surface
shortcut through the PLAN route across held-out serialization variants.
The fixed-template control guards against claiming that MINI-009 only
worked because one byte layout was hardcoded.

## Claim Boundary

This is still a toy schema-aware byte-carrier result. It does not prove
learned parsing, natural-language AnchorCells, or symbol grounding at scale.
