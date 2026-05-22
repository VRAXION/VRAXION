# ANCHOR-MINI-009 Result

## Verdict

```text
ANCHOR_MINI_009_BYTE_PLAN_STRONG_POSITIVE
```

ANCHOR-MINI-009 tested whether process-first sparse routing survives
when the task situation is supplied as a fixed ASCII byte record.

## Run

```bash
python S:/Git/VRAXION_anchorwiki/tools/anchorweave/run_anchor_mini009_byte_plan_bridge.py --out target/anchorweave/anchor_mini009_byte_plan_bridge/full_2026_05_10 --seeds 2026-2125 --jobs 16 --budget-hours 8 --skip-build
```

Runtime:

```text
1879.91 seconds
```

## Summary

```text
completed_jobs: 700
valid_jobs: 700
valid_seed_count: 100
blocked_jobs: 0
```

| carrier | answer_eval | trap_rate | invalid_reject | policy_bit | plan_exact | consistency | byte_ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BYTE_DIRECT_ANSWER` | 0.100 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `BYTE_AUX_PLAN_DIRECT_ANSWER` | 0.100 | 1.000 | 1.000 | 0.987 | 0.950 | 0.108 | 1.000 |
| `BYTE_PLAN_FIRST` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `BYTE_PLAN_FIRST_HYBRID` | 1.000 | 0.000 | 1.000 | 0.999 | 0.997 | 1.000 | 1.000 |
| `BYTE_SHUFFLED_TEACHER` | 0.000 | 0.333 | 0.667 | 0.500 | 0.000 | 1.000 | 1.000 |
| `BYTE_SHORTCUT_TEACHER` | 0.100 | 1.000 | 0.000 | 0.351 | 0.009 | 1.000 | 1.000 |
| `BYTE_ORACLE_PLAN_VISIBLE` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation

A positive result means the sparse carrier learned to close the surface
shortcut through a PLAN route inferred from serialized byte input. The
direct and aux-direct controls show that merely seeing the task or learning
auxiliary PLAN outputs is not enough unless the final answer routes through
the predicted PLAN/policy state.

## Claim Boundary

This is still a toy byte-carrier result. It does not prove full
natural-language AnchorCells or symbol grounding at scale.
