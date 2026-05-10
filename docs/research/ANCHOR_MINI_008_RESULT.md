# ANCHOR-MINI-008 Result

## Verdict

```text
ANCHOR_MINI_008_PROCESS_FIRST_STRONG_POSITIVE
```

ANCHOR-MINI-008 tested process-first teaching in the sparse audit carrier.
The model had to learn a scored PLAN route before emitting the answer.

## Run

```bash
python S:/Git/VRAXION_anchorwiki/tools/anchorweave/run_anchor_mini008_process_teacher.py --out target/anchorweave/anchor_mini008_process_teacher/full_2026_05_10 --seeds 2026-2125 --jobs 16 --budget-hours 8 --skip-build
```

Runtime:

```text
407.88 seconds
```

## Summary

```text
completed_jobs: 700
valid_jobs: 700
valid_seed_count: 100
blocked_jobs: 0
```

| carrier | answer_eval | trap_rate | invalid_reject | policy_bit | plan_exact | consistency |
|---|---:|---:|---:|---:|---:|---:|
| `SPARSE_DIRECT_ANSWER` | 0.100 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `SPARSE_AUX_PLAN_DIRECT_ANSWER` | 0.100 | 1.000 | 0.998 | 0.985 | 0.941 | 0.109 |
| `SPARSE_PLAN_FIRST` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `SPARSE_PLAN_FIRST_HYBRID` | 1.000 | 0.000 | 1.000 | 0.999 | 0.997 | 1.000 |
| `SPARSE_SHUFFLED_TEACHER` | 0.000 | 0.333 | 0.667 | 0.500 | 0.000 | 1.000 |
| `SPARSE_SHORTCUT_TEACHER` | 0.100 | 1.000 | 0.000 | 0.253 | 0.001 | 1.000 |
| `SPARSE_ORACLE_PLAN_VISIBLE` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation

A positive result means that the sparse carrier learned to close the
surface shortcut through a predicted process route. Wrong/shuffled teacher
controls guard against the result being caused only by extra supervision or
routing structure.

## Claim Boundary

This is still a toy sparse-carrier result. It does not prove full
natural-language AnchorCells or symbol grounding at scale.
