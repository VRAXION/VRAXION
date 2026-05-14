# ANCHOR-MINI-014 Result: Operation-Plan Parser Bisect

## Verdict

```text
ANCHOR_MINI_014_PARTIAL_BUDGET
```

The overnight CPU sweep did not finish enough jobs to settle the main
`BLOCK_ONLY_PLAN_FIRST` vs `QUERY_FULL_TEXT_PLAN_FIRST` question. It did,
however, produce a useful partial diagnostic.

## Run

Smoke:

```bash
python tools/anchorweave/run_anchor_mini014_operation_plan.py ^
  --out target/anchorweave/anchor_mini014_operation_plan/smoke_v2 ^
  --seeds 2026 ^
  --models CHAR_CNN ^
  --arms ANSWER_ONLY_DIRECT,BLOCK_ONLY_PLAN_FIRST,QUERY_FULL_TEXT_PLAN_FIRST,SHUFFLED_QUERY_PLAN_FIRST,ORACLE_PARSED_PLAN_VISIBLE ^
  --candidate-counts 4 ^
  --steps 1,2 ^
  --train-examples 512 ^
  --eval-examples 512 ^
  --epochs 10 ^
  --jobs 4
```

Overnight partial:

```bash
python tools/anchorweave/run_anchor_mini014_operation_plan.py ^
  --out target/anchorweave/anchor_mini014_operation_plan/night_2026_05_11 ^
  --seeds 2026-2035 ^
  --models CHAR_CNN ^
  --arms ANSWER_ONLY_DIRECT,GLOBAL_PLAN_FIRST,BLOCK_ONLY_PLAN_FIRST,QUERY_FULL_TEXT_PLAN_FIRST,SHUFFLED_QUERY_PLAN_FIRST,SHORTCUT_TEACHER,ORACLE_PARSED_PLAN_VISIBLE ^
  --candidate-counts 4,8 ^
  --steps 1,2,3 ^
  --train-examples 4096 ^
  --eval-examples 2048 ^
  --epochs 80 ^
  --jobs 20 ^
  --budget-hours 3.5
```

The run was stopped after it exceeded the intended overnight window without
new completed metrics for the expensive block/query jobs. A partial summary was
written from the completed rows.

## Key Numbers

Completed overnight rows:

```text
completed_jobs: 14 / 420
valid_jobs: 14
```

Grouped means:

| arm | candidates | steps | jobs | eval acc | trap rate | start | goal | ops | final | policy | plan exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ANSWER_ONLY_DIRECT` | 4 | 1 | 5 | 0.196 | 0.733 | 0.046 | 0.015 | 0.035 | 0.016 | 0.447 | 0.000 |
| `GLOBAL_PLAN_FIRST` | 4 | 1 | 5 | 0.167 | 0.772 | 1.000 | 0.998 | 0.742 | 0.412 | 0.593 | 0.001 |
| `ORACLE_PARSED_PLAN_VISIBLE` | 4 | 1 | 4 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation

The completed rows establish three bounded facts:

```text
1. The stress is valid: answer-only overfits the surface shortcut.
2. The task itself is valid: oracle parsed PLAN solves it perfectly.
3. Global PLAN-first text encoding is not enough: it learns some fields but
   still routes to the shortcut.
```

The important missing evidence is the block/query comparison:

```text
BLOCK_ONLY_PLAN_FIRST:
  not enough completed overnight data

QUERY_FULL_TEXT_PLAN_FIRST:
  not enough completed overnight data
```

Therefore this result must not be reported as a final positive or final
negative for operation-plan learning. It only narrows the problem.

## Claim Boundary

MINI-014 does not prove symbol grounding, natural-language AnchorCells, or a
successful text carrier. It confirms that the oracle PLAN route remains valid
on the operation task and that direct/global text routes are insufficient.

## Next Step

Do not scale to a 256-cell human AnchorCell dataset from this state.

Run a smaller `MINI-014B` that prioritizes only the expensive unresolved arms:

```text
BLOCK_ONLY_PLAN_FIRST
QUERY_FULL_TEXT_PLAN_FIRST
SHUFFLED_QUERY_PLAN_FIRST
ORACLE_PARSED_PLAN_VISIBLE
ANSWER_ONLY_DIRECT
```

Use fewer seeds, fewer epochs, and smaller examples first. The goal is to
settle whether block-local operation learning works at all before testing full
text candidate binding.
