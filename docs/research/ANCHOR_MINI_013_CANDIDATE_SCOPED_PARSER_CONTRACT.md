# ANCHOR-MINI-013: Candidate-Scoped Parser A/B

## Claim

MINI-013 tests whether MINI-012 failed because of insufficient template scale or
because the parser did not bind candidate-local fields.

Narrow question:

```text
Can a query-scoped model learn:
candidate slot -> candidate-local effect -> candidate-local policy bit
under held-out serialization templates?
```

This remains a toy parser/binding test, not natural-language grounding.

## Task

Reuse the MINI-012 shortcut-flip setup:

```text
4 candidates
4 categories
train surface shortcut points to gold with p=0.90
eval surface shortcut points to wrong with p=0.90
held-out eval templates
```

Eval input is raw text only. No answer labels, match bits, policy bits, or PLAN
fields may appear in eval text.

## Arms

```text
ANSWER_ONLY_DIRECT
  Global raw text -> direct answer.

GLOBAL_PLAN_FIRST
  MINI-012-style global encoder -> goal/effect/policy heads.

SCALE_ONLY_GLOBAL
  Same global route at 64/128/256 templates.

QUERY_SCOPED_PLAN_FIRST
  Shared encoder over:
    QUERY=A + full task
    QUERY=B + full task
    QUERY=C + full task
    QUERY=D + full task
  Then candidate-local effect and policy predictions.

SHUFFLED_QUERY_SCOPED
  Query-scoped route with shifted/wrong PLAN labels.
```

`QUERY=A/B/C/D` is allowed because it scopes the candidate being inspected. It
does not reveal effect, match, policy, or answer.

## Primary Gates

`QUERY_SCOPED_PLAN_FIRST` is positive if:

```text
answer_eval_accuracy >= 0.80
shortcut_trap_rate <= 0.25
candidate_effect_accuracy >= 0.80
candidate_policy_accuracy >= 0.80
answer_from_policy_consistency >= 0.95
beats GLOBAL_PLAN_FIRST by >= 0.25 answer_eval_accuracy
beats SHUFFLED_QUERY_SCOPED by >= 0.25 answer_eval_accuracy
```

Diagnostic outcomes:

```text
CANDIDATE_SCOPED_FIXES_BINDING
SCALE_ONLY_FIXES_BINDING
BOTH_PASS
BOTH_FAIL
```

## Run

Smoke:

```bash
python tools/anchorweave/run_anchor_mini013_candidate_scoped_parser.py ^
  --out target/anchorweave/anchor_mini013_candidate_scoped_parser/smoke ^
  --seeds 2026 ^
  --models CHAR_CNN_QUERY_SCOPED ^
  --arms GLOBAL_PLAN_FIRST,QUERY_SCOPED_PLAN_FIRST,SHUFFLED_QUERY_SCOPED ^
  --template-counts 64 ^
  --train-examples 512 ^
  --eval-examples 512 ^
  --epochs 10 ^
  --jobs 2
```

Fast decision:

```bash
python tools/anchorweave/run_anchor_mini013_candidate_scoped_parser.py ^
  --out target/anchorweave/anchor_mini013_candidate_scoped_parser/fast_2026_05_10 ^
  --seeds 2026-2030 ^
  --models CHAR_CNN_QUERY_SCOPED ^
  --arms ANSWER_ONLY_DIRECT,GLOBAL_PLAN_FIRST,SCALE_ONLY_GLOBAL,QUERY_SCOPED_PLAN_FIRST,SHUFFLED_QUERY_SCOPED ^
  --template-counts 64,128,256 ^
  --train-examples 4096 ^
  --eval-examples 2048 ^
  --epochs 60 ^
  --jobs 8 ^
  --budget-minutes 45
```

Generated outputs stay under `target/`. Do not write under
`data/anchorweave/cells/`.
