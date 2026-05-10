# ANCHOR-MINI-014: Operation-Plan Parser Bisect

## Claim

MINI-014 tests whether PLAN-first supervision can learn a small operation
policy from readable structured text:

```text
START + GOAL + STEPS + candidate operation sequence
-> execute candidate-local operations
-> compare final value to GOAL
-> reject surface shortcut
-> choose answer
```

This is a toy parser/binding test, not symbol grounding at scale.

## Task

Each example contains one header and four or eight candidate operation blocks.
Exactly one candidate reaches the goal after the stated number of steps.

```text
START: 4
GOAL: 19
STEPS: 3

A:
  OP1: ADD 5
  OP2: MUL 2
  OP3: ADD 1
  SURFACE: 2
```

Generation rules:

```text
operations: ADD, SUB, MUL
steps: 1, 2, 3
candidate_count: 4 primary, 8 stress
exactly one gold candidate
surface shortcut points to gold in train with p=0.90
surface shortcut points to wrong candidate in eval with p=0.90
gold slot balanced across candidates
no answer/final-result/policy leak in eval input
```

## Arms

```text
ANSWER_ONLY_DIRECT
  Full raw text -> direct answer.

GLOBAL_PLAN_FIRST
  Full raw text -> PLAN heads -> answer from policy.

BLOCK_ONLY_PLAN_FIRST
  Header + one candidate block per candidate -> local PLAN -> answer from policy.

QUERY_FULL_TEXT_PLAN_FIRST
  QUERY=A/B/... + full task text -> local PLAN -> answer from policy.

SHUFFLED_QUERY_PLAN_FIRST
  Query-full route with wrong shifted PLAN labels.

SHORTCUT_TEACHER
  Teacher praises highest-surface candidate.

ORACLE_PARSED_PLAN_VISIBLE
  Parsed PLAN visible, upper-bound only.
```

## Metrics

Report per arm/model/candidate_count/steps/seed:

```text
answer_eval_accuracy
shortcut_trap_rate
start_accuracy
goal_accuracy
steps_accuracy
candidate_ops_accuracy
candidate_final_value_accuracy
candidate_policy_accuracy
plan_exact_row_accuracy
answer_from_policy_consistency
train_surface_alignment
eval_surface_flip_rate
feature_leak_audit
```

## Gates

Valid stress requires:

```text
train_surface_alignment >= 0.85
eval_surface_flip_rate >= 0.85
feature_leak_audit == pass
ANSWER_ONLY_DIRECT shortcut_trap_rate >= 0.45 OR answer_eval_accuracy <= 0.45
ORACLE_PARSED_PLAN_VISIBLE answer_eval_accuracy >= 0.95
```

Primary positive for `QUERY_FULL_TEXT_PLAN_FIRST`:

```text
answer_eval_accuracy >= 0.80
shortcut_trap_rate <= 0.25
candidate_final_value_accuracy >= 0.80
candidate_policy_accuracy >= 0.80
answer_from_policy_consistency >= 0.95
beats ANSWER_ONLY_DIRECT by >= 0.25
beats SHUFFLED_QUERY_PLAN_FIRST by >= 0.25
beats SHORTCUT_TEACHER by >= 0.25
```

`BLOCK_ONLY_PLAN_FIRST` uses the same gates, but is interpreted only as
operation-logic success, not full parser success.

## Statuses

```text
ANCHOR_MINI_014_QUERY_FULL_STRONG_POSITIVE
ANCHOR_MINI_014_BLOCK_ONLY_POSITIVE
ANCHOR_MINI_014_DEPTH_LIMIT_FOUND
ANCHOR_MINI_014_CANDIDATE_SCALE_LIMIT_FOUND
ANCHOR_MINI_014_NEGATIVE
ANCHOR_MINI_014_INVALID_STRESS
ANCHOR_MINI_014_PARTIAL_BUDGET
ANCHOR_MINI_014_RESOURCE_BLOCKED
```

## Interpretation

```text
BLOCK_ONLY passes, QUERY_FULL fails:
  operation logic works; full-text candidate localization/binding is the blocker.

BLOCK_ONLY fails, ORACLE passes:
  text carrier cannot learn the operation PLAN yet.

QUERY_FULL passes:
  candidate-local binding is fixed by the operation-plan setup.

STEPS=1 passes but STEPS=2/3 fails:
  sequential operation depth is the blocker.

4 candidates pass but 8 candidates fail:
  candidate-set scaling is the blocker.
```
