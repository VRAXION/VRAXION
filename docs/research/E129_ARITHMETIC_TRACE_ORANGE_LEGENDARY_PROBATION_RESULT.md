# E129 Arithmetic Trace Orange/Legendary Probation Result

```text
decision = e129_arithmetic_trace_orange_legendary_probation_confirmed
next = E130_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET
boundary = exact arithmetic trace/operator behavior only; not word-problem solving or neural LLM training

operator_count = 9
orange_legendary_candidate_count = 9
qualified_activation_min = 300000
qualified_activation_total = 2700000
case_count_total = 2700000
negative_scope_case_count_total = 9000

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
min_in_scope_accuracy_min = 1.000
negative_scope_pass_rate_min = 1.000
```

## Summary

E129 tests whether direct arithmetic training data can become scoped
Operator-level knowledge under Orange/Legendary stress and prune pressure.

The confirmed operators are:

```text
e129_add_sub_trace_operator
e129_multiplication_trace_operator
e129_exact_division_trace_operator
e129_floor_division_trace_operator
e129_signed_integer_trace_operator
e129_decimal_fraction_trace_operator
e129_parenthesized_mixed_precedence_operator
e129_invalid_trace_rejection_guard
e129_division_by_zero_guard
```

Each operator reached:

```text
rank_after = OrangeLegendaryCandidate
qualified_activation = 300000
campaign_count = 15
negative_scope_case_count = 1000
hard_negative = 0
false_commit = 0
wrong_scope_call = 0
unsupported_answer = 0
```

## What Was Learned

The run confirms scoped exact arithmetic behavior over:

```text
addition / subtraction
multiplication
exact division
floor division
signed integer arithmetic
decimal and fraction rendering
parenthesized mixed precedence
invalid trace rejection
division-by-zero rejection
```

The strongest interpretation is:

```text
Direct arithmetic training data can promote into stable scoped arithmetic
Operators when the task is exact compute / trace validation with explicit
scope guards and negative-scope no-call checks.
```

## What Is Not Claimed

E129 does not claim:

```text
natural-language word-problem solving
GSM8K solving
open-domain math reasoning
neural LLM training
production assistant behavior
PermaCore / TrueGolden
```

The negative-scope checks intentionally keep natural-language word problems
without a visible arithmetic expression or trace as no-call cases. The next
needed test is E130 transfer: more surface formats, longer text around the
expression, and stronger word-problem no-call routing.

## Artifacts

```text
docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation/report.md
docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation/summary.json
docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation/decision.json
docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation/operator_orange_results.json
docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation/variant_report.json
docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation/stress_report.json
docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation/negative_scope_report.json
docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation/deterministic_replay.json
docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation/row_level_samples.jsonl
```

## Reproduce

```powershell
python scripts/probes/run_e129_arithmetic_trace_orange_legendary_probation.py --out target/pilot_wave/e129_arithmetic_trace_orange_legendary_probation --sample-out docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation
```
