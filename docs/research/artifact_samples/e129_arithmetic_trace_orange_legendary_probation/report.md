# E129 Arithmetic Trace Orange/Legendary Probation

boundary = exact arithmetic trace/operator behavior only; not word-problem solving or neural LLM training
decision = e129_arithmetic_trace_orange_legendary_probation_confirmed
next = E130_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET

## Metrics

```text
operator_count = 9
orange_legendary_candidate_count = 9
qualified_activation_min = 300000
case_count_total = 2700000
hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
min_in_scope_accuracy_min = 1.000
negative_scope_pass_rate_min = 1.000
```

## Confirmed Operators

```text
e129_add_sub_trace_operator -> OrangeLegendaryCandidate (300000 activations)
e129_multiplication_trace_operator -> OrangeLegendaryCandidate (300000 activations)
e129_exact_division_trace_operator -> OrangeLegendaryCandidate (300000 activations)
e129_floor_division_trace_operator -> OrangeLegendaryCandidate (300000 activations)
e129_signed_integer_trace_operator -> OrangeLegendaryCandidate (300000 activations)
e129_decimal_fraction_trace_operator -> OrangeLegendaryCandidate (300000 activations)
e129_parenthesized_mixed_precedence_operator -> OrangeLegendaryCandidate (300000 activations)
e129_invalid_trace_rejection_guard -> OrangeLegendaryCandidate (300000 activations)
e129_division_by_zero_guard -> OrangeLegendaryCandidate (300000 activations)
```

## Interpretation

E129 confirms that direct arithmetic training data can be promoted into
scoped arithmetic Operator knowledge when it is framed as exact compute
and trace validation rather than freeform word-problem reasoning.

The result covers plus/minus, multiplication, exact division, floor
division, signed integer arithmetic, decimal/fraction rendering, mixed
precedence, invalid-trace rejection, and division-by-zero rejection.

The claim remains scoped: these operators compute or validate visible
arithmetic expressions and traces. They do not solve hidden natural
language math problems without a visible arithmetic expression/trace.
