# E130B Arithmetic Text-IO Transfer And Word-Problem No-Call Gauntlet Result

```text
decision = e130b_arithmetic_text_io_transfer_word_problem_no_call_confirmed
next = E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET
boundary = arithmetic text-IO transfer only; not word-problem solving

operator_count = 9
transfer_pass_operator_count = 9
visible_transfer_case_count_total = 270000
word_problem_no_call_case_count_total = 135000
qualified_transfer_activation_total = 270000
visible_transfer_accuracy_min = 1.000
word_problem_no_call_accuracy_min = 1.000

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0
overbroad_control_wrong_scope_call_total = 18000
```

## Summary

E130B confirms that the E129 scoped arithmetic trace operators transfer to
longer text-IO wrappers when an explicit arithmetic expression or trace is
visible. The selected route refuses hidden natural-language word problems
with no visible expression/trace.

## Boundary

This is not natural-language word-problem solving. A prompt with only prose
and quantities remains a no-call case.

## Operator Results

```text
e129_add_sub_trace_operator -> E130BTextIOTransferConfirmed (visible=1.000, word_no_call=1.000)
e129_decimal_fraction_trace_operator -> E130BTextIOTransferConfirmed (visible=1.000, word_no_call=1.000)
e129_division_by_zero_guard -> E130BTextIOTransferConfirmed (visible=1.000, word_no_call=1.000)
e129_exact_division_trace_operator -> E130BTextIOTransferConfirmed (visible=1.000, word_no_call=1.000)
e129_floor_division_trace_operator -> E130BTextIOTransferConfirmed (visible=1.000, word_no_call=1.000)
e129_invalid_trace_rejection_guard -> E130BTextIOTransferConfirmed (visible=1.000, word_no_call=1.000)
e129_multiplication_trace_operator -> E130BTextIOTransferConfirmed (visible=1.000, word_no_call=1.000)
e129_parenthesized_mixed_precedence_operator -> E130BTextIOTransferConfirmed (visible=1.000, word_no_call=1.000)
e129_signed_integer_trace_operator -> E130BTextIOTransferConfirmed (visible=1.000, word_no_call=1.000)
```
