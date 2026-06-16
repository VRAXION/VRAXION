# E135 Math Text Multi-Route Assistant Dialogue-State Gauntlet Result

```text
decision = e135_math_text_multi_route_dialogue_state_confirmed
next = E136_ASSISTANT_MATH_TEXT_DIALOGUE_ROUTE_TRANSFER_AND_LATENCY_COMPARE
boundary = controlled multi-route dialogue-state proxy only; not open-domain dialogue

dataset_rows_loaded = 215051
operator_count = 16
dialogue_pass_operator_count = 16
dialogue_case_count_total = 136000
dialogue_turn_count_total = 367400
hidden_word_problem_dialogue_no_solve_case_count_total = 29500
visible_reentry_dialogue_case_count_total = 10500
stale_route_rejection_case_count_total = 22400
cross_thread_rejection_case_count_total = 11200
counterexample_dialogue_case_count_total = 76500
dialogue_state_accuracy_min = 1.000
current_turn_route_accuracy_min = 1.000
route_state_integrity_min = 1.000
hidden_word_problem_dialogue_no_solve_accuracy_min = 1.000
counterexample_dialogue_accuracy_min = 1.000

hard_negative_total = 0
wrong_scope_call_total = 0
false_commit_total = 0
unsupported_answer_total = 0
boundary_claim_violation_total = 0
direct_flow_write_total = 0
stale_route_reuse_total = 0
cross_thread_contamination_total = 0
```

## Summary

E135 confirms that the E134 math-text route layer keeps current-turn
route state stable across controlled multi-turn assistant dialogue
surfaces. Hidden prose-only word problems remain no-call; stale,
cross-thread, and counterexample turns do not contaminate the active route.

## Operator Results

```text
answer_format_instruction_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
assistant_tir_output_error_repair_guard -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
boxed_answer_boundary_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
equation_system_alignment_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
fraction_ratio_probability_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
geometry_diagram_reference_guard -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
latex_display_math_block_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
latex_inline_math_boundary_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
matrix_vector_block_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
piecewise_case_function_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
proof_step_connector_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
summation_sequence_series_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
tir_python_block_boundary_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
unit_quantity_binding_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
variable_definition_binding_lens -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
word_problem_no_solve_guard_v2 -> E135DialogueStateConfirmed (dialogue=1.000, active=1.000, state=1.000)
```
