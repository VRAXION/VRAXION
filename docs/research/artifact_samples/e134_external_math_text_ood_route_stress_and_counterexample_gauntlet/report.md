# E134 External Math Text OOD Route Stress And Counterexample Gauntlet Result

```text
decision = e134_external_math_text_ood_route_stress_counterexample_confirmed
next = E135_MATH_TEXT_MULTI_ROUTE_ASSISTANT_DIALOGUE_STATE_GAUNTLET
boundary = OOD route stress and counterexample rejection only; not math benchmark solving

dataset_rows_loaded = 215051
operator_count = 16
ood_pass_operator_count = 16
ood_case_count_total = 208000
visible_arithmetic_ood_case_count_total = 11875
structural_guard_ood_case_count_total = 153125
hidden_word_problem_ood_no_solve_case_count_total = 43000
counterexample_case_count_total = 48000
ood_route_accuracy_min = 1.000
visible_arithmetic_ood_accuracy_min = 1.000
structural_guard_ood_accuracy_min = 1.000
hidden_word_problem_ood_no_solve_accuracy_min = 1.000
counterexample_accuracy_min = 1.000

hard_negative_total = 0
wrong_scope_call_total = 0
false_commit_total = 0
unsupported_answer_total = 0
boundary_claim_violation_total = 0
direct_flow_write_total = 0

e133_baseline_ood_miss_total = 36275
overbroad_solver_control_wrong_scope_call_total = 19200
trust_control_false_commit_total = 4200
trust_control_direct_flow_write_total = 2400
```

## Summary

E134 confirms that the E133 route-composition layer survives OOD math-text
wrappers, counterexamples, lure text, and trust-control attacks while
preserving the no-solve boundary for hidden prose-only word problems.

## Operator Results

```text
answer_format_instruction_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
assistant_tir_output_error_repair_guard -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
boxed_answer_boundary_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
equation_system_alignment_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
fraction_ratio_probability_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
geometry_diagram_reference_guard -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
latex_display_math_block_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
latex_inline_math_boundary_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
matrix_vector_block_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
piecewise_case_function_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
proof_step_connector_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
summation_sequence_series_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
tir_python_block_boundary_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
unit_quantity_binding_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
variable_definition_binding_lens -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
word_problem_no_solve_guard_v2 -> E134OODRouteStressConfirmed (ood=1.000, counter=1.000, hidden=1.000)
```
