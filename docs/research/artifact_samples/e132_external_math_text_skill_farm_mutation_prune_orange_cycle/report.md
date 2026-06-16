# E132 External Math Text Skill Farm Mutation/Prune Orange Cycle

```text
decision = e132_external_math_text_skill_farm_mutation_prune_orange_cycle_confirmed
next     = E133_MATH_TEXT_ROUTE_COMPOSITION_AND_NO_SOLVE_ASSISTANT_CONFIRM
dataset_rows_loaded = 215051
operator_count = 16
orange_legendary_candidate_count = 16
external_support_min = 5953
qualified_activation_min = 302510
negative_scope_case_count_total = 78859
hard_negative_total = 0
wrong_scope_call_total = 0
overbroad_solver_control_wrong_scope_call_total = 16703
```

Boundary: scoped math-text lenses and guards only. This is not GSM8K/MATH solving, not open-domain word-problem solving, not neural training, and not Core/PermaCore/TrueGolden.

## Promoted Operators

- `latex_inline_math_boundary_lens` -> OrangeLegendaryCandidate (support=100574, prune=0.8, variant=orange_pruned_minimal)
- `latex_display_math_block_lens` -> OrangeLegendaryCandidate (support=102106, prune=0.78, variant=orange_pruned_minimal)
- `boxed_answer_boundary_lens` -> OrangeLegendaryCandidate (support=142231, prune=0.68, variant=orange_pruned_minimal)
- `tir_python_block_boundary_lens` -> OrangeLegendaryCandidate (support=138705, prune=0.78, variant=orange_pruned_minimal)
- `proof_step_connector_lens` -> OrangeLegendaryCandidate (support=73683, prune=0.72, variant=orange_pruned_minimal)
- `geometry_diagram_reference_guard` -> OrangeLegendaryCandidate (support=20835, prune=0.78, variant=orange_pruned_minimal)
- `matrix_vector_block_lens` -> OrangeLegendaryCandidate (support=5953, prune=0.69, variant=orange_pruned_minimal)
- `equation_system_alignment_lens` -> OrangeLegendaryCandidate (support=112651, prune=0.74, variant=orange_pruned_minimal)
- `piecewise_case_function_lens` -> OrangeLegendaryCandidate (support=129584, prune=0.71, variant=orange_pruned_minimal)
- `fraction_ratio_probability_lens` -> OrangeLegendaryCandidate (support=170835, prune=0.8, variant=orange_pruned_minimal)
- `variable_definition_binding_lens` -> OrangeLegendaryCandidate (support=130714, prune=0.71, variant=orange_pruned_minimal)
- `summation_sequence_series_lens` -> OrangeLegendaryCandidate (support=103758, prune=0.74, variant=orange_pruned_minimal)
- `unit_quantity_binding_lens` -> OrangeLegendaryCandidate (support=136532, prune=0.68, variant=orange_pruned_minimal)
- `word_problem_no_solve_guard_v2` -> OrangeLegendaryCandidate (support=151143, prune=0.73, variant=orange_pruned_minimal)
- `assistant_tir_output_error_repair_guard` -> OrangeLegendaryCandidate (support=97349, prune=0.76, variant=orange_pruned_minimal)
- `answer_format_instruction_lens` -> OrangeLegendaryCandidate (support=142384, prune=0.69, variant=orange_pruned_minimal)
