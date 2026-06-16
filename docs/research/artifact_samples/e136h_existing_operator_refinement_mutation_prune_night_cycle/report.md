# E136H Existing Operator Refinement Mutation/Prune Night Cycle

```text
decision = e136h_existing_operator_refinement_mutation_prune_confirmed
next     = E136I_OPERATOR_SUPERSESSION_AND_OUTPUT_LEDGER_PLANNING
```

## Metrics

```text
cycles_completed = 40
operator_count = 34
rows_seen_total = 12480000
current_activation_total = 3373788
selected_activation_total = 2891151
pruned_activation_total = 482637
mean_label_alignment = 0.701638
verified_label_count = 16
tentative_tighten_count = 11
abstract_but_useful_count = 7
hold_for_more_evidence_count = 0
hard_negative_total = 0
direct_flow_write_total = 0
```

## Interpretation

This run refines existing operators by separating kernel utility from the
human semantic label. Useful but low-alignment operators are preserved as
abstract/tentative kernels instead of being destructively pruned.

## Selected Operators

### answer_format_instruction_lens

```text
source = E132
display_name = Answer Format Instruction Lens
label_status = abstract_but_useful
selected_variant = abstract_kernel_shadow
current_activation = 149627
selected_activation = 149627
selected_prune_ratio = 0.000000
alignment = 0.013955
kernel_value_score = 1.000000
```

### equation_system_alignment_lens

```text
source = E132
display_name = Equation System Alignment Lens
label_status = abstract_but_useful
selected_variant = abstract_kernel_shadow
current_activation = 116424
selected_activation = 116424
selected_prune_ratio = 0.000000
alignment = 0.116703
kernel_value_score = 1.000000
```

### fraction_ratio_probability_lens

```text
source = E132
display_name = Fraction / Ratio / Probability Lens
label_status = abstract_but_useful
selected_variant = abstract_kernel_shadow
current_activation = 240000
selected_activation = 240000
selected_prune_ratio = 0.000000
alignment = 0.271900
kernel_value_score = 1.000000
```

### piecewise_case_function_lens

```text
source = E132
display_name = Piecewise / Case Function Lens
label_status = abstract_but_useful
selected_variant = abstract_kernel_shadow
current_activation = 135991
selected_activation = 135991
selected_prune_ratio = 0.000000
alignment = 0.249708
kernel_value_score = 1.000000
```

### summation_sequence_series_lens

```text
source = E132
display_name = Summation / Sequence / Series Lens
label_status = abstract_but_useful
selected_variant = abstract_kernel_shadow
current_activation = 106688
selected_activation = 106688
selected_prune_ratio = 0.000000
alignment = 0.152529
kernel_value_score = 1.000000
```

### unit_quantity_binding_lens

```text
source = E132
display_name = Unit Quantity Binding Lens
label_status = abstract_but_useful
selected_variant = abstract_kernel_shadow
current_activation = 149541
selected_activation = 149541
selected_prune_ratio = 0.000000
alignment = 0.113909
kernel_value_score = 1.000000
```

### variable_definition_binding_lens

```text
source = E132
display_name = Variable Definition Binding Lens
label_status = abstract_but_useful
selected_variant = abstract_kernel_shadow
current_activation = 137986
selected_activation = 137986
selected_prune_ratio = 0.000000
alignment = 0.297037
kernel_value_score = 1.000000
```

### assistant_comparison_evaluation_lens

```text
source = E136A
display_name = Assistant Comparison / Evaluation Lens
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 32158
selected_activation = 21735
selected_prune_ratio = 0.324118
alignment = 0.675882
kernel_value_score = 0.860362
```

### assistant_longform_generation_request_lens

```text
source = E136A
display_name = Assistant Longform Generation Request Lens
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 106916
selected_activation = 83841
selected_prune_ratio = 0.215824
alignment = 0.784176
kernel_value_score = 1.000000
```

### assistant_math_text_no_solve_guard

```text
source = E136A
display_name = Assistant Math Text No-Solve Guard
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 66098
selected_activation = 49671
selected_prune_ratio = 0.248525
alignment = 0.751475
kernel_value_score = 0.986144
```

### assistant_role_turn_boundary_lens

```text
source = E136A
display_name = Assistant Role / Turn Boundary Lens
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 143336
selected_activation = 88662
selected_prune_ratio = 0.381439
alignment = 0.618561
kernel_value_score = 1.000000
```

### assistant_safety_sensitive_domain_guard

```text
source = E136A
display_name = Assistant Safety-Sensitive Domain Guard
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 33258
selected_activation = 22906
selected_prune_ratio = 0.311263
alignment = 0.688737
kernel_value_score = 0.865224
```

### assistant_tir_output_error_repair_guard

```text
source = E132
display_name = Assistant TIR Output / Error Repair Guard
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 100175
selected_activation = 72649
selected_prune_ratio = 0.274779
alignment = 0.725221
kernel_value_score = 1.000000
```

### code_instruction_boundary_lens

```text
source = E136A
display_name = Code Instruction Boundary Lens
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 73490
selected_activation = 47378
selected_prune_ratio = 0.355314
alignment = 0.644686
kernel_value_score = 1.000000
```

### latex_display_math_block_lens

```text
source = E132
display_name = LaTeX Display Math Block Lens
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 104330
selected_activation = 53961
selected_prune_ratio = 0.482785
alignment = 0.517215
kernel_value_score = 1.000000
```

### multi_turn_context_continuity_lens

```text
source = E136A
display_name = Multi-Turn Context Continuity Lens
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 107088
selected_activation = 70427
selected_prune_ratio = 0.342345
alignment = 0.657655
kernel_value_score = 1.000000
```

### response_format_constraint_lens

```text
source = E136A
display_name = Response Format Constraint Lens
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 247090
selected_activation = 124225
selected_prune_ratio = 0.497248
alignment = 0.502752
kernel_value_score = 1.000000
```

### tir_python_block_boundary_lens

```text
source = E132
display_name = TIR Python Block Boundary Lens
label_status = tentative_label_tighten_trigger
selected_variant = semantic_tightened_trigger
current_activation = 150666
selected_activation = 105010
selected_prune_ratio = 0.303028
alignment = 0.696972
kernel_value_score = 1.000000
```

### assistant_translation_multilingual_lens

```text
source = E136A
display_name = Assistant Translation / Multilingual Lens
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 53873
selected_activation = 51172
selected_prune_ratio = 0.050136
alignment = 0.949864
kernel_value_score = 0.945043
```

### boxed_answer_boundary_lens

```text
source = E132
display_name = Boxed Answer Boundary Lens
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 149454
selected_activation = 124742
selected_prune_ratio = 0.165349
alignment = 0.834651
kernel_value_score = 1.000000
```

### geometry_diagram_reference_guard

```text
source = E132
display_name = Geometry Diagram Reference Guard
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 21785
selected_activation = 19196
selected_prune_ratio = 0.118843
alignment = 0.881157
kernel_value_score = 0.897092
```

### helpful_harmless_preference_guard

```text
source = E136A
display_name = Helpful / Harmless Preference Guard
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 4978
selected_activation = 4978
selected_prune_ratio = 0.000000
alignment = 1.000000
kernel_value_score = 0.661134
```

### human_written_instruction_style_lens

```text
source = E136A
display_name = Human-Written Instruction Style Lens
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 4866
selected_activation = 4866
selected_prune_ratio = 0.000000
alignment = 1.000000
kernel_value_score = 0.659161
```

### instruction_task_decomposition_lens

```text
source = E136A
display_name = Instruction Task Decomposition Lens
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 198577
selected_activation = 186102
selected_prune_ratio = 0.062822
alignment = 0.937178
kernel_value_score = 1.000000
```

### latex_inline_math_boundary_lens

```text
source = E132
display_name = LaTeX Inline Math Boundary Lens
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 102596
selected_activation = 101349
selected_prune_ratio = 0.012154
alignment = 0.987846
kernel_value_score = 1.000000
```

### matrix_vector_block_lens

```text
source = E132
display_name = Matrix / Vector Block Lens
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 6251
selected_activation = 5754
selected_prune_ratio = 0.079507
alignment = 0.920493
kernel_value_score = 0.731598
```

### proof_step_connector_lens

```text
source = E132
display_name = Proof Step Connector Lens
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 76959
selected_activation = 72115
selected_prune_ratio = 0.062943
alignment = 0.937057
kernel_value_score = 1.000000
```

### reasoning_instruction_lens

```text
source = E136A
display_name = Reasoning Instruction Lens
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 75707
selected_activation = 75707
selected_prune_ratio = 0.000000
alignment = 1.000000
kernel_value_score = 1.000000
```

### refusal_boundary_guard

```text
source = E136A
display_name = Refusal Boundary Guard
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 10974
selected_activation = 10970
selected_prune_ratio = 0.000364
alignment = 0.999636
kernel_value_score = 0.734046
```

### rejected_response_contrast_lens

```text
source = E136A
display_name = Rejected Response Contrast Lens
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 60000
selected_activation = 60000
selected_prune_ratio = 0.000000
alignment = 1.000000
kernel_value_score = 0.966041
```

### source_absence_defer_guard

```text
source = E136A
display_name = Source Absence Defer Guard
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 5386
selected_activation = 5383
selected_prune_ratio = 0.000557
alignment = 0.999443
kernel_value_score = 0.668004
```

### summarization_request_lens

```text
source = E136A
display_name = Summarization Request Lens
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 37139
selected_activation = 36508
selected_prune_ratio = 0.016990
alignment = 0.983010
kernel_value_score = 0.881747
```

### synthetic_dialogue_noise_guard

```text
source = E136A
display_name = Synthetic Dialogue Noise Guard
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 200668
selected_activation = 200668
selected_prune_ratio = 0.000000
alignment = 1.000000
kernel_value_score = 1.000000
```

### word_problem_no_solve_guard_v2

```text
source = E132
display_name = Word Problem No-Solve Guard V2
label_status = verified_label
selected_variant = semantic_verified_pruned
current_activation = 163713
selected_activation = 154919
selected_prune_ratio = 0.053716
alignment = 0.946284
kernel_value_score = 1.000000
```

## Boundary

This is an operator-governance refinement artifact. It is not a claim of
open-domain assistant behavior, new neural weights, or destructive removal
from the committed runtime library.
