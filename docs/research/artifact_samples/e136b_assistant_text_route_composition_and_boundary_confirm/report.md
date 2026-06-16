# E136B Assistant Text Route Composition And Boundary Confirm Result

```text
decision = e136b_assistant_text_route_composition_boundary_confirmed
next = E136C_ASSISTANT_TEXT_MULTI_TURN_ROUTE_STATE_AND_LATENCY_COMPARE
boundary = controlled assistant/text route composition only; not neural training or open-domain assistant readiness

operator_count = 18
route_pass_operator_count = 18
route_case_count_total = 144000
multi_route_composition_case_count_total = 53000
boundary_case_count_total = 72000
negative_scope_case_count_total = 18000
qualified_route_activation_total = 144000
qualified_route_activation_min = 8000

route_accuracy_min = 1.0
route_stack_accuracy_min = 1.0
primary_route_accuracy_min = 1.0
boundary_accuracy_min = 1.0
negative_scope_accuracy_min = 1.0

hard_negative_total = 0
wrong_scope_call_total = 0
false_commit_total = 0
unsupported_answer_total = 0
boundary_claim_violation_total = 0
direct_flow_write_total = 0

overbroad_chatbot_control_wrong_scope_call_total = 14400
unsafe_direct_write_control_direct_flow_write_total = 14400
source_hallucination_control_unsupported_answer_total = 14400
```

## Operator Results

```text
assistant_role_turn_boundary_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
multi_turn_context_continuity_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
instruction_task_decomposition_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
summarization_request_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
code_instruction_boundary_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
refusal_boundary_guard -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
helpful_harmless_preference_guard -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
rejected_response_contrast_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
source_absence_defer_guard -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
response_format_constraint_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
human_written_instruction_style_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
synthetic_dialogue_noise_guard -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
reasoning_instruction_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
assistant_math_text_no_solve_guard -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
assistant_safety_sensitive_domain_guard -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
assistant_longform_generation_request_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
assistant_comparison_evaluation_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
assistant_translation_multilingual_lens -> E136BAssistantTextRouteBoundaryConfirmed (cases=8000, stack=1.0, boundary=1.0)
```
