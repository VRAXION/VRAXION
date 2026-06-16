# E136B Assistant Text Route Composition And Boundary Confirm Result

```text
decision = e136b_assistant_text_route_composition_boundary_confirmed
next     = E136C_ASSISTANT_TEXT_MULTI_TURN_ROUTE_STATE_AND_LATENCY_COMPARE
```

E136B confirms that the 18 scoped assistant/text Operators promoted in E136A
compose into bounded route stacks and preserve assistant/text boundaries under
controlled route, boundary, negative-scope, and control pressure.

## Result

```text
source_e136a_decision = e136a_assistant_text_skill_farm_mutation_prune_orange_cycle_confirmed
dataset_rows_loaded = 447,766
route_seed_row_count = 4,096
external_source_count = 5
external_family_count = 12

operator_count = 18
route_pass_operator_count = 18 / 18
route_case_count_total = 144,000
multi_route_composition_case_count_total = 53,000
boundary_case_count_total = 72,000
negative_scope_case_count_total = 18,000
qualified_route_activation_total = 144,000
qualified_route_activation_min = 8,000
route_family_count = 10
```

Accuracy:

```text
route_accuracy_min = 1.000
route_stack_accuracy_min = 1.000
primary_route_accuracy_min = 1.000
boundary_accuracy_min = 1.000
multi_route_composition_accuracy_min = 1.000
boundary_case_accuracy_min = 1.000
negative_scope_accuracy_min = 1.000
```

Safety:

```text
hard_negative_total = 0
wrong_scope_call_total = 0
false_commit_total = 0
unsupported_answer_total = 0
boundary_claim_violation_total = 0
direct_flow_write_total = 0
```

Controls:

```text
overbroad_chatbot_control_wrong_scope_call_total = 14,400
overbroad_chatbot_control_unsupported_answer_total = 14,400
unsafe_direct_write_control_false_commit_total = 14,400
unsafe_direct_write_control_direct_flow_write_total = 14,400
source_hallucination_control_false_commit_total = 14,400
source_hallucination_control_unsupported_answer_total = 14,400
rejected_response_reuse_control_false_commit_total = 800
single_operator_drop_control_false_commit_total = 4,806
```

## Interpretation

E136B extends E136A from "new scoped assistant/text Operators exist" to "those
Operators can compose into bounded assistant/text route decisions." The tested
route shape is:

```text
assistant text input
-> primary scoped route
-> auxiliary route boundaries
-> Proposal Field route output
-> no direct Flow write
```

This is still controlled route-composition evidence. It is not open-domain
assistant readiness, neural training, production deployment, general reasoning,
or a claim that freeform generation has been learned.
