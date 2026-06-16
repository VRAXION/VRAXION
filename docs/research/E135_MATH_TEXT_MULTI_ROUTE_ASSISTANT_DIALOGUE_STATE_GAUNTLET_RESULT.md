# E135 Math Text Multi-Route Assistant Dialogue-State Gauntlet Result

```text
decision = e135_math_text_multi_route_dialogue_state_confirmed
next     = E136_ASSISTANT_MATH_TEXT_DIALOGUE_ROUTE_TRANSFER_AND_LATENCY_COMPARE
```

E135 confirms that the 16 E134 math-text route operators keep current-turn route
state stable across controlled multi-turn assistant dialogue surfaces.

## Dialogue Result

```text
operator_count = 16
dialogue_pass_operator_count = 16 / 16
dialogue_case_count_total = 136,000
dialogue_turn_count_total = 367,400
qualified_dialogue_route_activation_total = 136,000
qualified_dialogue_route_activation_min = 8,500

hidden_word_problem_dialogue_no_solve_case_count_total = 29,500
visible_reentry_dialogue_case_count_total = 10,500
stale_route_rejection_case_count_total = 22,400
cross_thread_rejection_case_count_total = 11,200
counterexample_dialogue_case_count_total = 76,500

dialogue_state_accuracy_min = 1.000
current_turn_route_accuracy_min = 1.000
route_state_integrity_min = 1.000
all_turn_route_accuracy_min = 1.000
hidden_word_problem_dialogue_no_solve_accuracy_min = 1.000
visible_reentry_dialogue_accuracy_min = 1.000
stale_route_rejection_accuracy_min = 1.000
cross_thread_rejection_accuracy_min = 1.000
counterexample_dialogue_accuracy_min = 1.000
```

Safety:

```text
hard_negative_total = 0
wrong_scope_call_total = 0
false_commit_total = 0
unsupported_answer_total = 0
boundary_claim_violation_total = 0
direct_flow_write_total = 0
stale_route_reuse_total = 0
cross_thread_contamination_total = 0
```

Controls:

```text
latest_route_reuse_control_failure_total = 1,620
stale_route_reuse_control_failure_total = 2,880
cross_thread_contamination_control_failure_total = 1,440
counterexample_trust_control_failure_total = 6,750
single_turn_reset_control_failure_total = 14,400
```

## Interpretation

E135 extends E134 from single-turn OOD route robustness to controlled
multi-turn route-state robustness:

```text
prior hidden no-call -> current visible arithmetic route
prior visible/structural route -> current hidden no-call
prior route -> stale lure rejected
cross-thread route -> ignored
counterexample turn -> guarded, not trusted
new cycle -> current route wins
```

This is not open-domain dialogue. It proves that the scoped math-text route
layer can preserve current-turn route state in controlled assistant dialogue
proxies without allowing stale route reuse, cross-thread contamination,
counterexample trust, direct Flow writes, or hidden prose word-problem solving.
