# E134 External Math Text OOD Route Stress And Counterexample Gauntlet Result

```text
decision = e134_external_math_text_ood_route_stress_counterexample_confirmed
next     = E135_MATH_TEXT_MULTI_ROUTE_ASSISTANT_DIALOGUE_STATE_GAUNTLET
```

E134 confirms that the 16 E133 math-text route-composition operators survive
OOD wrapper stress and counterexample lures while keeping the no-solve boundary
intact.

## OOD Result

```text
operator_count = 16
ood_pass_operator_count = 16 / 16
ood_case_count_total = 208,000
qualified_ood_route_activation_total = 208,000
qualified_ood_route_activation_min = 13,000

visible_arithmetic_ood_case_count_total = 11,875
structural_guard_ood_case_count_total = 153,125
hidden_word_problem_ood_no_solve_case_count_total = 43,000
counterexample_case_count_total = 48,000

ood_route_accuracy_min = 1.000
visible_arithmetic_ood_accuracy_min = 1.000
structural_guard_ood_accuracy_min = 1.000
hidden_word_problem_ood_no_solve_accuracy_min = 1.000
counterexample_accuracy_min = 1.000
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
e133_baseline_ood_miss_total = 36,275
overbroad_solver_control_wrong_scope_call_total = 19,200
overbroad_solver_control_false_commit_total = 19,200
trust_control_false_commit_total = 4,200
trust_control_boundary_claim_violation_total = 4,200
trust_control_direct_flow_write_total = 2,400
```

## Interpretation

E134 extends E133 from clean route composition to noisy/OOD route stress:

```text
OOD visible arithmetic wrappers
-> scoped visible arithmetic route

proof / TIR / matrix / geometry / unit / answer-format OOD lures
-> bounded guard, preserve, reject, or defer route

wrong boxed answers, spoofed TIR outputs, and conflicting final answers
-> reject unsafe trust

hidden prose-only word problems
-> no-call
```

The important falsification check is that the E133 baseline missed 36,275 OOD
cases that E134 covered, while the E134 candidate route layer kept all tracked
safety counters at zero.

This is not math benchmark solving. It confirms OOD route robustness and
counterexample rejection for scoped math-text lenses/guards without allowing
boxed-answer trust, TIR-output trust, direct Flow writes, or hidden prose
word-problem solving.
