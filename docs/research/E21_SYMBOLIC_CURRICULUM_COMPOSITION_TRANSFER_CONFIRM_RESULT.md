# E21_SYMBOLIC_CURRICULUM_COMPOSITION_TRANSFER_CONFIRM Result

## Snapshot

- decision = e21_symbolic_curriculum_composition_transfer_confirmed.
- next = E22_SYMBOLIC_CURRICULUM_STRESS_OR_REAL_TOOL_USE_PLAN.
- primary_system = CURRICULUM_WITH_REUSABLE_POCKETS_PRUNED_PRIMARY.
- positive_gate_passed = true.
- checker_failure_count = 0.
- run_budget_class = full_budget.
- full_budget_met = true.
- runtime_minutes = 0.03719098598333706.

## Budget

- generations_completed = 140.
- population_size = 192.
- candidate_count_evaluated = 26880.
- checkpoint_count = 140.
- heldout_episode_count = 3000.
- stress_episode_count = 3000.
- locked_hard_pretest_episode_count = 1200.
- locked_hard_posttest_episode_count = 1200.
- curriculum_stage_count = 12.
- committed_sample_episode_count = 536.

## Key metrics

- locked_hard_pretest_accuracy = 0.4191666666666667.
- locked_hard_posttest_accuracy = 0.9041666666666667.
- improvement_vs_pretest = 0.485.
- new_heldout_hard_accuracy = 0.9103333333333333.
- heldout_composition_transfer_accuracy = 0.9103333333333333.
- earlier_stage_regression_average = 0.937.
- primitive_reuse_rate = 0.9085714285714286.
- learned_operator_count = 11.
- composition_depth_mean = 4.997142857142857.
- composition_depth_max = 8.
- delta_vs_monolithic_equal_budget = 0.21499999999999997.
- delta_vs_no_reusable_pocket_transfer_ablation = 0.32666666666666666.
- canonical_answer_accuracy = 0.9143333333333333.
- undefined_handling_accuracy = 0.9439252336448598.
- ambiguous_handling_accuracy = 0.9345794392523364.
- trace_validity = 0.9143333333333333.
- renderer_faithfulness = 1.0.
- deterministic_replay_match_rate = 1.0.

## Audit

- oracle_parse_tree_leakage_detected = false.
- oracle_answer_leakage_detected = false.
- direct_eval_usage_detected = false.
- sympy_usage_detected = false.
- hand_solver_primary_detected = false.
- static_metric_audit_passed = true.
- sample_only_checker_passed = true.
- target_checker_passed = true.

## Boundary

This is a controlled symbolic curriculum-composition transfer audit for a Flow/Pocket policy. It tests whether reusable primitive pockets learned through staged curriculum improve locked hard symbolic reasoning tasks. It does not prove general mathematics, theorem proving, GPT-like generation, AGI, consciousness, or production readiness.
