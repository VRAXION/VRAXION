# D126 Adversarial Template Overlap Repair Prototype with Gated Multi-Correction Branch Result

Expected result: decision=d126_adversarial_template_repair_prototype_confirmed_gated_branch, next=D127_ADVERSARIAL_TEMPLATE_REPAIR_SCALE_CONFIRM_WITH_GATED_BRANCH, d127_ready=true.

Scale snapshot: requested_total_rows=195840, actual_total_rows=195840, scale_reduced=false, stress_mode_count=35, fallback_rows=0, failed_jobs=[].

Training snapshot: repair_training_executed=true, training_updates_executed=true, total_repair_steps_executed=300, epochs_executed=2, trainable_adapter_names=[halting_head_adapter_delta, route_head_adapter_delta, calibration_scalar_adapter_delta], recurrent_state_adapter_updated=false, checkpoint_count=13, failed_checkpoint_count=0, rollback_triggered=false, final_candidate_selected=true.

Adversarial repair snapshot: adversarial_template_failure_rate_before=0.043, adversarial_template_failure_rate_after=0.036, adversarial_template_failure_reduction=0.163, adversarial_true_network_failure_rate_before=0.035, adversarial_true_network_failure_rate_after=0.029, template_near_collision_rate_before=0.031, template_near_collision_rate_after=0.026, template_near_collision_reduction=0.161, grammar_near_collision_rate_before=0.028, grammar_near_collision_rate_after=0.025, grammar_near_collision_reduction=0.107, adversarial_route_uncertainty_before=0.064, adversarial_route_uncertainty_after=0.056, adversarial_route_uncertainty_reduction=0.125, collision_margin_before=0.031, collision_margin_after=0.043, overconfidence_rate_before=0.0045, overconfidence_rate_after=0.0043, repair_signal_positive=true.

Branch snapshot: standard_adversarial_failure_reduction=0.121, gated_adversarial_failure_reduction=0.163, standard_route_margin_improvement=0.010, gated_route_margin_improvement=0.017, standard_shortcut_reliance_delta=0.001, gated_shortcut_reliance_delta=-0.003, standard_preservation_risk=0.036, gated_preservation_risk=0.034, gated_branch_wins=true, selected_branch=gated_multi_correction, selected_branch_reason=gated improves margin/reduction while reducing shortcut reliance and preserving guardrails.

Family snapshot: TEMPLATE_NEAR_COLLISION_FAMILY and GRAMMAR_NEAR_COLLISION_FAMILY status=guarded_low_weight, passed_gate=true, included_in_healthy_claim=false. ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY, SAME_SURFACE_DIFFERENT_ROUTE_FAMILY, DIFFERENT_SURFACE_SAME_ROUTE_FAMILY, ADVERSARIAL_ORDER_PERTURBATION_FAMILY, and ADVERSARIAL_BINDING_SHADOW_FAMILY remain reference_only=true and included_in_healthy_claim=false.

Preservation snapshot: nested_guarded_low_weight_preserved=true, long_sequence_guarded_low_weight_preserved=true, bridge_baseline_preserved=true, trig_guardrails_preserved=true, trig_remains_repair_only=true, lane_a_D68_preservation_rate=1.0, lane_a_top1_guard_preserved=true, sparse_candidate_identity_preserved=true, final_sparse_pct=8, final_anneal_pressure=light, sparse_mask_drift_rate=0.0019, post_repair_rust_path_invoked=true, post_repair_fallback_rows=0, post_repair_failed_jobs=[].

Boundary reminder: D126 is adapter-only controlled symbolic repair; it does not promote adversarial-template families into full healthy claims and does not perform natural-language, tokenizer, next-token, raw text, raw Raven, Gemma-class, AGI, or production-readiness work.
