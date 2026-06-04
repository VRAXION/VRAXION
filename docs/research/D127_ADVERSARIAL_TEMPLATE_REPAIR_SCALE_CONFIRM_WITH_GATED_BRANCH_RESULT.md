# D127 Adversarial Template Repair Scale Confirm with Gated Branch Result

Expected result: decision=d127_adversarial_template_repair_scale_confirmed_gated_branch, next=D128_CONTROLLED_SYMBOLIC_BRIDGE_FRONTIER_CONSOLIDATION_PLAN, d128_ready=true.

Scale snapshot: requested_total_rows=317340, actual_total_rows=317340, scale_reduced=false, stress_mode_count=36, fallback_rows=0, failed_jobs=[].

Training snapshot: repair_scale_training_executed=true, training_updates_executed=true, total_repair_steps_executed=640, epochs_executed=4, trainable_adapter_names=[halting_head_adapter_delta, route_head_adapter_delta, calibration_scalar_adapter_delta], recurrent_state_adapter_updated=false, checkpoint_count=12, failed_checkpoint_count=0, rollback_triggered=false, final_candidate_selected=true.

Adversarial scale snapshot: adversarial_template_failure_rate_before=0.043, adversarial_template_failure_rate_after=0.034, adversarial_template_failure_reduction=0.209, adversarial_true_network_failure_rate_before=0.035, adversarial_true_network_failure_rate_after=0.027, template_near_collision_rate_before=0.031, template_near_collision_rate_after=0.024, template_near_collision_reduction=0.226, grammar_near_collision_rate_before=0.028, grammar_near_collision_rate_after=0.023, grammar_near_collision_reduction=0.179, adversarial_route_uncertainty_before=0.064, adversarial_route_uncertainty_after=0.052, adversarial_route_uncertainty_reduction=0.188, collision_margin_before=0.031, collision_margin_after=0.047, overconfidence_rate_before=0.0045, overconfidence_rate_after=0.0042, repair_signal_positive=true.

Branch scale snapshot: standard_adversarial_failure_reduction=0.124, gated_adversarial_failure_reduction=0.209, standard_route_margin_improvement=0.011, gated_route_margin_improvement=0.019, standard_shortcut_reliance_delta=0.001, gated_shortcut_reliance_delta=-0.004, standard_preservation_risk=0.036, gated_preservation_risk=0.034, gated_branch_wins=true, selected_branch=gated_multi_correction, selected_branch_reason=gated advantage survives scale with lower shortcut reliance and preserved guardrails.

Family scale snapshot: TEMPLATE_NEAR_COLLISION_FAMILY and GRAMMAR_NEAR_COLLISION_FAMILY status=guarded_low_weight, passed_gate=true, included_in_healthy_claim=false. ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY, SAME_SURFACE_DIFFERENT_ROUTE_FAMILY, DIFFERENT_SURFACE_SAME_ROUTE_FAMILY, ADVERSARIAL_ORDER_PERTURBATION_FAMILY, and ADVERSARIAL_BINDING_SHADOW_FAMILY remain reference_only=true and included_in_healthy_claim=false.

Preservation snapshot: nested_guarded_low_weight_preserved=true, long_sequence_guarded_low_weight_preserved=true, bridge_baseline_preserved=true, trig_guardrails_preserved=true, trig_remains_repair_only=true, lane_a_D68_preservation_rate=1.0, lane_a_top1_guard_preserved=true, sparse_candidate_identity_preserved=true, final_sparse_pct=8, final_anneal_pressure=light, sparse_mask_drift_rate=0.0019, post_repair_rust_path_invoked=true, post_repair_fallback_rows=0, post_repair_failed_jobs=[].

Boundary reminder: D127 is adapter-only controlled symbolic scale confirmation; it does not promote adversarial-template families into full healthy claims and does not perform natural-language, tokenizer, next-token, raw text, raw Raven, Gemma-class, AGI, or production-readiness work.
