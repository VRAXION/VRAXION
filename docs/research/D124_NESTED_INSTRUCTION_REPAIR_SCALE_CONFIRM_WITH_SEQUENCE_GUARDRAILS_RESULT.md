# D124 Nested Instruction Repair Scale Confirm with Sequence Guardrails Result

Expected result: d124_nested_instruction_repair_scale_confirmed with next=D125_ADVERSARIAL_TEMPLATE_OVERLAP_DEEP_FORENSICS_AND_REPAIR_PLAN_WITH_SEQUENCE_GUARDRAILS when D123 handoff is valid and D124 scale gates pass.

Scale snapshot: requested_total_rows=243900, actual_total_rows=243900, scale_reduced=false, stress_mode_count=38, fallback_rows=0, and failed_jobs=[].

Training snapshot: repair_scale_training_executed=true, training_updates_executed=true, total_repair_steps_executed=480, epochs_executed=3, trainable_adapter_names=[halting_head_adapter_delta, route_head_adapter_delta, calibration_scalar_adapter_delta], recurrent_state_adapter_updated=false, checkpoint_count=12, failed_checkpoint_count=0, rollback_triggered=false, and final_candidate_selected=true.

Nested scale snapshot: nested_failure_rate_before=0.041, nested_failure_rate_after=0.034, nested_failure_reduction=0.171, nested_true_network_failure_rate_before=0.037, nested_true_network_failure_rate_after=0.030, nested_route_stack_failure_rate_before=0.038, nested_route_stack_failure_rate_after=0.032, nested_route_stack_failure_reduction=0.158, nested_scope_resolution_failure_rate_after=0.028, nested_binding_scope_drift_rate_after=0.025, nested_halting_margin_floor_after=0.036, nested_route_uncertainty_after=0.049, nested_route_uncertainty_reduction=0.140, and repair_signal_positive=true.

Depth/consistency snapshot: route_stack_margin_depth2_after=0.060, route_stack_margin_depth3_after=0.046, route_stack_margin_depth4_plus_after=0.028, binding_consistency_depth2_after=0.977, binding_consistency_depth3_after=0.958, binding_consistency_depth4_plus_after=0.929, depth4_cliff_detected=false, and depth4_cliff_worsened=false.

Subfamily snapshot: NESTED_DEPTH_2_INSTRUCTION_FAMILY, NESTED_DEPTH_3_INSTRUCTION_FAMILY, NESTED_ROUTE_STACK_FAMILY, and NESTED_SCOPE_RESOLUTION_FAMILY remain guarded_low_weight and excluded from full healthy claims; NESTED_DEPTH_4_PLUS_INSTRUCTION_FAMILY, NESTED_CONDITIONAL_BINDING_FAMILY, NESTED_STOP_CONTINUE_BOUNDARY_FAMILY, and all adversarial-template families remain reference_only and excluded from healthy claims.

Preservation snapshot: long_sequence_guarded_low_weight_preserved=true, long_sequence_halting_risk=0.051, long_sequence_shortcut_risk=0.095, two_step_preserved=true, three_step_preserved=true, four_step_preserved=true, variable_binding_preserved=true, conditional_branch_preserved=true, bridge_baseline_preserved=true, trig_guardrails_preserved=true, lane_a_D68_preservation_rate=1.0, post_repair_rust_path_invoked=true, fallback_rows=0, and failed_jobs=[].

Boundary reminder: D124 is an adapter-only controlled nested instruction repair scale-confirmation run; it does not perform natural-language pretraining, introduce tokenizers or next-token objectives, use raw text or raw Raven, train a Gemma-class model, or prove AGI/production readiness.
