# D123 Nested Instruction Repair Prototype with Sequence Guardrails Result

Expected result: d123_nested_instruction_repair_prototype_confirmed with next=D124_NESTED_INSTRUCTION_REPAIR_SCALE_CONFIRM_WITH_SEQUENCE_GUARDRAILS when D122 handoff is valid and D123 prototype gates pass.

Scale snapshot: requested_total_rows=171360, actual_total_rows=171360, scale_reduced=false, stress_mode_count=37, fallback_rows=0, and failed_jobs=[].

Training snapshot: repair_training_executed=true, training_updates_executed=true, total_repair_steps_executed=240, epochs_executed=2, trainable_adapter_names=[halting_head_adapter_delta, route_head_adapter_delta, calibration_scalar_adapter_delta], recurrent_state_adapter_updated=false, checkpoint_count=12, failed_checkpoint_count=0, rollback_triggered=false, and final_candidate_selected=true.

Nested repair snapshot: nested_failure_rate_before=0.041, nested_failure_rate_after=0.035, nested_failure_reduction=0.146, nested_true_network_failure_rate_before=0.037, nested_true_network_failure_rate_after=0.031, nested_route_stack_failure_rate_before=0.038, nested_route_stack_failure_rate_after=0.033, nested_route_stack_failure_reduction=0.132, nested_scope_resolution_failure_rate_after=0.029, nested_binding_scope_drift_rate_after=0.026, nested_halting_margin_floor_after=0.034, nested_route_uncertainty_after=0.050, nested_route_uncertainty_reduction=0.123, and repair_signal_positive=true.

Depth/consistency snapshot: route_stack_margin_depth2_after=0.059, route_stack_margin_depth3_after=0.044, route_stack_margin_depth4_plus_after=0.027, binding_consistency_depth2_after=0.976, binding_consistency_depth3_after=0.956, binding_consistency_depth4_plus_after=0.928, and depth4_cliff_detected=false.

Subfamily snapshot: NESTED_DEPTH_2_INSTRUCTION_FAMILY, NESTED_DEPTH_3_INSTRUCTION_FAMILY, NESTED_ROUTE_STACK_FAMILY, and NESTED_SCOPE_RESOLUTION_FAMILY are guarded_low_weight and excluded from full healthy claims; NESTED_DEPTH_4_PLUS_INSTRUCTION_FAMILY, NESTED_CONDITIONAL_BINDING_FAMILY, NESTED_STOP_CONTINUE_BOUNDARY_FAMILY, and all adversarial-template families remain reference_only and excluded from healthy claims.

Preservation snapshot: long_sequence_guarded_low_weight_preserved=true, long_sequence_halting_risk=0.051, long_sequence_shortcut_risk=0.095, two_step_preserved=true, three_step_preserved=true, four_step_preserved=true, variable_binding_preserved=true, conditional_branch_preserved=true, bridge_baseline_preserved=true, trig_guardrails_preserved=true, lane_a_D68_preservation_rate=1.0, post_repair_rust_path_invoked=true, fallback_rows=0, and failed_jobs=[].

Boundary reminder: D123 is an adapter-only controlled nested instruction repair prototype; it does not perform natural-language pretraining, introduce tokenizers or next-token objectives, use raw text or raw Raven, train a Gemma-class model, or prove AGI/production readiness.
