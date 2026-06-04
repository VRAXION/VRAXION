# D120 Long-Sequence Halting Repair Prototype with Sequence Guardrails Result

Expected result: d120_long_sequence_halting_repair_prototype_confirmed with next=D121_LONG_SEQUENCE_HALTING_REPAIR_SCALE_CONFIRM_WITH_SEQUENCE_GUARDRAILS when D119 handoff is valid and D120 gates pass.

Scale snapshot: requested_total_rows=126000, actual_total_rows=126000, scale_reduced=false, stress_mode_count=36, fallback_rows=0, failed_jobs=[].

Training snapshot: repair_training_executed=true, training_updates_executed=true, total_repair_steps_executed=240, epochs_executed=2, trainable_adapter_names=[halting_head_adapter_delta, route_head_adapter_delta, calibration_scalar_adapter_delta], recurrent_state_adapter_updated=false, checkpoint_count=13, failed_checkpoint_count=0, rollback_triggered=false, and final_candidate_selected=true.

Repair snapshot: long_sequence_failure_rate_before=0.046, long_sequence_failure_rate_after=0.038, long_sequence_failure_reduction=0.174, step5_halting_margin_floor_before=0.027, step5_halting_margin_floor_after=0.041, step6_halting_margin_floor_before=0.019, step6_halting_margin_floor_after=0.030, step7_plus_halting_margin_floor_before=0.014, step7_plus_halting_margin_floor_after=0.019, long_sequence_route_uncertainty_reduction=0.148, calibration_tail_decay_reduction=0.136, overconfidence_rate_after=0.0042, and repair_signal_positive=true.

Failure-cliff snapshot: failure_cliff_shift_detected=false, failure_cliff_true_stabilization_score=0.68, step6_or_step7_cliff_worsened=false, residual_failure_rate_before=0.032, residual_failure_rate_after=0.027, residual_failure_reduction=0.156, residual_long_sequence_failure_rate_after=0.038, residual_nested_failure_rate_after=0.041, and residual_adversarial_template_failure_rate_after=0.043.

Preservation snapshot: sparse_candidate_identity_preserved=true, final_sparse_pct=8, final_anneal_pressure=light, protected_components_frozen=true, protected_component_modification_count=0, sparse_mask_frozen=true, sparse_mask_drift_rate=0.0018, bridge_baseline_preserved=true, trig_guardrails_preserved=true, trig_remains_repair_only=true, lane_a_D68_preservation_rate=1.0, post_repair_rust_path_invoked=true, post_repair_fallback_rows=0, and post_repair_failed_jobs=[].

Boundary reminder: D120 is an adapter-only controlled long-sequence halting repair prototype; it does not perform natural-language pretraining, introduce tokenizers or next-token objectives, use raw text or raw Raven, train a Gemma-class model, or prove AGI/production readiness.
