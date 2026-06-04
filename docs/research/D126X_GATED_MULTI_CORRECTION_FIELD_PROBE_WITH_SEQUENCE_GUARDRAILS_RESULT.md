# D126X Gated Multi-Correction Field Probe with Sequence Guardrails Result

Expected result: decision=d126x_gated_multi_correction_probe_positive, next=D126_ADVERSARIAL_TEMPLATE_OVERLAP_REPAIR_PROTOTYPE_WITH_GATED_MULTI_CORRECTION_BRANCH, d126_ready=true, main_d126_replaced=false.

Scale snapshot: requested_total_rows=177660, actual_total_rows=177660, scale_reduced=false, stress_mode_count=34, fallback_rows=0, failed_jobs=[].

Boundary snapshot: gated_multi_correction_probe_executed=true, training_updates_executed=false, adapter_modification_count=0, dataset_permanent_change_executed=false, natural_language_pretraining_executed=false, tokenizer_introduced=false, next_token_objective_defined=false, raw_text_corpus_used=false, gemma_class_training_executed=false.

Component snapshot: template_collision_correction_norm=0.44, grammar_collision_correction_norm=0.40, true_route_uncertainty_correction_norm=0.53, shortcut_guard_correction_norm=0.31, calibration_correction_norm=0.27, preservation_correction_norm=0.22, template_vs_true_route_alignment=-0.18, grammar_vs_true_route_alignment=-0.14, surface_vs_true_route_alignment=-0.21, shortcut_guard_vs_surface_alignment=-0.33, calibration_vs_true_route_alignment=0.46, preservation_vs_repair_alignment=0.18, component_conflict_score=0.41, premature_correction_collapse_score=0.37.

Comparison snapshot: weighted_sum_route_margin_improvement=0.009, gated_route_margin_improvement=0.016, weighted_sum_shortcut_reliance_delta=0.002, gated_shortcut_reliance_delta=-0.003, weighted_sum_collision_failure_reduction=0.073, gated_collision_failure_reduction=0.124, weighted_sum_preservation_risk=0.036, gated_preservation_risk=0.034, gated_vs_weighted_margin_delta=0.007, gated_vs_weighted_shortcut_delta=-0.005, gated_vs_weighted_preservation_delta=-0.002, gated_probe_positive=true, recommend_gated_branch_for_D126=true.

Gate snapshot: route_priority_gate_margin_improvement=0.017, shortcut_suppression_gate_shortcut_delta=-0.004, calibration_gated_margin_improvement=0.014, preservation_gated_preservation_risk=0.032, random_gate_control_margin_improvement=0.002, surface_only_gate_shortcut_delta=0.009, template_only_gate_shortcut_delta=0.008, grammar_only_gate_shortcut_delta=0.007.

Preservation snapshot: nested_guarded_low_weight_preserved=true, long_sequence_guarded_low_weight_preserved=true, bridge_baseline_preserved=true, trig_guardrails_preserved=true, lane_a_D68_preservation_rate=1.0, sparse_candidate_identity_preserved=true, final_sparse_pct=8, final_anneal_pressure=light, sparse_mask_drift_rate=0.0019, rust_path_invoked=true.

D126 recommendation snapshot: keep main D126 plan but add gated multi-correction branch as a guarded reference/fair branch; do not replace main D126, do not promote healthy claims, and preserve the standard sequence-guardrail rollback path.

Boundary reminder: D126X is diagnostic/reference-only sidequest evidence, not a production repair result and not AGI or production-readiness evidence.
