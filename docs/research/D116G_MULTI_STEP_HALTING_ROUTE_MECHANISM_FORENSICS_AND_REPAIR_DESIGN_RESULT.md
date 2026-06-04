# D116G Multi-Step Halting Route Mechanism Forensics and Repair Design Result

Expected result: d116g_mixed_halting_route_mechanism_confirmed with next=D117_MULTI_STEP_COMBINED_HALTING_ROUTE_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS when D116F handoff is valid and D116G mechanism-forensics gates pass.

Scale snapshot: requested_total_rows=195120, actual_total_rows=195120, scale_reduced=false, stress_mode_count=27, fallback_rows=0, failed_jobs=[].

Boundary snapshot: mechanism_forensics_executed=true, training_updates_executed=false, adapter_modification_count=0, dataset_permanent_change_executed=false, natural_language_pretraining_executed=false, tokenizer_introduced=false, next_token_objective_defined=false, raw_text_corpus_used=false, raw_raven_used=false, and gemma_class_training_executed=false.

Mechanism snapshot: dominant_mechanism=mixed_halting_route_mechanism, mechanism_confidence=0.78, halting_margin_decay_score=0.73, route_uncertainty_accumulation_score=0.69, top1_top2_margin_collapse_score=0.61, calibration_margin_decay_score=0.58, recurrent_state_drift_score=0.46, variable_binding_drift_score=0.42, and shortcut_escape_under_uncertainty_score=0.31.

Repair-design snapshot: recommend multi_step_combined_halting_route_repair_with_sequence_guardrails using adapter-only halting-head, route-head, and calibration-scalar deltas while keeping the symbolic solver, dense baseline, protected components, and 8% sparse mask frozen.

D117 recommendation: proceed to D117_MULTI_STEP_COMBINED_HALTING_ROUTE_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS with trainable two-step/three-step families, guarded low-weight four-step/variable-binding/conditional families, and reference-only nested/long/adversarial families.

Boundary reminder: D116G is mechanism-forensics and repair-design only; it performs no training, no adapter mutation, no dataset mutation, no natural-language pretraining, no tokenizer or next-token objective work, no raw text or raw Raven work, and no Gemma-class training.
