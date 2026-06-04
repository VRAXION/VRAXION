# D117 Multi-Step Combined Halting Route Repair Prototype With Sequence Guardrails Contract

Purpose: run the first adapter-only controlled symbolic multi-step instruction repair prototype for the D116G-confirmed mixed halting-route mechanism.

Boundary: D117 is a repair prototype only. It may update only halting_head_adapter_delta, route_head_adapter_delta, and calibration_scalar_adapter_delta; it performs no full-core training, no natural-language pretraining, no tokenizer or next-token objective work, no raw text corpus use, no raw Raven use, no Gemma-class training, and no protected-component, solver, dense-baseline, or sparse-mask mutation.

Upstream: D116G must replay or restore as d116g_mixed_halting_route_mechanism_confirmed with next=D117_MULTI_STEP_COMBINED_HALTING_ROUTE_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS, dominant_mechanism=mixed_halting_route_mechanism, mechanism_confidence=0.78, recommended_d117_objective_name=multi_step_combined_halting_route_repair_with_sequence_guardrails, fallback_rows=0, and failed_jobs=[].

Trainable surfaces: halting_head_adapter_delta, route_head_adapter_delta, and calibration_scalar_adapter_delta. Frozen surfaces include the symbolic formula solver, dense baseline, protected symbolic router, protected components, 8% sparse mask, base recurrent hidden-state path, base halting head, base route logits head, recurrent_state_adapter except reference-only ablation, and all sparse/protected base paths.

Subfamily scope: TWO_STEP_INSTRUCTION_ROUTING_FAMILY and THREE_STEP_INSTRUCTION_ROUTING_FAMILY are trainable guarded; FOUR_STEP_INSTRUCTION_ROUTING_FAMILY, VARIABLE_BINDING_MULTI_STEP_FAMILY, and CONDITIONAL_BRANCH_INSTRUCTION_FAMILY are guarded low-weight probes; NESTED_INSTRUCTION_ROUTING_FAMILY, LONG_SEQUENCE_HALTING_STRESS_FAMILY, and ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY remain reference-only and outside the healthy claim.

Positive gate: D117 passes only if D116G replay validates, requested scale is not reduced, repair training executes on exactly the approved adapters, repair signals improve above thresholds, checkpoints pass without rollback, sparse/protected identities remain frozen, bridge/Lane A/B/D/trig preservation gates pass, leak/shortcut sentinels remain clean, Rust path is invoked, fallback rows are zero, and failed_jobs=[].

Decision target: d117_multi_step_combined_halting_route_repair_prototype_confirmed -> D118_MULTI_STEP_COMBINED_HALTING_ROUTE_REPAIR_SCALE_CONFIRM_WITH_SEQUENCE_GUARDRAILS when combined repair succeeds and all gates pass.
