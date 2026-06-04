# D124 Nested Instruction Repair Scale Confirm with Sequence Guardrails Contract

Purpose: scale-confirm the D123 adapter-only nested-instruction route-stack/binding-scope repair prototype and decide whether the nested guarded repair is stable enough to hand off to D125 adversarial-template deep forensics and repair planning.

Boundary: D124 is controlled symbolic nested-instruction repair scale confirmation only. It preserves the symbolic formula solver, dense baseline, protected symbolic router, protected components, recurrent base paths, and D102 8% light sparse mask. It performs no natural-language pretraining, introduces no tokenizer or next-token objective, uses no raw text corpus or raw Raven tasks, and does not train a Gemma-class model.

Upstream: D123 must replay or restore as d123_nested_instruction_repair_prototype_confirmed with d124_ready=true, repair_training_executed=true, nested_failure_reduction=0.146, nested_route_stack_failure_reduction=0.132, nested_binding_scope_drift_reduction=0.161, depth4_cliff_detected=false, sparse/protected preservation, Rust invocation, fallback_rows=0, and failed_jobs=[].

Trainable adapter surfaces: halting_head_adapter_delta, route_head_adapter_delta, and calibration_scalar_adapter_delta only. recurrent_state_adapter remains frozen except reference-only ablations.

Scale objective: nested_instruction_route_stack_repair_scale_confirm_with_sequence_guardrails using route-stack, binding-scope, nested halting-margin, route-uncertainty, evaluator-edge, depth4-cliff, adversarial-reference, long-sequence preservation, bridge/trig/Lane preservation, sparse/protected, and shortcut losses.

Subfamily policy: NESTED_DEPTH_2_INSTRUCTION_FAMILY, NESTED_DEPTH_3_INSTRUCTION_FAMILY, NESTED_ROUTE_STACK_FAMILY, and NESTED_SCOPE_RESOLUTION_FAMILY remain guarded_low_weight and excluded from full healthy claims; NESTED_DEPTH_4_PLUS_INSTRUCTION_FAMILY, NESTED_CONDITIONAL_BINDING_FAMILY, NESTED_STOP_CONTINUE_BOUNDARY_FAMILY, and all adversarial-template families remain reference-only and excluded from healthy claims.

Positive gate: D124 passes only if D123 handoff validates, scale is not reduced, repair scale training executes with only allowed adapter deltas, sparse/protected identity is preserved, nested failure and route-stack/binding/scope/halting/uncertainty improvements survive scale, depth-4+ does not worsen, family policies hold, long-sequence/two-three/four-variable-conditional/bridge/Lane/trig/Rust/D68 preservation holds, leak/shortcut sentinels are clean, fallback_rows=0, failed_jobs=[], and report/metric crosschecks pass.

Decision target: d124_nested_instruction_repair_scale_confirmed -> D125_ADVERSARIAL_TEMPLATE_OVERLAP_DEEP_FORENSICS_AND_REPAIR_PLAN_WITH_SEQUENCE_GUARDRAILS when all gates pass.
