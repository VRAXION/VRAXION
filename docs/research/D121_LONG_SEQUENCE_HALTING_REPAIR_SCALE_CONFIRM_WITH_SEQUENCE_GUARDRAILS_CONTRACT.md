# D121 Long-Sequence Halting Repair Scale Confirm with Sequence Guardrails Contract

Purpose: scale-confirm the D120 adapter-only long-sequence halting repair prototype and decide whether the long-sequence guarded repair is stable enough to hand off to D122 nested/adversarial residual-frontier planning.

Boundary: D121 is controlled symbolic long-sequence halting repair scale confirmation only. It preserves the symbolic formula solver, dense baseline, protected symbolic router, protected components, recurrent base paths, and D102 8% light sparse mask. It performs no natural-language pretraining, introduces no tokenizer or next-token objective, uses no raw text corpus or raw Raven tasks, and does not train a Gemma-class model.

Upstream: D120 must replay or restore as d120_long_sequence_halting_repair_prototype_confirmed with d121_ready=true, long_sequence_failure_reduction=0.174, no failure-cliff shift, residual_failure_reduction=0.156, long-sequence guarded_low_weight status, nested/adversarial reference-only status, sparse/protected preservation, Rust invocation, fallback_rows=0, and failed_jobs=[].

Trainable adapter surfaces: halting_head_adapter_delta, route_head_adapter_delta, calibration_scalar_adapter_delta. recurrent_state_adapter remains frozen except reference-only arms.

Scale objective: long_sequence_halting_margin_floor_repair_scale_confirm_with_sequence_guardrails using the D120 long-sequence halting floor, route-tail, calibration-tail, residual-frontier, overconfidence, failure-cliff, preservation, sparse/protected, and shortcut losses.

Subfamily policy: LONG_SEQUENCE_HALTING_STRESS_FAMILY remains guarded_low_weight and is not promoted into a healthy claim; TWO_STEP_INSTRUCTION_ROUTING_FAMILY and THREE_STEP_INSTRUCTION_ROUTING_FAMILY remain stable trainable preservation baselines; FOUR_STEP_INSTRUCTION_ROUTING_FAMILY, VARIABLE_BINDING_MULTI_STEP_FAMILY, and CONDITIONAL_BRANCH_INSTRUCTION_FAMILY remain guarded low-weight; NESTED_INSTRUCTION_ROUTING_FAMILY and ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY remain reference-only.

Positive gate: D121 passes only if D120 handoff validates, scale is not reduced, repair scale training executes with only the three allowed adapter deltas, sparse/protected identity is preserved, long-sequence failure and step5/step6/step7+ halting floors remain improved at scale, overconfidence and step6/step7 cliff shift do not regress, family policies hold, bridge/Lane/trig/Rust/D68 preservation holds, leak/shortcut sentinels are clean, fallback_rows=0, failed_jobs=[], and report/metric crosschecks pass.

Decision target: d121_long_sequence_halting_repair_scale_confirmed -> D122_NESTED_AND_ADVERSARIAL_RESIDUAL_FRONTIER_PLAN when all gates pass.
