# D120 Long-Sequence Halting Repair Prototype with Sequence Guardrails Contract

Purpose: execute the first adapter-only long-sequence halting-margin floor repair prototype after D119 mapped the residual long-sequence frontier.

Boundary: D120 is controlled symbolic long-sequence halting repair only. It preserves the symbolic formula solver, dense baseline, protected symbolic router, protected components, base recurrent path, and D102 8% light sparse mask. It performs no natural-language pretraining, introduces no tokenizer or next-token objective, uses no raw text corpus or raw Raven tasks, and does not train a Gemma-class model.

Upstream: D119 must replay or restore as d119_residual_long_sequence_halting_frontier_mapped with d120_ready=true, residual_failure_rate=0.032, dominant_residual_cluster=long_sequence_step5_halting_margin_floor, dominant_first_bad_step=5, route_flip_step=5, stop_continue_boundary_flip_step=5, step-5 halting margin floor=0.027, long-sequence frontier recommended as guarded low-weight candidate, nested/adversarial reference-only, training_updates_executed=false, adapter_modification_count=0, fallback_rows=0, and failed_jobs=[].

Trainable adapter surfaces: halting_head_adapter_delta, route_head_adapter_delta, calibration_scalar_adapter_delta. recurrent_state_adapter remains frozen except reference-only arms.

Repair objective: long_sequence_halting_margin_floor_repair_with_sequence_guardrails using long_sequence_halting_margin_floor_loss, route_uncertainty_tail_loss, step5_step6_margin_floor_loss, calibration_tail_stability_loss, residual_frontier_guard_loss, stop_continue_boundary_floor_loss, overconfidence_prevention_loss, failure_cliff_shift_penalty, preservation losses, sparse/protected penalties, and shortcut guards.

Subfamily policy: TWO_STEP_INSTRUCTION_ROUTING_FAMILY and THREE_STEP_INSTRUCTION_ROUTING_FAMILY remain stable trainable preservation baselines; FOUR_STEP_INSTRUCTION_ROUTING_FAMILY, VARIABLE_BINDING_MULTI_STEP_FAMILY, CONDITIONAL_BRANCH_INSTRUCTION_FAMILY, and LONG_SEQUENCE_HALTING_STRESS_FAMILY run as guarded low-weight candidates/probes; NESTED_INSTRUCTION_ROUTING_FAMILY and ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY remain reference-only and are not promoted into healthy claims.

Positive gate: D120 passes only if D119 handoff validates, scale is not reduced, repair training executes with the three allowed adapters only, sparse/protected identity is preserved, long-sequence failure and halting-floor metrics improve without overconfidence or step6/step7 cliff regression, long-sequence remains guarded low-weight, stable/guarded/reference family policies hold, bridge/Lane/trig/Rust/D68 preservation holds, leak/shortcut sentinels are clean, fallback_rows=0, failed_jobs=[], and report/metric crosschecks pass.

Decision target: d120_long_sequence_halting_repair_prototype_confirmed -> D121_LONG_SEQUENCE_HALTING_REPAIR_SCALE_CONFIRM_WITH_SEQUENCE_GUARDRAILS when all gates pass.
