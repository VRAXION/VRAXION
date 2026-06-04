# D116 Multi-Step Instruction Bridge Plan With Sequence Guardrails Contract

Purpose: plan and dry-run a controlled symbolic multi-step instruction bridge after D115 scale-confirmed the symbolic-sequence bridge, with explicit diagnosis of long-sequence halting and shortcut risk.

Boundary: D116 is planning and non-destructive dry-run only. It does not perform full multi-step training, natural-language pretraining, Gemma-class training, tokenizer work, next-token prediction, raw text corpus use, raw Raven use, protected component mutation, or sparse-mask mutation.

Upstream: D115 must replay or restore as d115_symbolic_sequence_bridge_scale_confirmed with d116_ready=true, bridge gates passed, MULTI_STEP_INSTRUCTION_ROUTING_FAMILY reference-only, trig repair-only, no fallback rows, and no failed jobs.

Scope: decompose MULTI_STEP_INSTRUCTION_ROUTING_FAMILY into two-step, three-step, four-step, nested, conditional, variable-binding, long-sequence halting, and adversarial template-overlap symbolic subfamilies.

Required planning outputs: failure decomposition, subfamily readiness map, long-sequence halting breakdown, shortcut breakdown, variable-binding/nested/conditional risk reports, bridge and lane preservation policies, trig guardrail policy, D117 objective, batch mix, curriculum, stop/rollback, eval harness, metric gates, and contract recommendation.

Positive gate: D116 is D117-ready only if upstream, scale, boundary, sparse/freeze identity, planning, diagnostics, dry-run, leak/shortcut sentinels, Rust path, and report consistency gates pass.

Decision target: d116_multi_step_instruction_bridge_plan_ready -> D117_MULTI_STEP_INSTRUCTION_BRIDGE_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS when limited safe multi-step subfamilies are identified and all guardrails hold.
