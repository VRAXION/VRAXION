# D116F Multi-Step Failure Attribution and Dataset Overlap Audit Contract

Purpose: audit the D116 multi-step instruction failure before D117 to determine whether the observed risk is caused by dataset/label ambiguity, evaluator artifacts, shortcut/overlap artifacts, split contamination, true halting/route accumulation, or mixed sources.

Boundary: D116F is diagnostic-only. It performs no multi-step training, no natural-language pretraining, no tokenizer or next-token objective work, no raw text corpus use, no raw Raven use, no Gemma-class training, no sparse-candidate mutation, no adapter mutation, and no protected-component mutation.

Mutation guard: the audit must report dataset_permanent_change_executed=false, sparse_candidate_identity_preserved=true, final_sparse_pct=8, final_anneal_pressure=light, adapter_modification_count=0, protected_component_modification_count=0, protected_components_frozen=true, and sparse_mask_frozen=true.

Upstream: D116 must replay or restore as d116_multi_step_instruction_bridge_plan_ready with d117_ready=true, primary_failure_mode=long_sequence_halting_accumulation, halting risk 0.056, shortcut risk 0.104, fallback_rows=0, and failed_jobs=[].

Audit scope: dataset/label ambiguity, metric/evaluator artifacts, command-template and grammar-rule overlap, shortcut baselines, order sensitivity, cleaned-subset true halting accumulation, split contamination, and per-subfamily attribution.

Positive gate: D116F passes only if all required audit reports are emitted, all audits execute, no training or forbidden boundary action occurs, D116 replay validates, scale is not reduced, fallback rows and failed jobs are zero, and a failure-source decision plus D117 go/no-go recommendation are produced.

Decision target: d116f_true_halting_accumulation_confirmed -> D117_MULTI_STEP_INSTRUCTION_BRIDGE_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS when dataset, metric, shortcut, and split artifacts are low and true halting/route accumulation evidence remains high on cleaned subsets.
