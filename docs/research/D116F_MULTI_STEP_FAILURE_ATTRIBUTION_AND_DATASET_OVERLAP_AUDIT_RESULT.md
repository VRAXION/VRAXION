# D116F Multi-Step Failure Attribution and Dataset Overlap Audit Result

Expected result: d116f_true_halting_accumulation_confirmed with next=D117_MULTI_STEP_INSTRUCTION_BRIDGE_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS when the D116 handoff is valid and audit gates pass.

Scale snapshot: requested_total_rows=210240, actual_total_rows=210240, scale_reduced=false, stress_mode_count=26, fallback_rows=0, failed_jobs=[].

Mutation guard snapshot: dataset_permanent_change_executed=false, sparse_candidate_identity_preserved=true, final_sparse_pct=8, final_anneal_pressure=light, adapter_modification_count=0, protected_component_modification_count=0, protected_components_frozen=true, and sparse_mask_frozen=true.

Attribution snapshot: label_ambiguity_rate=0.004, metric_artifact_likelihood_score=0.14, shortcut_artifact_likelihood_score=0.22, split_contamination_detected=false, and true_network_halting_evidence_score=0.72.

Audit conclusion: dataset, metric, shortcut, and split explanations are low; cleaned-subset failures retain long-sequence halting and route accumulation evidence, so D117 remains appropriate with sequence guardrails.

D117 recommendation: proceed to D117_MULTI_STEP_INSTRUCTION_BRIDGE_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS using the D116 limited scope and D116F audit guardrails.

Boundary reminder: D116F is diagnostic-only and performs no training, no natural-language pretraining, no tokenizer or next-token objective work, no raw text or raw Raven work, and no Gemma-class training.
