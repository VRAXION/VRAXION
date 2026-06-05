# E5_SUBSTRATE_NECESSITY_TEST Contract

## Purpose

E5 tests whether the E4 decision-relevant abstraction-routing result requires a neural substrate, or whether non-neural mutation/rollback routing remains sufficient on the controlled symbolic proxy.

This is not a natural-language reasoning, AGI, consciousness, or model-scale claim.

## Systems

- `e4_top_down_hierarchical_router`: non-neural E4 reference, mutation + rollback.
- `tiny_mlp_gradient`: tiny PyTorch MLP, gradient-trained.
- `tiny_mlp_mutation_only`: same MLP-style architecture in NumPy, mutation + rollback only.
- `tiny_recurrent_gradient`: tiny GRU-like PyTorch recurrent model, gradient-trained.
- `tiny_recurrent_mutation_only`: same recurrent-style architecture in NumPy, mutation + rollback only.
- `hybrid_neural_frontend_mutation_router`: gradient-trained neural feature frontend plus frozen non-neural mutation router head.
- Controls: `flat_detail_scanner`, `bottom_up_evidence_scanner`, `random_classifier`, `oracle_reference_only`.

## Required Artifacts

- `e5_backend_manifest.json`
- `e5_task_generation_report.json`
- `e5_substrate_comparison_report.json`
- `e5_leakage_and_memorization_report.json`
- `e5_training_cost_report.json`
- `e5_no_synthetic_metric_audit.json`
- `e5_deterministic_replay_report.json`
- `e5_accept_reject_rollback_report.json`
- `e5_generation_metrics.json`
- `e5_row_level_eval_sample_heldout.json`
- `e5_row_level_eval_sample_ood.json`
- `e5_row_level_eval_sample_counterfactual.json`
- `e5_row_level_eval_sample_adversarial.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`
- `progress.jsonl`

Per system:

- `e5_candidate_<system>_summary.json`
- `e5_parameter_diff_<system>.json`
- mutation systems: `e5_mutation_history_<system>.json`
- gradient systems: `e5_training_history_<system>.json`

## Gates

The checker fails on missing artifacts, missing systems, missing row-level eval, replay mismatch, rollback mismatch, missing parameter diff, leakage-control failure without leak decision, mutation-only optimizer/backprop use, or missing progress writeouts.

Branch-order remap is a required leakage/artifact control. Any system that passes the normal abstraction-routing threshold must preserve usefulness under branch-order shuffle after target remap. If it does not, the run must either set `decision = e5_leak_or_artifact_detected` or fail the checker.

The runner writes append-only progress throughout primary and replay execution. Parallel execution uses locked `progress.jsonl` appends so multiple workers cannot corrupt partial writeouts.

## Decision Boundary

E5 answers only this substrate question for the current symbolic abstraction-routing proxy. It does not establish that the same behavior transfers to open-ended language or larger models.
