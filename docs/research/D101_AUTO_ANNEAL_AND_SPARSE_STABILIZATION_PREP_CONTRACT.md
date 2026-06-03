# D101 Auto Anneal and Sparse Stabilization Prep Contract

## Purpose

D101 prepares safe auto-annealing and sparse stabilization for the D100 scale-confirmed recurrent routing microcircuit. It is not destructive pruning, irreversible weight deletion, or a broad architecture proof. It must produce a safe sparsity/anneal pressure map, identify critical protected circuit regions and candidate redundant regions, run non-destructive shadow-mask and anneal-pressure probes, audit loop/halting/convergence fragility, and recommend D102 only when safety gates support a controlled sparse auto-anneal prototype.

## Boundary

D101 is controlled symbolic ECF/IPF joint formula discovery only. The formula solver remains symbolic. D101 uses non-destructive shadow masks and pressure probes only; it performs no irreversible pruning or permanent sparse rewrite. It does not claim a full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, production readiness, or architecture superiority.

## Phase 0 upstream audit

The runner must verify branch/HEAD, check D100 commit `590995acdb9976812599b0fe2aa0f1fa7e7af1ea`, verify `target/pilot_wave/d100_recurrent_routing_microcircuit_scale_confirm/`, validate D100 decision `d100_recurrent_routing_microcircuit_scale_confirmed` with next `D101_AUTO_ANNEAL_AND_SPARSE_STABILIZATION_PREP`, restore/rerun D100 if required, and write `d100_upstream_manifest.json`. D101 must not silently assume D100 was pushed.

## Scale and stress settings

Requested scale is seeds `22001,22002,22003,22004,22005,22006,22007,22008,22009,22010` with rows `train/test/ood=560` per seed/regime/split. Requested stress extension is seeds `22101,22102,22103,22104,22105,22106`, stress rows `720`, and stress modes covering D100 recurrent tails plus shadow sparsity, saliency instability, protected component ablation, and redundant component mask tails. Any scale reduction must be recorded and cannot overclaim.

## Feature, target, and sparse-prep constraints

Fair arms may use only D97/D98/D99/D100-approved inference-time non-truth symbolic/proxy features. Fair arms must not use truth labels, oracle route labels, support-regime labels, row IDs, seed IDs, Python hashes, file order, artifact indexes, filenames, object IDs, `repr(row)`, generated answer labels, post-hoc correctness labels, hidden-state initialization derived from forbidden fields, halt-step or step-count shortcuts, mask-id shortcuts, sparsity-pattern shortcuts, or synthetic keys identifying row, regime, seed, route, answer, or family label. The teacher remains the validated symbolic router decision generated from inference-available signals only.

## Required reports

Artifacts are written under `target/pilot_wave/d101_auto_anneal_and_sparse_stabilization_prep/` and must include `d100_upstream_manifest.json`, scale, component schema, saliency/importance, ablation, shadow-mask, pressure-probe, recommended D102 schedule, targeted carryover, recurrent safety, sentinel, Rust, schema/crosscheck, deterministic replay reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

D101 passes only if D100 handoff/replay is valid, requested and actual scale are recorded with `scale_reduced=false`, all stress modes execute, baseline D100 recurrent behavior is preserved, saliency ranks are stable, protected and redundant components are identifiable, eligible shadow masks are non-destructive and safe, pressure probes support a conservative D102 schedule, top1/D68/truth/oracle/Rust/fallback invariants hold, leak/shortcut sentinels collapse, controls remain worse, deterministic replay/schema/metric crosscheck pass, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Passing sparse-stabilization preparation: `decision=d101_auto_anneal_sparse_stabilization_prep_ready`, `next=D102_CONTROLLED_SPARSE_AUTO_ANNEAL_PROTOTYPE`.
- D100 replay regression: `decision=d101_d100_preservation_regression_detected`, `next=D101U_UPSTREAM_PRESERVATION_REPAIR`.
- Saliency instability: `decision=d101_saliency_instability_detected`, `next=D101S_SALIENCY_STABILITY_REPAIR`.
- Protected map failure: `decision=d101_protected_component_map_not_confirmed`, `next=D101P_PROTECTED_COMPONENT_AUDIT_REPAIR`.
- No safe redundancy: `decision=d101_no_safe_redundancy_found`, `next=D101R_REDUNDANCY_DISCOVERY_REPAIR`.
- Shadow sparsity safety failure: `decision=d101_shadow_sparsity_safety_failure`, `next=D101M_MASK_SAFETY_REPAIR`.
- Loop utility sparsity failure: `decision=d101_loop_utility_sparsity_failure`, `next=D101L_LOOP_UTILITY_STABILIZATION`.
- Halting/convergence sparsity failure: `decision=d101_halting_or_convergence_sparsity_failure`, `next=D101H_HALTING_CONVERGENCE_REPAIR`.
- Guard regression: `decision=d101_guard_regression_detected`, `next=D101G_GUARD_PRESERVATION_REPAIR`.
- Truth/oracle contamination: `decision=d101_truth_leak_or_oracle_contamination_detected`, `next=D101L_TRUTH_LEAK_REPAIR`.
- Shortcut memorization: `decision=d101_shortcut_memorization_detected`, `next=D101H_SHORTCUT_MEMORIZATION_REPAIR`.
- Rust fallback: `decision=d101_rust_fallback_detected`, `next=D101R_RUST_PATH_REPAIR`.
- Reporting inconsistency: `decision=d101_invalid_metric_or_report_inconsistency`, `next=D101_REPORTING_REPAIR`.
- Incomplete run: `decision=d101_invalid_or_incomplete_run`, `next=D101_RETRY_WITH_FULL_AUDIT`.
