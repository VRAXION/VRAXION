# D102 Controlled Sparse Auto-Anneal Prototype Contract

## Purpose

D102 implements a controlled sparse auto-anneal prototype using the D101-approved safe schedule. It creates a sparse candidate copy of the D100/D101 recurrent routing microcircuit, preserves the dense baseline, locks D101 protected components, targets only D101 candidate redundant components, checkpoints every stage, and rolls back instead of overclaiming if any safety, loop, halting, convergence, top1, D68, truth/oracle, Rust/fallback, or reporting gate fails.

## Boundary

D102 is controlled symbolic ECF/IPF joint formula discovery only. The formula solver remains symbolic. D102 is a controlled sparse auto-anneal prototype, not production pruning. It creates a checkpointed sparse candidate copy only, preserves the dense baseline, and does not claim a full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, production readiness, or architecture superiority.

## Phase 0 upstream audit

The runner must verify branch/HEAD, check D101 commit `3fd46643bd52515f2f2d0bfa4ce77c6661b33993`, verify `target/pilot_wave/d101_auto_anneal_and_sparse_stabilization_prep/`, validate D101 decision `d101_auto_anneal_sparse_stabilization_prep_ready` with next `D102_CONTROLLED_SPARSE_AUTO_ANNEAL_PROTOTYPE`, confirm `d102_ready=true`, recommended safe sparsity `8`, pressure `light`, restore/rerun D101 if required, and write `d101_upstream_manifest.json`. D102 must not silently assume D101 was pushed.

## Scale and stress settings

Requested scale is seeds `23001,23002,23003,23004,23005,23006,23007,23008,23009,23010` with rows `train/test/ood=560` per seed/regime/split. Requested stress extension is seeds `23101,23102,23103,23104,23105,23106`, stress rows `720`, and stress modes covering D101 recurrent/sparsity tails plus sparse candidate, checkpoint, mask stability, recovery, protected-component integrity, and candidate-component pressure tails. Any scale reduction must be recorded and cannot overclaim.

## Controlled sparse schedule

D102 healthy fair candidate must use `pressure_floor=very_light`, `pressure_ceiling=light`, `max_target_sparsity_pct=8`, protected component locks, candidate component targeting, route/tail/halting/loop protected pressure, rollback, and checkpoint validation. The schedule is: stage0 dense replay, stage1 2% very-light pressure, stage2 4% very-light-to-light pressure, stage3 6% light pressure, and stage4 8% light pressure. The runner must stop/rollback if any fair stage fails hard gates and must never use reference-only 10%/moderate/aggressive/unprotected/protected-pruning arms to support a healthy decision.

## Required reports

Artifacts are written under `target/pilot_wave/d102_controlled_sparse_auto_anneal_prototype/` and must include `d101_upstream_manifest.json`, scale, dense baseline, protected lock, candidate targeting, checkpoint/rollback, stage0/stage1/stage2/stage3/stage4, stagewise regression, final sparse candidate, sparse safety/carryover/cost, reference-only, control, sentinel, split, overfit, truth/oracle, schema/crosscheck, deterministic replay reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

D102 passes only if D101 handoff/replay is valid, requested and actual scale are recorded with `scale_reduced=false`, all stress modes execute, dense baseline is preserved, sparse candidate is a copy, protected components remain locked with zero modifications, candidate redundant components are targeted, rollback/checkpoints are enabled, all fair stages through stage4 8% pass hard gates, final sparse candidate passes accuracy/tail/loop/halting/top1/D68/truth/oracle/Rust/support/cost gates, reference-only arms are not used to overclaim, leak/shortcut sentinels collapse, controls remain worse, deterministic replay/schema/metric crosscheck pass, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Passing controlled sparse prototype: `decision=d102_controlled_sparse_auto_anneal_prototype_confirmed`, `next=D103_SPARSE_RECURRENT_MICROCIRCUIT_SCALE_CONFIRM`.
- Partial 6% confirmation: `decision=d102_sparse_auto_anneal_partial_confirmed`, `next=D102A_SPARSE_TARGET_RECALIBRATION`.
- Stage failure with rollback: `decision=d102_sparse_stage_failure_rollback_succeeded`, `next=D102R_STAGE_FAILURE_REPAIR`.
- Rollback failure: `decision=d102_rollback_failure`, `next=D102B_CHECKPOINT_ROLLBACK_REPAIR`.
- Protected component violation: `decision=d102_protected_component_violation`, `next=D102P_PROTECTED_COMPONENT_LOCK_REPAIR`.
- Guard regression: `decision=d102_guard_regression_detected`, `next=D102G_GUARD_PRESERVATION_REPAIR`.
- Loop regression: `decision=d102_loop_utility_sparse_regression`, `next=D102L_LOOP_UTILITY_REPAIR`.
- Halting/convergence regression: `decision=d102_halting_convergence_sparse_regression`, `next=D102H_HALTING_CONVERGENCE_REPAIR`.
- Sparse tail risk: `decision=d102_sparse_tail_risk_detected`, `next=D102T_SPARSE_TAIL_RISK_REPAIR`.
- Truth/oracle contamination: `decision=d102_truth_leak_or_oracle_contamination_detected`, `next=D102L_TRUTH_LEAK_REPAIR`.
- Shortcut memorization: `decision=d102_shortcut_memorization_detected`, `next=D102S_SHORTCUT_MEMORIZATION_REPAIR`.
- Rust fallback: `decision=d102_rust_fallback_detected`, `next=D102R_RUST_PATH_REPAIR`.
- Reporting inconsistency: `decision=d102_invalid_metric_or_report_inconsistency`, `next=D102_REPORTING_REPAIR`.
- Incomplete run: `decision=d102_invalid_or_incomplete_run`, `next=D102_RETRY_WITH_FULL_AUDIT`.
