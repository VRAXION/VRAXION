# D103 Sparse Recurrent Microcircuit Scale Confirm Contract

## Purpose

D103 scale-confirms the D102 8% light-pressure protected sparse recurrent routing microcircuit. It does not search for higher sparsity, does not switch to moderate/aggressive pressure, and does not mutate protected components. The run asks whether `D102_SPARSE_AUTO_ANNEAL_8PCT_LIGHT_PROTECTED_FAIR` remains stable under larger scale, more seeds, sparse-specific stress, checkpoint replay, mask replay/stability, protected-component integrity checks, and shortcut/leak sentinels.

## Boundary

D103 is controlled symbolic ECF/IPF joint formula discovery only. The formula solver remains symbolic. D103 is sparse recurrent microcircuit scale-confirmation only; it does not increase sparsity, does not perform production pruning, and does not claim a full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, production readiness, or architecture superiority.

## Phase 0 upstream audit

The runner must verify branch/HEAD, check D102 commit `1765ae8562fb1b806f02264c1ea6451ab97247d0`, verify `target/pilot_wave/d102_controlled_sparse_auto_anneal_prototype/`, validate D102 decision `d102_controlled_sparse_auto_anneal_prototype_confirmed` with next `D103_SPARSE_RECURRENT_MICROCIRCUIT_SCALE_CONFIRM`, confirm `d103_ready=true`, final sparse candidate `D102_SPARSE_AUTO_ANNEAL_8PCT_LIGHT_PROTECTED_FAIR`, final sparse percent `8`, final pressure `light`, restore/rerun D102 if required, and write `d102_upstream_manifest.json`. D103 must not silently assume D102 was pushed.

## Scale and stress settings

Requested scale is seeds `24001,24002,24003,24004,24005,24006,24007,24008,24009,24010,24011,24012` with rows `train/test/ood=640` per seed/regime/split. Requested stress extension is seeds `24101,24102,24103,24104,24105,24106,24107,24108`, stress rows `820`, and stress modes covering recurrent tails plus sparse candidate, checkpoint, deterministic mask replay, mask permutation/jitter, protected-component integrity, sparse Rust invocation, sparse cost frontier, and sparse tail min-seed replay. Any scale reduction must be recorded and cannot overclaim.

## Sparse candidate identity

The primary fair arm is `D103_SPARSE_RECURRENT_8PCT_LIGHT_PROTECTED_SCALE`, the scale replay of D102's `D102_SPARSE_AUTO_ANNEAL_8PCT_LIGHT_PROTECTED_FAIR`. The final sparse target must remain `8%`, final pressure must remain `light`, protected components must stay locked with zero modifications, candidate components must remain the only targeted components, and checkpoint/mask replay must pass with match rates at least `0.999`.

## Required reports

Artifacts are written under `target/pilot_wave/d103_sparse_recurrent_microcircuit_scale_confirm/` and must include `d102_upstream_manifest.json`, scale, dense baseline preservation, sparse candidate scale, checkpoint replay, mask replay/stability/permutation/jitter, recovery, protected/candidate component, sparse route/loop/convergence/halting/state, tail/carryover, guard, D68, oracle/support/cost, sparse Rust, reference-only, control, sentinel, split, overfit, truth/oracle, schema/crosscheck, deterministic replay reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

D103 passes only if D102 handoff/replay is valid, requested and actual scale are recorded with `scale_reduced=false`, all stress modes execute, failed jobs are empty, sparse candidate identity remains 8% light protected, protected components remain unmodified, candidate components are targeted, checkpoint/mask replay and stability gates pass, sparse scale accuracy/tail/loop/halting/top1/D68/truth/oracle/Rust/support/cost gates pass, reference-only arms are not used to overclaim, leak/shortcut sentinels collapse, controls remain worse, deterministic replay/schema/metric crosscheck pass, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Passing sparse scale confirmation: `decision=d103_sparse_recurrent_microcircuit_scale_confirmed`, `next=D104_SPARSE_RECURRENT_GENERALIZATION_AND_COMPRESSION_FRONTIER_MAP`.
- Sparse tail risk: `decision=d103_sparse_tail_risk_detected`, `next=D103T_SPARSE_TAIL_RISK_REPAIR`.
- Sparse loop utility scale failure: `decision=d103_sparse_loop_utility_scale_failure`, `next=D103L_LOOP_UTILITY_REPAIR`.
- Sparse halting/convergence failure: `decision=d103_sparse_halting_convergence_failure`, `next=D103H_HALTING_CONVERGENCE_REPAIR`.
- Sparse mask or checkpoint replay failure: `decision=d103_sparse_mask_or_checkpoint_replay_failure`, `next=D103M_MASK_CHECKPOINT_REPAIR`.
- Protected component violation: `decision=d103_protected_component_violation`, `next=D103P_PROTECTED_COMPONENT_LOCK_REPAIR`.
- Guard regression: `decision=d103_guard_regression_detected`, `next=D103G_GUARD_PRESERVATION_REPAIR`.
- Truth/oracle contamination: `decision=d103_truth_leak_or_oracle_contamination_detected`, `next=D103L_TRUTH_LEAK_REPAIR`.
- Shortcut memorization: `decision=d103_shortcut_memorization_detected`, `next=D103S_SHORTCUT_MEMORIZATION_REPAIR`.
- Rust fallback: `decision=d103_rust_fallback_detected`, `next=D103R_RUST_PATH_REPAIR`.
- Reporting inconsistency: `decision=d103_invalid_metric_or_report_inconsistency`, `next=D103_REPORTING_REPAIR`.
- Incomplete run: `decision=d103_invalid_or_incomplete_run`, `next=D103_RETRY_WITH_FULL_AUDIT`.
