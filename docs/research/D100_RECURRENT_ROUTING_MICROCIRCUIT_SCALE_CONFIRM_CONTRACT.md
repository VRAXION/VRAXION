# D100 Recurrent Routing Microcircuit Scale Confirm Contract

## Purpose

D100 scale-confirms the D99 recurrent routing microcircuit, `D99_RECURRENT_HALTING_CONFIDENCE_FAIR`, under larger scale, more seeds, stronger recurrent-state stress, longer tail mixes, halting pressure, state noise/reset/drift/saturation/oscillation attacks, and leak/shortcut sentinels. D100 must verify recurrent accuracy, non-trivial loop utility, hidden-state usefulness, safe halting, stable convergence, OOD/stress/tail carryover, top1/D68/safety/truth/oracle/Rust/fallback invariants, and no row/hash/file/seed/hidden-state/halt-step/step-count shortcut.

## Boundary

D100 is controlled symbolic ECF/IPF joint formula discovery only. The formula solver remains symbolic. D100 does not claim a full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, production readiness, or architecture superiority.

## Phase 0 upstream audit

The runner must verify branch/HEAD, check D99 commit `bcc15b5599d1e48f83f68d4939043ec2e13e5c82`, verify `target/pilot_wave/d99_recurrent_routing_microcircuit_prototype/`, validate D99 decision `d99_recurrent_routing_microcircuit_prototype_confirmed` with next `D100_RECURRENT_ROUTING_MICROCIRCUIT_SCALE_CONFIRM`, restore/rerun D99 if required, and write `d99_upstream_manifest.json`. D100 must not silently assume D99 was pushed.

## D99 handoff

D99 confirmed `D99_RECURRENT_HALTING_CONFIDENCE_FAIR` with test accuracy `0.9946`, OOD accuracy `0.9923`, stress accuracy `0.9915`, min-seed accuracy `0.9910`, worst-seed accuracy `0.9900`, convergence rate `0.9988`, non-convergence rate `0.0005`, oscillation rate `0.0004`, loop usefulness `0.74`, tail loop usefulness `0.73`, low-cost/OOD/top1 tail score `0.746`, min-seed tail score `0.742`, combined OOD/joint-boundary breakpoint `0.756`, top1 guard preserved, D68 preservation `1.0`, truth/oracle/Rust gates passing, `fallback_rows=0`, and `failed_jobs=[]`.

## Scale and stress settings

Requested scale is seeds `21001,21002,21003,21004,21005,21006,21007,21008,21009,21010,21011,21012` with rows `train/test/ood=640` per seed/regime/split. Requested stress extension is seeds `21101,21102,21103,21104,21105,21106,21107,21108`, stress rows `train/test/ood=820`, and stress modes covering combined low-cost/OOD/top1, OOD shift, joint boundary, feature noise/dropout, calibration, mixed tail, worst-seed replay, recurrent state noise/reset/drift/saturation/oscillation, halting pressure, delayed/early halt, step budget, and hidden-state shuffle tails. Any scale reduction must be recorded and cannot overclaim.

## Feature, target, and recurrent constraints

Fair arms may use only D97/D98/D99-approved inference-time non-truth symbolic/proxy features. Fair arms must not use truth labels, oracle route labels, support-regime labels, row IDs, seed IDs, Python hashes, file order, artifact indexes, filenames, object IDs, `repr(row)`, generated answer labels, post-hoc correctness labels, hidden-state initialization derived from forbidden fields, halt steps derived from forbidden fields, or synthetic shortcut keys identifying row, regime, seed, or route. The target remains the validated symbolic router decision generated from inference-available signals only.

## Required reports

Artifacts are written under `target/pilot_wave/d100_recurrent_routing_microcircuit_scale_confirm/` and must include `d99_upstream_manifest.json`, all recurrent scale/eval/loop/convergence/halting/state/usefulness/carryover/guard/safety/sentinel/provenance reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

D100 passes only if D99 handoff/replay is valid, requested and actual scale are recorded with `scale_reduced=false`, all stress modes execute, the best fair recurrent arm meets scale accuracy/min-seed/worst-seed/overfit/safety/top1/D68/truth/oracle/support/step gates, recurrent convergence and loop-utility gates survive scale, targeted low-cost/OOD/top1 tail and combined OOD/joint carryover are preserved, state robustness gates pass, leak/shortcut sentinels collapse, no forbidden features or split contamination appear, controls remain worse, deterministic replay/schema/metric crosscheck pass, Rust path is invoked, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Passing recurrent scale confirmation: `decision=d100_recurrent_routing_microcircuit_scale_confirmed`, `next=D101_AUTO_ANNEAL_AND_SPARSE_STABILIZATION_PREP`.
- Loop utility scale failure: `decision=d100_recurrent_loop_utility_scale_failure`, `next=D100L_LOOP_UTILITY_REPAIR`.
- Tail/worst-seed failure: `decision=d100_recurrent_tail_risk_detected`, `next=D100T_RECURRENT_TAIL_RISK_REPAIR`.
- State stability failure: `decision=d100_recurrent_state_stability_failure`, `next=D100S_RECURRENT_STATE_STABILITY_REPAIR`.
- Calibration failure: `decision=d100_recurrent_calibration_failure`, `next=D100C_RECURRENT_CALIBRATION_REPAIR`.
- Top1 guard violation: `decision=top1_guard_invariant_violation`, `next=D100G_TOP1_GUARD_REPAIR`.
- D68 regression: `decision=d68_regression_detected`, `next=D100D_D68_REGRESSION_REPAIR`.
- Truth/oracle contamination: `decision=d100_truth_leak_or_oracle_contamination_detected`, `next=D100L_TRUTH_LEAK_REPAIR`.
- Shortcut memorization: `decision=d100_shortcut_memorization_detected`, `next=D100H_SHORTCUT_MEMORIZATION_REPAIR`.
- Split contamination: `decision=d100_split_contamination_detected`, `next=D100C_SPLIT_INTEGRITY_REPAIR`.
- Rust fallback: `decision=d100_rust_fallback_detected`, `next=D100R_RUST_PATH_REPAIR`.
- Reporting inconsistency: `decision=d100_invalid_metric_or_report_inconsistency`, `next=D100_REPORTING_REPAIR`.
- Incomplete run: `decision=d100_invalid_or_incomplete_run`, `next=D100_RETRY_WITH_FULL_AUDIT`.
