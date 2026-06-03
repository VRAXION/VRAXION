# D99 Recurrent Routing Microcircuit Prototype Contract

## Purpose

D99 prototypes a recurrent routing microcircuit that reproduces or improves the D98 `SURROGATE_SMALL_MLP_FAIR` routing behavior using an explicit iterative hidden-state update loop. The prototype must consume only inference-time symbolic/proxy features, converge to safe route decisions, preserve top1/D68/safety/truth/Rust/fallback invariants, survive OOD/stress/feature/state corruption, and demonstrate non-trivial loop utility beyond a feedforward wrapper.

## Boundary

D99 is controlled symbolic ECF/IPF joint formula discovery only. The formula solver remains symbolic. D99 does not claim a full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, production readiness, or architecture superiority.

## Phase 0 upstream audit

The runner must verify branch/HEAD, check D98 commit `565986b2cdee64251d45e0048fe96f76a59b712b`, verify `target/pilot_wave/d98_surrogate_adversarial_scale_confirm/`, validate D98 decision `d98_surrogate_adversarial_scale_confirmed` with next `D99_RECURRENT_ROUTING_MICROCIRCUIT_PROTOTYPE`, restore/rerun D98 if required, and write `d98_upstream_manifest.json`. D99 must not silently assume D98 was pushed.

## D98 handoff

D98 confirmed `SURROGATE_SMALL_MLP_FAIR` with test accuracy `0.9945`, OOD accuracy `0.9921`, stress accuracy `0.9913`, min-seed accuracy `0.9908`, worst-seed accuracy `0.9898`, low-cost/OOD/top1 tail score `0.745`, min-seed tail score `0.741`, combined OOD/joint-boundary breakpoint `0.756`, min-seed combined breakpoint `0.753`, top1 guard preserved, D68 preservation `1.0`, truth/oracle/Rust gates passing, `fallback_rows=0`, and `failed_jobs=[]`.

## Scale and stress settings

Requested scale is seeds `20001,20002,20003,20004,20005,20006,20007,20008,20009,20010` with rows `train/test/ood=480` per seed/regime/split. Requested stress extension is stress seeds `20101,20102,20103,20104,20105,20106`, stress rows `train/test/ood=640`, and stress modes `combined_low_cost_ood_top1_ambiguity_tail`, `boundary_thin_margin`, `ood_support_shift_tail`, `joint_required_ambiguous_top1`, `low_cost_pressure_tail`, `external_required_tail`, `correlated_echo_distractor_tail`, `adversarial_counter_tail`, `indistinguishable_abstain_tail`, `feature_noise_tail`, `feature_dropout_tail`, `calibration_pressure_tail`, `mixed_tail_compound`, `worst_seed_replay_tail`, `recurrent_state_noise_tail`, `recurrent_state_reset_tail`, `recurrent_oscillation_tail`, and `recurrent_halting_pressure_tail`. Any scale reduction must be recorded and cannot overclaim.

## Recurrent mechanism requirements

Each fair recurrent arm must initialize a small hidden routing state from allowed features only, update that state for `K` steps, compute route logits at each step, optionally halt on confidence/convergence criteria, output a final route, and record per-step logits, hidden-state norms, hidden-state delta norms, confidence, halt step, convergence/oscillation/non-convergence flags, final route, teacher route, allowed feature names, and forbidden feature audit result.

## Feature and target constraints

Fair recurrent inputs and hidden-state initialization may use only D97/D98-approved inference-time non-truth symbolic/proxy features. Fair arms must not use truth labels, oracle route labels, support-regime labels, row IDs, seed IDs, Python hashes, file order, artifact indexes, filenames, object IDs, `repr(row)`, generated answer labels, post-hoc correctness labels, hidden-state initialization derived from forbidden fields, or any truth/regime/route-equivalent lookup. The target remains the validated symbolic router decision generated from inference-available signals only.

## Required reports

Artifacts are written under `target/pilot_wave/d99_recurrent_routing_microcircuit_prototype/` and must include `d98_upstream_manifest.json`, all recurrent training/loop/convergence/halting/oscillation/state/usefulness/carryover/guard/safety/sentinel/provenance reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

D99 passes only if D98 handoff/replay is valid, requested and actual scale are recorded with `scale_reduced=false`, all stress modes execute, the best fair recurrent arm meets accuracy/min-seed/worst-seed/overfit/safety/top1/D68/truth/oracle/support/step gates, recurrent convergence and loop-utility gates pass, targeted low-cost/OOD/top1 tail and combined OOD/joint carryover are preserved, state robustness gates pass, leak/shortcut sentinels collapse, no forbidden features or split contamination appear, controls remain worse, deterministic replay/schema/metric crosscheck pass, Rust path is invoked, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Passing recurrent prototype: `decision=d99_recurrent_routing_microcircuit_prototype_confirmed`, `next=D100_RECURRENT_ROUTING_MICROCIRCUIT_SCALE_CONFIRM`.
- Recurrent loop not useful: `decision=d99_recurrent_loop_utility_not_confirmed`, `next=D99U_LOOP_UTILITY_REPAIR`.
- Tail/worst-seed failure: `decision=d99_recurrent_tail_risk_detected`, `next=D99T_RECURRENT_TAIL_RISK_REPAIR`.
- State robustness failure: `decision=d99_recurrent_state_robustness_failure`, `next=D99F_STATE_ROBUSTNESS_REPAIR`.
- Calibration failure: `decision=d99_recurrent_calibration_failure`, `next=D99C_CALIBRATION_REPAIR`.
- Top1 guard violation: `decision=top1_guard_invariant_violation`, `next=D99G_TOP1_GUARD_REPAIR`.
- D68 regression: `decision=d68_regression_detected`, `next=D99D_D68_REGRESSION_REPAIR`.
- Truth/oracle contamination: `decision=d99_truth_leak_or_oracle_contamination_detected`, `next=D99L_TRUTH_LEAK_REPAIR`.
- Shortcut memorization: `decision=d99_shortcut_memorization_detected`, `next=D99H_SHORTCUT_MEMORIZATION_REPAIR`.
- Split contamination: `decision=d99_split_contamination_detected`, `next=D99S_SPLIT_INTEGRITY_REPAIR`.
- Rust fallback: `decision=d99_rust_fallback_detected`, `next=D99R_RUST_PATH_REPAIR`.
- Reporting inconsistency: `decision=d99_invalid_metric_or_report_inconsistency`, `next=D99_REPORTING_REPAIR`.
- Incomplete run: `decision=d99_invalid_or_incomplete_run`, `next=D99_RETRY_WITH_FULL_AUDIT`.
