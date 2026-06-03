# D96 Next Breakpoint and Train Loop Bridge Plan Contract

## Purpose

D96 maps the next strongest remaining breakpoint after the D95 combined OOD + joint-boundary scale confirmation and audits whether the current symbolic routing behavior is ready to be distilled into a trainable surrogate. D96 is a controlled symbolic ECF/IPF diagnostic and planning milestone only; it is not full model training and does not add a broad architecture claim.

## Phase 0 upstream audit

The runner must verify the current branch and `HEAD`, check whether D95 commit `1ecc5694f404da52ee1793a1fda3e18ca3bef045` is present locally, verify `target/pilot_wave/d95_combined_ood_joint_boundary_scale_confirm/`, restore/rerun D95 artifacts if required, validate the D95 handoff, and write `d95_upstream_manifest.json`. D96 must not silently assume D95 was pushed.

## D95 handoff

D95 confirmed:

- `decision=combined_ood_joint_boundary_scale_confirmed`
- `next=D96_NEXT_BREAKPOINT_OR_GENERALIZATION_PLAN`
- `best_arm=COMBINED_OOD_JOINT_BOUNDARY_REPAIR_COST_AWARE_SCALE`
- `combined_ood_joint_boundary_breakpoint=0.757`
- `min_seed_combined_ood_joint_boundary_breakpoint=0.753`
- `scale_reduced=false`
- `D68_loss_repair_preservation_rate=1.0`
- `top1_guard_preserved=true`
- `top1_guard_weakened=false`
- `truth_leak_audit_passed=true`
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`

## Scale and stress settings

Requested main scale is seeds `17001,17002,17003,17004,17005,17006,17007,17008` with rows `train/test/ood=360` per seed/regime/split. Requested stress extension is seeds `17101,17102,17103,17104`, stress rows `train/test/ood=480`, and stress modes `boundary_thin_margin`, `ood_support_shift_tail`, `joint_required_ambiguous_top1`, `low_cost_pressure_tail`, `external_required_tail`, `correlated_echo_distractor_tail`, `adversarial_counter_tail`, and `indistinguishable_abstain_tail`. If runtime constraints force a reduction, requested and actual scale must be recorded and the decision must not overclaim.

## Tracks

D96 includes D95 replay, post-D95 breakpoint ranking, stress and min-seed tail maps, OOD/joint/top1/low-cost/external/correlated/adversarial/indistinguishable tail audits, oracle and support frontier audits, fair feature signal audit, trainable surrogate feasibility audit, routing-label distillation audit, top1/D68/truth/row-id/hash/oracle/Rust audits, and report-schema consistency audit.

## Arms

D96 evaluates D95 replay/stress replay, a post-D95 breakpoint ranker, fair feature extractor, fair trainable surrogate linear-probe/small-MLP/rule-distillation planning arms, top1 ablation and partial-corruption controls, random/never/always controls, label-shuffle/regime-label/row-id/Python-hash sentinels, concrete oracle reference-only, and truth-leak sentinel reference-only arms.

## Fair feature constraints

Fair trainable surrogate arms may use only inference-available non-truth features such as top1/top2 scores and gap, normalized support entropy, support dispersion, boundary-distance estimate, OOD shift proxy, low-cost pressure score, joint evidence pressure proxy, external requirement proxy, abstain/confidence risk proxies, support count estimate, and non-label symbolic structural features. Fair arms must not use ground truth answers, correct route labels, support-regime labels, oracle support, concrete counter identity, row IDs, seed IDs as predictive features, Python hashes, artifact lookup keys, generated answer labels, or post-hoc correctness labels.

## Required reports

Artifacts are written under `target/pilot_wave/d96_next_breakpoint_and_train_loop_bridge_plan/` and must include the upstream manifest, breakpoint/stress/tail reports, oracle/support reports, feature/surrogate/distillation reports, top1/D68/truth/row-id/hash/oracle/Rust reports, schema consistency report, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

D96 passes only if D95 handoff/replay is valid, the next breakpoint and min-seed tail breakpoint are identified with rank confidence at least `0.75` and seed stability at least `0.70`, stress-tail false confidence is at most `0.0046`, stress-tail routing failures are `0`, D95 preservation metrics hold, top1/D68/truth/oracle/Rust/fallback/failed-job hard gates hold, fair feature signal counts are sufficient, forbidden features are absent, route distillation target is defined without label leak risk, surrogate readiness gates pass when ready, adversarial sentinels do not contaminate fair arms, report schema consistency passes, metric crosscheck passes, and deterministic replay passes.

## Decisions

- Stable breakpoint map and train-loop bridge ready: `decision=d96_breakpoint_map_complete_train_loop_bridge_ready`, `next=D97_MECHANISM_FEATURE_AUDIT_AND_SURROGATE_TRAINING_PROTOTYPE`.
- Stable breakpoint map but train-loop bridge not ready: `decision=d96_breakpoint_map_complete_surrogate_not_ready`, `next=D97_FEATURE_SIGNAL_REPAIR_OR_LABEL_DISTILLATION_AUDIT`.
- Tail-risk breakpoint dominates: `decision=d96_tail_risk_breakpoint_identified`, `next=D97_TAIL_RISK_REPAIR_PLAN`.
- D95 preservation regresses: `decision=d95_preservation_regression_detected`, `next=D96S_PRESERVATION_REPAIR`.
- Top1 guard weakens: `decision=top1_guard_invariant_violation`, `next=D96G_TOP1_GUARD_REPAIR`.
- D68 regresses: `decision=d68_regression_detected`, `next=D96D_D68_REGRESSION_REPAIR`.
- Truth leak or oracle contamination: `decision=truth_leak_or_oracle_contamination_detected`, `next=D96L_TRUTH_LEAK_REPAIR`.
- Row-id/hash shortcut: `decision=row_id_or_hash_shortcut_detected`, `next=D96H_SHORTCUT_LEAK_REPAIR`.
- Metric/report inconsistency: `decision=d96_invalid_metric_or_report_inconsistency`, `next=D96R_REPORTING_REPAIR`.
- Incomplete run: `decision=d96_invalid_or_incomplete_run`, `next=D96_RETRY_WITH_FULL_AUDIT`.

## Boundary

D96 is only a next-breakpoint map and train-loop bridge audit for controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
