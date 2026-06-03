# D97 Mechanism Feature Audit and Surrogate Training Prototype Contract

## Purpose

D97 turns the D96 train-loop bridge into a controlled surrogate-training prototype. It trains/evaluates fair surrogate routers that imitate the validated symbolic D94/D95/D96 routing behavior using only inference-time, non-truth feature signals while preserving top1 guard, D68 protection, OOD/joint-boundary repair, low-cost/OOD/top1 ambiguity tail behavior, safety margins, truth isolation, Rust sparse invocation, `fallback_rows=0`, and `failed_jobs=[]`.

## Boundary

D97 is controlled symbolic ECF/IPF joint formula discovery only. It is not full VRAXION training, raw visual Raven/visual reasoning, AGI, consciousness, DNA/genome success, production readiness, or architecture superiority.

## Phase 0 upstream audit

The runner must verify branch/HEAD, check whether D96 commit `d05dd767224ce75ef8b647f3db0480a55efced7e` is present locally, verify `target/pilot_wave/d96_next_breakpoint_and_train_loop_bridge_plan/`, restore/rerun D96 if required, validate the D96 handoff, and write `d96_upstream_manifest.json`. D97 must not silently assume D96 was pushed.

## D96 handoff

D96 confirmed `decision=d96_breakpoint_map_complete_train_loop_bridge_ready`, `next=D97_MECHANISM_FEATURE_AUDIT_AND_SURROGATE_TRAINING_PROTOTYPE`, next breakpoint `COMBINED_LOW_COST_OOD_TOP1_AMBIGUITY_TAIL`, `trainable_surrogate_ready=true`, no forbidden features, no route-label leak risk, top1/D68/truth/oracle/Rust gates passing, `fallback_rows=0`, and `failed_jobs=[]`.

## Scale and stress settings

Requested main scale is seeds `18001,18002,18003,18004,18005,18006,18007,18008` with rows `train/test/ood=420` per seed/regime/split. Requested stress extension is seeds `18101,18102,18103,18104`, stress rows `train/test/ood=540`, and stress modes `combined_low_cost_ood_top1_ambiguity_tail`, `boundary_thin_margin`, `ood_support_shift_tail`, `joint_required_ambiguous_top1`, `low_cost_pressure_tail`, `external_required_tail`, `correlated_echo_distractor_tail`, `adversarial_counter_tail`, and `indistinguishable_abstain_tail`. Scale reductions must be recorded and cannot overclaim.

## Feature and target constraints

Fair surrogate inputs may include only inference-available non-truth features such as top1/top2 scores and gap, normalized support entropy, support dispersion, boundary-distance estimate, OOD shift proxy, low-cost pressure score, joint evidence pressure proxy, external requirement proxy, abstain/confidence risk proxy, support count estimate, and deterministic non-label symbolic structural features. Fair inputs must not include ground truth answers, symbolic correctness labels, oracle route labels, support-regime labels, oracle support, concrete counter identity, row IDs, seed IDs as predictive features, Python hashes, `repr(row)`, object IDs, file order, artifact indexes, filenames, generated answer labels, post-hoc correctness labels, or truth/regime/route-equivalent lookup features. The route distillation target is the validated symbolic router decision, not a ground-truth answer or oracle route label.

## Tracks and arms

D97 includes D96 replay, feature schema and forbidden feature audits, route-distillation target audit, split integrity, low-cost/OOD/top1 tail focus, surrogate linear/MLP/rule/threshold training, OOD/stress/min-seed/top1/D68/safety/oracle/support eval, label/regime/row/hash sentinels, feature ablation and importance stability, overfit/memorization audit, top1 ablation and partial-corruption controls, negative controls, truth/oracle isolation, Rust invocation, and schema/metric crosscheck. Arms include symbolic replay/reference arms, fair surrogate arms, feature ablation arms, guard controls, negative controls, leakage sentinels, concrete oracle reference-only, and truth-leak sentinel reference-only.

## Required reports

Artifacts are written under `target/pilot_wave/d97_mechanism_feature_audit_and_surrogate_training_prototype/` and must include `d96_upstream_manifest.json`, all feature/schema/distillation/split/training/surrogate/eval/sentinel/control/truth/Rust/schema reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

D97 passes only if D96 handoff/replay is valid, feature and distillation safety gates pass, the best fair surrogate reaches accuracy/min-seed/OOD/stress/overfit/safety/D68/top1/truth/oracle/support gates, targeted low-cost/OOD/top1 tail and combined OOD/joint-boundary carryover are non-regressed, adversarial sentinels collapse, memorization risk is low, feature importance stability is at least `0.70`, controls remain worse, deterministic replay/schema/metric crosscheck pass, Rust path is invoked, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Passing fair surrogate prototype: `decision=d97_surrogate_training_prototype_confirmed`, `next=D98_SURROGATE_ADVERSARIAL_SCALE_CONFIRM`.
- Promising surrogate with tail/min-seed failure: `decision=d97_surrogate_tail_risk_detected`, `next=D97T_SURROGATE_TAIL_RISK_REPAIR`.
- Safety/top1/D68 regression: `decision=d97_surrogate_safety_regression`, `next=D97S_SURROGATE_SAFETY_REPAIR`.
- Forbidden feature or label leak: `decision=d97_forbidden_feature_or_label_leak_detected`, `next=D97L_FEATURE_LEAK_REPAIR`.
- Shortcut memorization: `decision=d97_shortcut_memorization_detected`, `next=D97H_SHORTCUT_MEMORIZATION_REPAIR`.
- Split contamination: `decision=d97_split_contamination_detected`, `next=D97C_SPLIT_INTEGRITY_REPAIR`.
- No surrogate meets gates: `decision=d97_surrogate_training_not_confirmed`, `next=D97R_FEATURE_OR_TARGET_REPAIR`.
- D96 preservation regression: `decision=d96_preservation_regression_detected`, `next=D97U_UPSTREAM_PRESERVATION_REPAIR`.
- Top1 guard violation: `decision=top1_guard_invariant_violation`, `next=D97G_TOP1_GUARD_REPAIR`.
- D68 regression: `decision=d68_regression_detected`, `next=D97D_D68_REGRESSION_REPAIR`.
- Reporting inconsistency: `decision=d97_invalid_metric_or_report_inconsistency`, `next=D97_REPORTING_REPAIR`.
- Incomplete run: `decision=d97_invalid_or_incomplete_run`, `next=D97_RETRY_WITH_FULL_AUDIT`.
