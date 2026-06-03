# D90 Combined Low-Cost OOD Repair Prototype Contract

## Purpose

D90 prototypes a bounded repair for the `COMBINED_LOW_COST_PLUS_OOD` support-distribution-shift breakpoint selected by D89. The repair remains inside controlled symbolic ECF/IPF joint formula discovery, preserves the top1/top2 sufficiency guard and D68 loss repair, keeps the formula solver symbolic, and does not add broad architecture claims.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D89 commit `e0d755d2bc166f3b538bc75ddc95c366f40d320b` is present and an ancestor of `HEAD`, restore/rerun D89 artifacts if missing, and write `d89_upstream_manifest.json`. D90 must not silently assume D89 was pushed.

## D89 handoff

D89 selected:

- `decision=combined_low_cost_ood_plan_selected`
- `next=D90_COMBINED_LOW_COST_OOD_REPAIR_PROTOTYPE`
- `selected_repair_path=COMBINED_LOW_COST_OOD_REPAIR_PLAN`
- `dominant_breakpoint=COMBINED_LOW_COST_PLUS_OOD`
- `top_breakpoint_threshold=0.744`
- top1 guard status is `hard_invariant_must_not_be_weakened`.

## Tracks

1. `D89_REPLAY`
2. `COMBINED_LOW_COST_OOD_SWEEP`
3. `OOD_SUPPORT_DISTRIBUTION_SHIFT_SWEEP`
4. `LOW_COST_PRESSURE_SWEEP`
5. `COMBINED_LOW_COST_TOP1_AMBIGUITY_WATCH`
6. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY_WATCH`
7. `TOP1_GUARD_PRESERVATION`
8. `TOP1_GUARD_ABLATION_CONTROL`
9. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
10. `D68_CHEAP_TOP1_REGRESSION_GUARD`
11. `HARD_CORRELATED_JOINT_RECALL`
12. `HARD_ADVERSARIAL_JOINT_RECALL`
13. `EXTERNAL_REQUIRED_WATCH`
14. `INDISTINGUISHABLE_ABSTAIN_WATCH`
15. `SAFETY_MARGIN_WATCH`
16. `ORACLE_DISTANCE_FRONTIER`

## Arms

D90 evaluates D87/D88 replays, combined low-cost OOD repair variants, repair-only controls, low-cost/OOD controls, top1 guard ablation/corruption controls, random/never/always joint controls, concrete oracle reference-only, and truth-leak sentinel reference-only arms.

## Required reports

Artifacts are written under `target/pilot_wave/d90_combined_low_cost_ood_repair_prototype/` and must include `d89_upstream_manifest.json`, repair/sweep/watch reports, top1 guard reports, D68/safety/oracle/support/truth/Rust reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

The best fair D90 arm must satisfy the D89 proof gates: combined low-cost + OOD breakpoint at least `0.760`, combined low-cost + top1 ambiguity at least `0.750`, low-cost pressure at least `0.740`, OOD support shift non-regression versus D88, core recall/safety/support thresholds, D68 preservation `1.0`, no routing failures, top1 guard preserved and not weakened, ablation remains worse, Rust path invoked, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Passing repair prototype: `decision=combined_low_cost_ood_repair_confirmed`, `next=D91_COMBINED_LOW_COST_OOD_SCALE_CONFIRM`.
- Safety/routing regression: `decision=combined_low_cost_ood_safety_regression`, `next=D90S_SAFETY_ROUTING_REPAIR`.
- Top1 guard weakening: `decision=top1_guard_invariant_violation`, `next=D90G_TOP1_GUARD_REPAIR`.
- Repair not confirmed: `decision=combined_low_cost_ood_repair_not_confirmed`, `next=D90_REPAIR`.

## Boundary

D90 only repairs the combined low-cost + OOD support distribution shift breakpoint while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
