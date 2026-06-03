# D94 Combined OOD Joint Boundary Repair Prototype Contract

## Purpose

D94 prototypes a targeted repair for the D93-selected combined OOD + joint-boundary breakpoint while preserving low-cost/OOD gains, the top1/top2 sufficiency guard, D68 loss repair, safety margins, truth-leak guards, and Rust sparse invocation in controlled symbolic ECF/IPF joint formula discovery. D94 does not add a broad architecture mechanism and must not weaken the top1 guard.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D93 commit `7a55a75fa050c68c2aec3f5f219127e087a51bf7` is present and an ancestor of `HEAD`, restore/rerun D93 artifacts if missing, and write `d93_upstream_manifest.json`. D94 must not silently assume D93 was pushed.

## D93 handoff

D93 selected:

- `decision=combined_ood_joint_boundary_plan_selected`
- `next=D94_COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PROTOTYPE`
- `selected_repair_path=COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PLAN`
- `dominant_breakpoint=COMBINED_OOD_JOINT_BOUNDARY`
- `top_breakpoint_threshold=0.739`
- `expected_ROI=0.79`

## Tracks

1. `D93_REPLAY`
2. `COMBINED_OOD_JOINT_BOUNDARY_SWEEP`
3. `JOINT_REQUIRED_NEAR_BOUNDARY_SWEEP`
4. `OOD_SUPPORT_DISTRIBUTION_SHIFT_SWEEP`
5. `COMBINED_LOW_COST_PLUS_OOD_WATCH`
6. `COMBINED_LOW_COST_OOD_TOP1_AMBIGUITY_WATCH`
7. `LOW_COST_PRESSURE_WATCH`
8. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY_WATCH`
9. `TOP1_GUARD_PRESERVATION`
10. `TOP1_GUARD_ABLATION_CONTROL`
11. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
12. `D68_CHEAP_TOP1_REGRESSION_GUARD`
13. `HARD_CORRELATED_JOINT_RECALL`
14. `HARD_ADVERSARIAL_JOINT_RECALL`
15. `EXTERNAL_REQUIRED_WATCH`
16. `INDISTINGUISHABLE_ABSTAIN_WATCH`
17. `SAFETY_MARGIN_WATCH`
18. `ORACLE_DISTANCE_FRONTIER`

## Arms

D94 evaluates D91/D92 replay arms, combined OOD + joint-boundary repair arms, joint-only/OOD-only/combined-low-cost-OOD repair-only arms, low-cost/OOD/joint controls, top1 guard ablation/corruption controls, random/never/always controls, concrete oracle reference-only, and truth-leak sentinel reference-only arms.

## Required reports

Artifacts are written under `target/pilot_wave/d94_combined_ood_joint_boundary_repair_prototype/` and must include `d93_upstream_manifest.json`, combined OOD + joint repair/sweep reports, joint/OOD/low-cost/top1 watch reports, top1 guard reports, D68/safety/oracle/support/truth/Rust reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

The best fair D94 arm must improve the combined OOD + joint-boundary breakpoint to at least `0.755`, preserve combined low-cost + OOD at least `0.760`, preserve OOD shift and joint boundary non-regression versus D92, preserve combined low-cost + OOD + top1 non-regression, hold accuracy/recall/safety/support gates, preserve D68 at `1.0`, keep routing failures at `0`, preserve the top1 guard without weakening, keep ablation worse, invoke the Rust path, keep `fallback_rows=0`, and expose `failed_jobs=[]`.

## Decisions

- Passing repair: `decision=combined_ood_joint_boundary_repair_confirmed`, `next=D95_COMBINED_OOD_JOINT_BOUNDARY_SCALE_CONFIRM`.
- Repair improves but safety/routing regresses: `decision=combined_ood_joint_boundary_safety_regression`, `next=D94S_SAFETY_ROUTING_REPAIR`.
- Top1 guard weakens: `decision=top1_guard_invariant_violation`, `next=D94G_TOP1_GUARD_REPAIR`.
- Repair does not improve: `decision=combined_ood_joint_boundary_repair_not_confirmed`, `next=D94_REPAIR`.

## Boundary

D94 only repairs the combined OOD + joint-boundary breakpoint while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
