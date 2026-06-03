# D93 Breakpoint Repair or Generalization Plan Contract

## Purpose

D93 plans the next targeted repair/generalization milestone after D92 stress-mapped the D91 combined low-cost + OOD repair. D93 selects a D94 target without adding broad architecture claims, without weakening the top1 guard, and without changing the controlled symbolic ECF/IPF joint formula discovery boundary.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D92 commit `bec4d531f476c6e5d2efa4b2800fe480510260dc` is present and an ancestor of `HEAD`, restore/rerun D92 artifacts if missing, and write `d92_upstream_manifest.json`. D93 must not silently assume D92 was pushed.

## D92 handoff

D92 completed:

- `decision=combined_low_cost_ood_stress_map_completed`
- `next=D93_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`
- `dominant_breakpoint=COMBINED_OOD_JOINT_BOUNDARY`
- `best_fair_arm=D91_COMBINED_LOW_COST_OOD_REPLAY`
- `combined_low_cost_plus_ood_breakpoint=0.763`
- `ood_support_distribution_shift_breakpoint=0.760`
- `low_cost_pressure_breakpoint=0.749`
- `combined_low_cost_plus_top1_ambiguity_breakpoint=0.754`
- `top1_top2_sufficiency_ambiguity_breakpoint=0.746`
- `combined_low_cost_ood_top1_ambiguity_breakpoint=0.741`
- `combined_ood_joint_boundary_breakpoint=0.739`
- `joint_required_near_boundary_breakpoint=0.779`
- D91 core held, D68 preservation was `1.0`, routing failures were `0`, and the top1 guard was preserved.

## Candidate repair paths

1. `COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PLAN`
2. `JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN`
3. `OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN`
4. `COMBINED_LOW_COST_OOD_JOINT_REPAIR_PLAN`
5. `COMBINED_OOD_JOINT_TOP1_GUARD_PLAN`
6. `EXTERNAL_PRESSURE_REPAIR_PLAN`
7. `TOP1_GUARD_HARDENING_REFERENCE_ONLY`
8. `NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE`

## Required reports

Artifacts are written under `target/pilot_wave/d93_breakpoint_repair_or_generalization_plan/` and must include `d92_upstream_manifest.json`, breakpoint ranking, combined OOD + joint-boundary analysis, candidate/generalization reports, top1 guard invariant report, ROI report, D94 proof-gate report, risk register, truth-leak audit, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Planning metrics

D93 scores breakpoint severity, expected frequency, support-cost impact, routing-risk impact, OOD-risk impact, joint-boundary risk impact, D68 recurrence risk, top1 guard dependency, implementation complexity, expected ROI, required ablations/controls, D94 proof gates, and recommended next milestone.

## Decisions

- Combined OOD + joint-boundary selected: `decision=combined_ood_joint_boundary_plan_selected`, `next=D94_COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PROTOTYPE`.
- Joint-boundary alone selected: `decision=joint_boundary_repair_plan_selected`, `next=D94_JOINT_REQUIRED_BOUNDARY_REPAIR_PROTOTYPE`.
- OOD support generalization selected: `decision=ood_support_shift_generalization_plan_selected`, `next=D94_OOD_SUPPORT_SHIFT_GENERALIZATION_PROTOTYPE`.
- Combined low-cost + OOD + joint selected: `decision=combined_low_cost_ood_joint_plan_selected`, `next=D94_COMBINED_LOW_COST_OOD_JOINT_REPAIR_PROTOTYPE`.
- No safe target: `decision=breakpoint_repair_plan_not_ready`, `next=D93_REPAIR_OR_BOUND_ACCEPTANCE`.

## Hard gates

D93 must make D68 recurrence prevention explicit, keep D94 gates measurable, prevent truth leakage, keep oracle/reference arms reference-only, keep failed jobs visible, and preserve the top1 guard as a hard invariant that must not be weakened.

## Boundary

D93 only plans repair/generalization after D92 stress mapping in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
