# D89 Breakpoint Repair or Generalization Plan Contract

## Purpose

D89 plans the next targeted repair/generalization milestone after the D88 stress map of the D87 combined low-cost + top1/top2 ambiguity repair. It selects a bounded next step for controlled symbolic ECF/IPF joint formula discovery and does not add broad architecture claims.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D88 commit `05a429f90fa55bd7c2be3218cbe74b6bcf52147c` is present and an ancestor of `HEAD`, restore/rerun D88 artifacts if missing, and write `d88_upstream_manifest.json`. D89 must not silently assume D88 was pushed.

## D88 handoff

D88 completed with:

- `decision=combined_low_cost_top1_ambiguity_stress_map_completed`
- `next=D89_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`
- `best_fair_arm=D87_COMBINED_REPAIR_REPLAY`
- `dominant_breakpoint=COMBINED_LOW_COST_PLUS_OOD`
- `hard_invariant_breakpoint=TOP1_GUARD_CORRUPTION_OR_ABLATION`
- `combined_low_cost_plus_top1_ambiguity_breakpoint=0.755`
- `low_cost_pressure_breakpoint=0.750`
- `top1_top2_sufficiency_ambiguity_breakpoint=0.746`
- `combined_low_cost_plus_ood_breakpoint=0.744`
- `ood_support_distribution_shift_breakpoint=0.758`
- top1 guard preserved, not weakened, and ablation remained worse.

## Candidate repair paths

1. `COMBINED_LOW_COST_OOD_REPAIR_PLAN`
2. `OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN`
3. `LOW_COST_OOD_WITH_TOP1_GUARD_PLAN`
4. `COMBINED_LOW_COST_TOP1_OOD_PLAN`
5. `JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN`
6. `EXTERNAL_PRESSURE_REPAIR_PLAN`
7. `TOP1_GUARD_HARDENING_REFERENCE_ONLY`
8. `NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE`

## Required reports

Artifacts are written under `target/pilot_wave/d89_breakpoint_repair_or_generalization_plan/` and must include `d88_upstream_manifest.json`, `breakpoint_ranking_report.json`, `combined_low_cost_ood_analysis_report.json`, `ood_generalization_candidate_report.json`, `top1_guard_invariant_report.json`, `repair_candidate_roi_report.json`, `generalization_candidate_report.json`, `D90_proof_gate_report.json`, `risk_register.json`, `truth_leak_audit_report.json`, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Planning metrics

D89 must report breakpoint severity, expected occurrence/frequency, support-cost impact, routing-risk impact, OOD-risk impact, D68 recurrence risk, top1 guard dependency, implementation complexity, expected ROI, required ablations/controls, D90 proof gates, and the recommended next milestone.

## Decisions

- Combined low-cost + OOD best: `decision=combined_low_cost_ood_plan_selected`, `next=D90_COMBINED_LOW_COST_OOD_REPAIR_PROTOTYPE`.
- OOD generalization alone best: `decision=ood_support_shift_generalization_plan_selected`, `next=D90_OOD_SUPPORT_SHIFT_GENERALIZATION_PROTOTYPE`.
- Combined low-cost + top1 + OOD required: `decision=combined_low_cost_top1_ood_plan_selected`, `next=D90_COMBINED_LOW_COST_TOP1_OOD_REPAIR_PROTOTYPE`.
- Joint-boundary best: `decision=joint_boundary_repair_plan_selected`, `next=D90_JOINT_REQUIRED_BOUNDARY_REPAIR_PROTOTYPE`.
- No safe repair target: `decision=breakpoint_repair_plan_not_ready`, `next=D89_REPAIR_OR_BOUND_ACCEPTANCE`.

## Hard gates

No full brain/Raven/AGI/consciousness claims; no fake metrics; no truth leakage; oracle/reference arms remain reference-only; the top1 guard must not be weakened; D68 recurrence prevention must be explicit; D90 gates must be measurable; and `failed_jobs` must be visible.

## Boundary

D89 only plans repair/generalization after D88 stress mapping in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
