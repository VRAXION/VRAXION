# D85 Breakpoint Repair Or Generalization Plan Contract

## Purpose

D85 plans the next targeted repair/generalization milestone after the D84 stress map. It selects a D86 target for the repaired low-cost-pressure route while preserving the top1/top2 sufficiency guard and D68 loss protection. It does not add a broad architecture claim and must not loosen the top1 guard.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D84 commit `9cb2341282d5d75f8dd1cc697751908f2f138ae0` is present and an ancestor of `HEAD`, restore/rerun D84 artifacts if missing, and write `d84_upstream_manifest.json`. D85 must not silently assume D84 was pushed.

## D84 handoff

D84 completed:

- `decision=low_cost_pressure_repair_stress_map_completed`
- `next=D85_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`
- `dominant_breakpoint=COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY`
- `low_cost_pressure_breakpoint=0.751`
- `stress_map_complete=true`
- `core_D83_holds_standard_stress=true`

The top1 guard remains a hard invariant. D85 may choose a repair that improves combined low-cost/top1 ambiguity, but it must not weaken the top1 guard or treat top1 guard ablation as a cost-saving option.

## Candidate repair paths

1. `COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN`
2. `TOP1_TOP2_AMBIGUITY_REPAIR_WITH_LOW_COST_GUARD`
3. `LOW_COST_PRESSURE_FOLLOWUP_REPAIR`
4. `OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN`
5. `JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN`
6. `COMBINED_LOW_COST_TOP1_OOD_PLAN`
7. `TOP1_GUARD_HARDENING_REFERENCE_ONLY`
8. `NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE`

## Required reports

Artifacts are written under `target/pilot_wave/d85_breakpoint_repair_or_generalization_plan/`:

- `d84_upstream_manifest.json`
- `breakpoint_ranking_report.json`
- `combined_breakpoint_analysis_report.json`
- `top1_guard_invariant_report.json`
- `repair_candidate_roi_report.json`
- `generalization_candidate_report.json`
- `D86_proof_gate_report.json`
- `risk_register.json`
- `truth_leak_audit_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

## Planning metrics

D85 must report breakpoint severity, expected occurrence/frequency, support-cost impact, routing-risk impact, D68 recurrence risk, top1 guard dependency, implementation complexity, expected ROI, required ablations/controls, D86 proof gates, and the recommended next milestone.

## Decisions

- Combined low-cost + top1 ambiguity selected: `decision=combined_low_cost_top1_ambiguity_plan_selected`, `next=D86_COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PROTOTYPE`.
- Top1/top2 ambiguity alone selected: `decision=top1_top2_ambiguity_repair_plan_selected`, `next=D86_TOP1_TOP2_AMBIGUITY_REPAIR_WITH_LOW_COST_GUARD`.
- OOD shift selected: `decision=ood_support_shift_generalization_plan_selected`, `next=D86_OOD_SUPPORT_SHIFT_GENERALIZATION_PROTOTYPE`.
- Joint-boundary selected: `decision=joint_boundary_repair_plan_selected`, `next=D86_JOINT_REQUIRED_BOUNDARY_REPAIR_PROTOTYPE`.
- No safe target: `decision=breakpoint_repair_plan_not_ready`, `next=D85_REPAIR_OR_BOUND_ACCEPTANCE`.

## Hard gates

No full brain/Raven/AGI/consciousness claims; no fake metrics; no truth leakage; oracle/reference arms reference-only; top1 guard must not be weakened; D68 recurrence prevention must be explicit; D86 gates must be measurable; failed jobs must remain visible.

## Boundary

D85 only plans repair/generalization after D84 stress mapping in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
