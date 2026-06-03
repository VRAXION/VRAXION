# D81 Breakpoint Repair Or Generalization Plan Contract

## Purpose

D81 plans the next targeted repair/generalization milestone after D80 stress mapping of the integrated joint-recall counter-action router. D81 does not change the core mechanism, does not loosen the top1 sufficiency guard blindly, and does not add any broad architecture claim.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D80 commit `bbf54f33b13163075f617bebca82e771463305c3` is present and an ancestor of `HEAD`, restore/rerun D80 artifacts if missing, and write `d80_upstream_manifest.json`. The manifest must confirm D80 `decision=integrated_joint_recall_stress_map_completed`, `next=D81_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`, and `stress_map_complete=true`. D81 must not silently assume D80 was pushed.

## D80 handoff reference

D80 completed the stress map with:

- `decision=integrated_joint_recall_stress_map_completed`
- `next=D81_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`
- `core_d79_holds_standard_stress=true`
- `stress_map_complete=true`
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`

Dominant breakpoint: `TOP1_GUARD_CORRUPTION_OR_ABLATION`.

Top1 guard ablation evidence:

- `routing_failure_rows=45`
- `D68_loss_repair_preservation_rate=0.961538`
- `weak_top1_top2_path_failure_rate=0.004`
- `top1_top2_sufficient_false_joint_rate=0.011`

Operational breakpoints:

- `LOW_COST_PRESSURE=0.70`
- `TOP1_TOP2_SUFFICIENCY_AMBIGUITY=0.74`
- `OOD_SUPPORT_DISTRIBUTION_SHIFT=0.76`
- `JOINT_REQUIRED_NEAR_BOUNDARY=0.78`
- `INDISTINGUISHABLE_BOUNDARY=0.82`
- `EXTERNAL_REQUIRED_PRESSURE=0.84`
- `ADVERSARIAL_DISTRACTOR=0.86`
- `CORRELATED_ECHO=0.88`

## Planning questions

1. Is top1 guard corruption a repair target, a hard invariant, or both?
2. Which operational breakpoint should be repaired first?
3. Is the best next milestone low-cost pressure repair, top1/top2 ambiguity repair, OOD generalization, or joint-boundary repair?
4. Should D82 be single-target or combined repair?
5. What exact proof gates must D82 satisfy?

## Candidate repair paths

1. `TOP1_GUARD_HARDENING_PLAN`
2. `LOW_COST_PRESSURE_REPAIR_PLAN`
3. `TOP1_TOP2_AMBIGUITY_REPAIR_PLAN`
4. `OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN`
5. `JOINT_REQUIRED_NEAR_BOUNDARY_REPAIR_PLAN`
6. `EXTERNAL_PRESSURE_REPAIR_PLAN`
7. `COMBINED_LOW_COST_TOP1_OOD_PLAN`
8. `NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE`

## Required reports

Artifacts are written under `target/pilot_wave/d81_breakpoint_repair_or_generalization_plan/`:

- `d80_upstream_manifest.json`
- `breakpoint_ranking_report.json`
- `top1_guard_invariant_report.json`
- `operational_breakpoint_priority_report.json`
- `repair_candidate_roi_report.json`
- `generalization_candidate_report.json`
- `D82_proof_gate_report.json`
- `risk_register.json`
- `truth_leak_audit_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

## Decisions

- Top1 guard hardening first: `decision=top1_guard_hardening_plan_selected`, `next=D82_TOP1_GUARD_HARDENING_PROTOTYPE`.
- Low-cost pressure first: `decision=low_cost_pressure_repair_plan_selected`, `next=D82_LOW_COST_PRESSURE_REPAIR_WITH_TOP1_GUARD`.
- Top1/top2 ambiguity first: `decision=top1_top2_ambiguity_repair_plan_selected`, `next=D82_TOP1_TOP2_SUFFICIENCY_AMBIGUITY_REPAIR`.
- OOD shift first: `decision=ood_support_shift_generalization_plan_selected`, `next=D82_OOD_SUPPORT_SHIFT_GENERALIZATION_PROTOTYPE`.
- Combined plan required: `decision=combined_breakpoint_repair_plan_selected`, `next=D82_COMBINED_LOW_COST_TOP1_OOD_REPAIR`.
- No safe repair target: `decision=breakpoint_repair_plan_not_ready`, `next=D81_REPAIR_OR_BOUND_ACCEPTANCE`.

## Hard gates

D81 must avoid full brain/Raven/AGI/consciousness claims, fake metrics, and truth leakage. Oracle/reference arms remain reference-only. The top1 guard must not be weakened without explicit guard proof. D68 recurrence prevention must be explicit. D82 gates must be measurable. Failed jobs must be visible.

## Boundary

D81 only plans breakpoint repair/generalization after D80 stress mapping in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
