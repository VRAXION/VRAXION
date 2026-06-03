# D86 Combined Low-Cost Top1 Ambiguity Repair Prototype Contract

## Purpose

D86 prototypes a targeted repair for the `COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY` breakpoint selected by D85 while preserving the top1/top2 sufficiency guard, D68 loss protection, safety margins, and Rust sparse invocation path in controlled symbolic ECF/IPF joint formula discovery. It does not add a broad architecture claim, does not loosen the top1 guard, and does not blindly optimize cost.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D85 commit `bded2327daca7288437352576698c9c05a5f4c91` is present and an ancestor of `HEAD`, restore/rerun D85 artifacts if missing, and write `d85_upstream_manifest.json`. D86 must not silently assume D85 was pushed.

## D85 handoff

D85 selected:

- `decision=combined_low_cost_top1_ambiguity_plan_selected`
- `next=D86_COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PROTOTYPE`
- `selected_repair_path=COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN`
- D84 dominant operational breakpoint `COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY` at `0.736`.
- top1 guard status `hard_invariant_and_control_required`.

## Tracks

1. `D85_REPLAY`
2. `COMBINED_LOW_COST_TOP1_AMBIGUITY_SWEEP`
3. `LOW_COST_PRESSURE_SWEEP`
4. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP`
5. `TOP1_GUARD_PRESERVATION`
6. `TOP1_GUARD_ABLATION_CONTROL`
7. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
8. `D68_CHEAP_TOP1_REGRESSION_GUARD`
9. `HARD_CORRELATED_JOINT_RECALL`
10. `HARD_ADVERSARIAL_JOINT_RECALL`
11. `OOD_SUPPORT_SHIFT_WATCH`
12. `EXTERNAL_REQUIRED_WATCH`
13. `INDISTINGUISHABLE_ABSTAIN_WATCH`
14. `SAFETY_MARGIN_WATCH`
15. `ORACLE_DISTANCE_FRONTIER`

## Arms

1. `D83_LOW_COST_REPAIR_REPLAY`
2. `D84_STRESS_BASELINE_REPLAY`
3. `COMBINED_LOW_COST_TOP1_REPAIR_BASE`
4. `COMBINED_LOW_COST_TOP1_REPAIR_COST_AWARE`
5. `COMBINED_LOW_COST_TOP1_REPAIR_HIGH_RECALL`
6. `COMBINED_LOW_COST_TOP1_REPAIR_BALANCED`
7. `COMBINED_LOW_COST_TOP1_REPAIR_LOW_COST`
8. `TOP1_TOP2_AMBIGUITY_REPAIR_ONLY`
9. `LOW_COST_PRESSURE_REPAIR_ONLY`
10. `LOW_COST_ONLY_CONTROL`
11. `TOP1_GUARD_ABLATION_CONTROL`
12. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
13. `RANDOM_ROUTER_CONTROL`
14. `NEVER_JOINT_CONTROL`
15. `ALWAYS_JOINT_CONTROL`
16. `CONCRETE_ORACLE_REFERENCE_ONLY`
17. `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`

## Required reports

Artifacts are written under `target/pilot_wave/d86_combined_low_cost_top1_ambiguity_repair_prototype/` and must include the upstream manifest, repair/sweep/watch reports, D68/safety/truth/Rust reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

The best fair D86 arm must satisfy the D85 proof gates: combined breakpoint at least `0.75`, low-cost breakpoint at least `0.74`, top1 ambiguity non-regression, core recall/safety support gates, D68 preservation `1.0`, no routing failures, top1 guard preserved and not weakened, ablation remains worse, Rust path invoked, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Passing repair: `decision=combined_low_cost_top1_ambiguity_repair_confirmed`, `next=D87_COMBINED_LOW_COST_TOP1_AMBIGUITY_SCALE_CONFIRM`.
- Safety/routing regression: `decision=combined_low_cost_top1_ambiguity_safety_regression`, `next=D86S_SAFETY_ROUTING_REPAIR`.
- Top1 guard weakening: `decision=top1_guard_invariant_violation`, `next=D86G_TOP1_GUARD_REPAIR`.
- No combined-breakpoint improvement: `decision=combined_low_cost_top1_ambiguity_repair_not_confirmed`, `next=D86_REPAIR`.

## Boundary

D86 only repairs the combined low-cost + top1/top2 ambiguity breakpoint while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
