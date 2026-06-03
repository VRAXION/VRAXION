# D88 Combined Low-Cost Top1 Ambiguity Stress Map Contract

## Purpose

D88 stress-tests the D87 scale-confirmed combined low-cost + top1/top2 ambiguity repair and maps the next dominant breakpoint while preserving the top1/top2 sufficiency guard, D68 protection, safety margins, and Rust sparse invocation path in controlled symbolic ECF/IPF joint formula discovery. It does not add a new broad architecture mechanism.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D87 commit `ef451771628c01f4509b5eb64c3f7ae15c5974ea` is present and an ancestor of `HEAD`, restore/rerun D87 artifacts if missing, and write `d87_upstream_manifest.json`. D88 must not silently assume D87 was pushed.

## D87 handoff

D87 confirmed:

- `decision=combined_low_cost_top1_ambiguity_scale_confirmed`
- `next=D88_COMBINED_LOW_COST_TOP1_AMBIGUITY_STRESS_MAP`
- `best_arm=D86_COMBINED_REPAIR_COST_AWARE_REPLAY`
- `combined_low_cost_plus_top1_ambiguity_breakpoint=0.755`
- `low_cost_pressure_breakpoint=0.750`
- `top1_top2_sufficiency_ambiguity_breakpoint=0.746`
- top1 guard preserved, not weakened, and ablation remained worse.

## Stress axes

1. `COMBINED_LOW_COST_TOP1_AMBIGUITY_EXTENDED_SWEEP`
2. `LOW_COST_PRESSURE_EXTENDED_SWEEP`
3. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP`
4. `COMBINED_LOW_COST_PLUS_OOD`
5. `OOD_SUPPORT_DISTRIBUTION_SHIFT`
6. `JOINT_REQUIRED_NEAR_BOUNDARY`
7. `HARD_CORRELATED_JOINT_RECALL`
8. `HARD_ADVERSARIAL_JOINT_RECALL`
9. `EXTERNAL_REQUIRED_PRESSURE`
10. `INDISTINGUISHABLE_BOUNDARY`
11. `TOP1_GUARD_CORRUPTION_OR_ABLATION`
12. `RUST_INVOCATION_FALLBACK_GUARD`

## Arms

1. `D87_COMBINED_REPAIR_REPLAY`
2. `D87_HIGH_RECALL_VARIANT`
3. `D87_LOW_COST_VARIANT`
4. `D87_BALANCED_VARIANT`
5. `D83_LOW_COST_REPAIR_REPLAY`
6. `D84_STRESS_BASELINE_REPLAY`
7. `TOP1_GUARD_ABLATION_CONTROL`
8. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
9. `LOW_COST_ONLY_CONTROL`
10. `TOP1_AMBIGUITY_ONLY_CONTROL`
11. `OOD_SHIFT_CONTROL`
12. `RANDOM_ROUTER_CONTROL`
13. `NEVER_JOINT_CONTROL`
14. `ALWAYS_JOINT_CONTROL`
15. `CONCRETE_ORACLE_REFERENCE_ONLY`
16. `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`

## Required reports

Artifacts are written under `target/pilot_wave/d88_combined_low_cost_top1_ambiguity_stress_map/` and must include `d87_upstream_manifest.json`, stress-axis reports, breakpoint taxonomy, safety/D68/truth/Rust reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive outcome

D88 is not required to improve support. It must produce a reliable stress map after the D87 combined repair while preserving core D87 metrics, top1 guard status, D68 loss repair, Rust invocation, `fallback_rows=0`, and visible `failed_jobs=[]`.

## Decisions

- Stress map complete and D87 core holds: `decision=combined_low_cost_top1_ambiguity_stress_map_completed`, `next=D89_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`.
- Repairable dominant breakpoint identified: `decision=combined_low_cost_top1_repairable_breakpoint_identified`, `next=D89_TARGETED_BREAKPOINT_REPAIR`.
- Severe broad regression: `decision=combined_low_cost_top1_stress_failure`, `next=D88_REPAIR`.

## Boundary

D88 only maps stress breakpoints after combined low-cost + top1/top2 ambiguity repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
