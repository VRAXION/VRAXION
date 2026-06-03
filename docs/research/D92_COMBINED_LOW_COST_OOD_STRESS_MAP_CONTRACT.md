# D92 Combined Low-Cost OOD Stress Map Contract

## Purpose

D92 stress-tests the D91 scale-confirmed combined low-cost + OOD repair and maps the next dominant breakpoint while preserving the top1/top2 sufficiency guard, D68 loss repair, safety margins, truth-leak guards, and Rust sparse invocation path in controlled symbolic ECF/IPF joint formula discovery. D92 does not add a new broad architecture mechanism and is not required to improve support.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D91 commit `84ff947478c1e9b2379d656e74a8e2b0498fa373` is present and an ancestor of `HEAD`, restore/rerun D91 artifacts if missing, and write `d91_upstream_manifest.json`. D92 must not silently assume D91 was pushed.

## D91 handoff

D91 confirmed:

- `decision=combined_low_cost_ood_scale_confirmed`
- `next=D92_COMBINED_LOW_COST_OOD_STRESS_MAP`
- `best_arm=D90_COMBINED_LOW_COST_OOD_REPAIR_REPLAY`
- `combined_low_cost_plus_ood_breakpoint=0.763`
- `ood_support_distribution_shift_breakpoint=0.760`
- `low_cost_pressure_breakpoint=0.749`
- `combined_low_cost_plus_top1_ambiguity_breakpoint=0.754`
- `top1_top2_sufficiency_ambiguity_breakpoint=0.746`
- top1 guard preserved, not weakened, and ablation remained worse.

## Stress axes

1. `COMBINED_LOW_COST_OOD_EXTENDED_SWEEP`
2. `OOD_SUPPORT_DISTRIBUTION_SHIFT_SWEEP`
3. `LOW_COST_PRESSURE_EXTENDED_SWEEP`
4. `COMBINED_LOW_COST_TOP1_AMBIGUITY_WATCH`
5. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP`
6. `COMBINED_LOW_COST_OOD_TOP1_AMBIGUITY`
7. `COMBINED_OOD_JOINT_BOUNDARY`
8. `JOINT_REQUIRED_NEAR_BOUNDARY`
9. `HARD_CORRELATED_JOINT_RECALL`
10. `HARD_ADVERSARIAL_JOINT_RECALL`
11. `EXTERNAL_REQUIRED_PRESSURE`
12. `INDISTINGUISHABLE_BOUNDARY`
13. `TOP1_GUARD_CORRUPTION_OR_ABLATION`
14. `RUST_INVOCATION_FALLBACK_GUARD`

## Arms

D92 evaluates D91 replay and variant arms, D87/D88 replay controls, OOD/low-cost/top1 controls, combined low-cost/top1 control, top1 guard ablation/corruption controls, random/never/always controls, concrete oracle reference-only, and truth-leak sentinel reference-only arms.

## Required reports

Artifacts are written under `target/pilot_wave/d92_combined_low_cost_ood_stress_map/` and must include `d91_upstream_manifest.json`, stress-axis reports, breakpoint taxonomy, top1 guard corruption/ablation report, D68/safety/truth/Rust reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive outcome

D92 should produce a reliable stress map after D91 scale confirmation. D91 core must hold under standard stress: exact/correlated/adversarial/external metrics remain above gate, D68 preservation remains `1.0`, routing failures remain `0`, the top1 guard remains preserved and not weakened, Rust path is invoked, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Complete stress map with D91 core holding: `decision=combined_low_cost_ood_stress_map_completed`, `next=D93_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`.
- Specific repairable breakpoint dominates: `decision=combined_low_cost_ood_repairable_breakpoint_identified`, `next=D93_TARGETED_BREAKPOINT_REPAIR`.
- Severe broad regression: `decision=combined_low_cost_ood_stress_failure`, `next=D92_REPAIR`.

## Boundary

D92 only maps stress breakpoints after combined low-cost + OOD repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
