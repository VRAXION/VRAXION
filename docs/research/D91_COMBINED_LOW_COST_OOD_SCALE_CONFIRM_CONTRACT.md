# D91 Combined Low-Cost OOD Scale Confirm Contract

## Purpose

D91 scale-confirms the D90 `COMBINED_LOW_COST_OOD_REPAIR_COST_AWARE` repair while preserving the top1/top2 sufficiency guard, D68 loss repair, safety margins, truth-leak guards, and Rust sparse invocation path in controlled symbolic ECF/IPF joint formula discovery. D91 does not add a new broad architecture mechanism.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D90 commit `d84508c5fa8a26ac323db094058d76f4c318de4a` is present and an ancestor of `HEAD`, restore/rerun D90 artifacts if missing, and write `d90_upstream_manifest.json`. D91 must not silently assume D90 was pushed.

## D90 handoff

D90 confirmed:

- `decision=combined_low_cost_ood_repair_confirmed`
- `next=D91_COMBINED_LOW_COST_OOD_SCALE_CONFIRM`
- `best_arm=COMBINED_LOW_COST_OOD_REPAIR_COST_AWARE`
- `combined_low_cost_plus_ood_breakpoint=0.764`
- `ood_support_distribution_shift_breakpoint=0.760`
- `low_cost_pressure_breakpoint=0.749`
- `combined_low_cost_plus_top1_ambiguity_breakpoint=0.754`
- `top1_top2_sufficiency_ambiguity_breakpoint=0.746`
- top1 guard preserved, not weakened, and ablation remained worse.

## Tracks

1. `D90_REPLAY`
2. `LARGER_SEED_SCALE`
3. `COMBINED_LOW_COST_OOD_SWEEP`
4. `OOD_SUPPORT_DISTRIBUTION_SHIFT_SWEEP`
5. `LOW_COST_PRESSURE_SWEEP`
6. `COMBINED_LOW_COST_TOP1_AMBIGUITY_WATCH`
7. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY_WATCH`
8. `TOP1_GUARD_PRESERVATION`
9. `TOP1_GUARD_ABLATION_CONTROL`
10. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
11. `D68_CHEAP_TOP1_REGRESSION_GUARD`
12. `HARD_CORRELATED_JOINT_RECALL`
13. `HARD_ADVERSARIAL_JOINT_RECALL`
14. `EXTERNAL_REQUIRED_WATCH`
15. `INDISTINGUISHABLE_ABSTAIN_WATCH`
16. `SAFETY_MARGIN_WATCH`
17. `ORACLE_DISTANCE_FRONTIER`

## Arms

D91 evaluates D90 repair replay/variant arms, D87/D88 replay controls, OOD/low-cost/top1 repair-only controls, low-cost/OOD controls, top1 guard ablation/corruption controls, random/never/always controls, concrete oracle reference-only, and truth-leak sentinel reference-only arms.

## Required reports

Artifacts are written under `target/pilot_wave/d91_combined_low_cost_ood_scale_confirm/` and must include `d90_upstream_manifest.json`, scale/sweep/watch reports, top1 guard reports, D68/safety/oracle/support/truth/Rust reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

The scaled D90 repair must satisfy: combined low-cost + OOD breakpoint at least `0.760`, OOD support shift non-regression versus D90, combined low-cost + top1 ambiguity at least `0.750`, low-cost pressure at least `0.740`, core recall/safety/support thresholds, gap reduction at least `0.1500`, D68 preservation `1.0`, no routing failures, top1 guard preserved and not weakened, ablation remains worse, Rust path invoked, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Passing scale confirm: `decision=combined_low_cost_ood_scale_confirmed`, `next=D92_COMBINED_LOW_COST_OOD_STRESS_MAP`.
- Passing but high support: `decision=combined_low_cost_ood_scale_high_cost`, `next=D91C_COST_REPAIR`.
- OOD/routing/safety regression: `decision=combined_low_cost_ood_scale_safety_regression`, `next=D91S_SAFETY_ROUTING_REPAIR`.
- Top1 guard weakening: `decision=top1_guard_invariant_violation_under_ood_scale`, `next=D91G_TOP1_GUARD_REPAIR`.
- Failed scale confirm: `decision=combined_low_cost_ood_scale_not_confirmed`, `next=D91_REPAIR`.

## Boundary

D91 only scale-confirms the combined low-cost + OOD support distribution repair while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
