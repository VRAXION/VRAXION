# D87 Combined Low-Cost Top1 Ambiguity Scale Confirm Contract

## Purpose

D87 scale-confirms the D86 `COMBINED_LOW_COST_TOP1_REPAIR_COST_AWARE` repair while preserving the top1/top2 sufficiency guard, D68 protection, safety margins, and Rust sparse invocation path in controlled symbolic ECF/IPF joint formula discovery. It does not add a new broad architecture mechanism.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D86 commit `7a7be5c5786b0f96671644836b254c6599d7cf46` is present and an ancestor of `HEAD`, restore/rerun D86 artifacts if missing, and write `d86_upstream_manifest.json`. D87 must not silently assume D86 was pushed.

## D86 handoff

D86 confirmed:

- `decision=combined_low_cost_top1_ambiguity_repair_confirmed`
- `next=D87_COMBINED_LOW_COST_TOP1_AMBIGUITY_SCALE_CONFIRM`
- `best_arm=COMBINED_LOW_COST_TOP1_REPAIR_COST_AWARE`
- `combined_low_cost_plus_top1_ambiguity_breakpoint=0.756`
- `low_cost_pressure_breakpoint=0.750`
- `top1_top2_sufficiency_ambiguity_breakpoint=0.746`
- top1 guard preserved, not weakened, and ablation remained worse.

## Tracks

1. `D86_REPLAY`
2. `LARGER_SEED_SCALE`
3. `COMBINED_LOW_COST_TOP1_AMBIGUITY_SWEEP`
4. `LOW_COST_PRESSURE_SWEEP`
5. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP`
6. `TOP1_GUARD_PRESERVATION`
7. `TOP1_GUARD_ABLATION_CONTROL`
8. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
9. `D68_CHEAP_TOP1_REGRESSION_GUARD`
10. `HARD_CORRELATED_JOINT_RECALL`
11. `HARD_ADVERSARIAL_JOINT_RECALL`
12. `OOD_SUPPORT_SHIFT_WATCH`
13. `EXTERNAL_REQUIRED_WATCH`
14. `INDISTINGUISHABLE_ABSTAIN_WATCH`
15. `SAFETY_MARGIN_WATCH`
16. `ORACLE_DISTANCE_FRONTIER`

## Arms

1. `D86_COMBINED_REPAIR_COST_AWARE_REPLAY`
2. `D86_COMBINED_REPAIR_HIGH_RECALL`
3. `D86_COMBINED_REPAIR_LOW_COST`
4. `D86_COMBINED_REPAIR_BALANCED`
5. `D83_LOW_COST_REPAIR_REPLAY`
6. `D84_STRESS_BASELINE_REPLAY`
7. `TOP1_GUARD_ABLATION_CONTROL`
8. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
9. `LOW_COST_ONLY_CONTROL`
10. `TOP1_AMBIGUITY_ONLY_CONTROL`
11. `RANDOM_ROUTER_CONTROL`
12. `NEVER_JOINT_CONTROL`
13. `ALWAYS_JOINT_CONTROL`
14. `CONCRETE_ORACLE_REFERENCE_ONLY`
15. `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`

## Required reports

Artifacts are written under `target/pilot_wave/d87_combined_low_cost_top1_ambiguity_scale_confirm/` and must include `d86_upstream_manifest.json`, scale/sweep/watch reports, top1 guard reports, D68/safety/oracle/support/truth/Rust reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

The scaled D86 repair must satisfy: combined breakpoint at least `0.750`, low-cost breakpoint at least `0.740`, top1 ambiguity non-regression, core recall/safety/support gates, gap reduction at least `0.1500`, D68 preservation `1.0`, no routing failures, top1 guard preserved and not weakened, ablation remains worse, Rust path invoked, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Passing scale confirm: `decision=combined_low_cost_top1_ambiguity_scale_confirmed`, `next=D88_COMBINED_LOW_COST_TOP1_AMBIGUITY_STRESS_MAP`.
- Passing but high support: `decision=combined_low_cost_top1_ambiguity_scale_high_cost`, `next=D87C_COST_REPAIR`.
- Top1 guard weakening: `decision=top1_guard_invariant_violation_under_scale`, `next=D87G_TOP1_GUARD_REPAIR`.
- Safety/routing regression: `decision=combined_low_cost_top1_ambiguity_scale_safety_regression`, `next=D87S_SAFETY_ROUTING_REPAIR`.
- Failed scale confirm: `decision=combined_low_cost_top1_ambiguity_scale_not_confirmed`, `next=D87_REPAIR`.

## Boundary

D87 only scale-confirms the combined low-cost + top1/top2 ambiguity repair while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
