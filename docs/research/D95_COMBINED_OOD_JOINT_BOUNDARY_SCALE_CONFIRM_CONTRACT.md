# D95 Combined OOD Joint Boundary Scale Confirm Contract

## Purpose

D95 scale-confirms the D94 best fair combined OOD + joint-boundary repair arm under larger seed and row coverage while preserving low-cost/OOD gains, the top1/top2 sufficiency guard, D68 loss repair preservation, safety margins, truth-leak controls, and Rust sparse invocation in controlled symbolic ECF/IPF joint formula discovery. D95 is not a new broad repair and must not add broad architecture claims.

## Phase 0 upstream audit

The runner must verify the current branch and `HEAD`, check whether D94 commit `2f440dccf572864304114d5e6b5d0f1284b4b0d0` is present locally, verify `target/pilot_wave/d94_combined_ood_joint_boundary_repair_prototype/`, restore/rerun D94 artifacts if required, validate the D94 handoff, and write `d94_upstream_manifest.json`. D95 must not silently assume D94 was pushed.

## D94 handoff

D94 confirmed:

- `decision=combined_ood_joint_boundary_repair_confirmed`
- `next=D95_COMBINED_OOD_JOINT_BOUNDARY_SCALE_CONFIRM`
- `best_arm=COMBINED_OOD_JOINT_BOUNDARY_REPAIR_COST_AWARE`
- `combined_ood_joint_boundary_breakpoint=0.758`
- `D68_loss_repair_preservation_rate=1.0`
- `routing_failure_rows=0`
- `top1_guard_preserved=true`
- `top1_guard_weakened=false`
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`

## Scale settings

Requested scale is `workers=auto`, `cpu_target=50-75`, `heartbeat_sec=20`, seeds `16001,16002,16003,16004,16005,16006,16007,16008`, and rows `train/test/ood=360` per seed/regime/split. If runtime constraints force a smaller run, the runner must record requested and actual scale plus a reason, execute all proof gates, and avoid overclaiming the decision.

## Tracks

1. `D94_REPLAY`
2. `D94_BEST_ARM_SCALE_CONFIRM`
3. `COMBINED_OOD_JOINT_BOUNDARY_SCALE_SWEEP`
4. `JOINT_REQUIRED_NEAR_BOUNDARY_SCALE_SWEEP`
5. `OOD_SUPPORT_DISTRIBUTION_SHIFT_SCALE_SWEEP`
6. `COMBINED_LOW_COST_PLUS_OOD_SCALE_WATCH`
7. `COMBINED_LOW_COST_OOD_TOP1_AMBIGUITY_SCALE_WATCH`
8. `LOW_COST_PRESSURE_SCALE_WATCH`
9. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SCALE_WATCH`
10. `TOP1_GUARD_PRESERVATION_SCALE`
11. `TOP1_GUARD_ABLATION_CONTROL_SCALE`
12. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL_SCALE`
13. `D68_CHEAP_TOP1_REGRESSION_GUARD_SCALE`
14. `HARD_CORRELATED_JOINT_RECALL_SCALE`
15. `HARD_ADVERSARIAL_JOINT_RECALL_SCALE`
16. `EXTERNAL_REQUIRED_SCALE_WATCH`
17. `INDISTINGUISHABLE_ABSTAIN_SCALE_WATCH`
18. `SAFETY_MARGIN_SCALE_WATCH`
19. `ORACLE_DISTANCE_FRONTIER_SCALE`
20. `SUPPORT_COST_FRONTIER_SCALE`
21. `TRUTH_LEAK_AUDIT_SCALE`
22. `RUST_INVOCATION_SCALE_AUDIT`

## Arms

D95 evaluates D94 replay, D94 cost-aware/balanced/high-recall/low-cost scale variants, joint-only/OOD-only/combined-low-cost-OOD repair-only scale arms, top1 guard ablation and partial-corruption controls, random/never/always controls, concrete oracle reference-only, and truth-leak sentinel reference-only arms.

## Required reports

Artifacts are written under `target/pilot_wave/d95_combined_ood_joint_boundary_scale_confirm/` and must include `d94_upstream_manifest.json`, scale/replay/sweep/watch reports, top1 guard reports, D68/safety/oracle/support/truth/Rust reports, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

The best fair D95 arm must hold `combined_ood_joint_boundary_breakpoint>=0.755`, `min_seed_combined_ood_joint_boundary_breakpoint>=0.752`, preserve combined low-cost + OOD and D94 OOD/joint/top1 watch breakpoints, meet accuracy/min-seed/safety/support/oracle gates, preserve D68 at `1.0`, keep routing failures at `0`, preserve the top1 guard without weakening, keep both ablation and partial-corruption controls worse, pass truth-leak audit, invoke the Rust path, keep `fallback_rows=0`, and expose `failed_jobs=[]`.

## Decisions

- Passing scale confirmation: `decision=combined_ood_joint_boundary_scale_confirmed`, `next=D96_NEXT_BREAKPOINT_OR_GENERALIZATION_PLAN`.
- Average scale holds but min-seed/stress tails fail: `decision=combined_ood_joint_boundary_scale_tail_risk`, `next=D95T_TAIL_RISK_REPAIR`.
- Safety/routing/D68/top1 regression: `decision=combined_ood_joint_boundary_scale_safety_regression`, `next=D95S_SAFETY_ROUTING_REPAIR`.
- Top1 guard weakens: `decision=top1_guard_invariant_violation`, `next=D95G_TOP1_GUARD_REPAIR`.
- Scale does not confirm D94 repair: `decision=combined_ood_joint_boundary_scale_not_confirmed`, `next=D94_REPAIR_REVISIT`.

## Boundary

D95 only scale-confirms the D94 combined OOD + joint-boundary repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
