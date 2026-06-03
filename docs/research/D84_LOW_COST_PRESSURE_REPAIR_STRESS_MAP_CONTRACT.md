# D84 Low-Cost Pressure Repair Stress Map Contract

## Purpose

D84 stress-tests the D83 scale-confirmed low-cost pressure repair and maps the new breakpoints while preserving the top1/top2 sufficiency guard and D68 protection in controlled symbolic ECF/IPF joint formula discovery. It does not add a new broad architecture mechanism.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D83 commit `4cf3d0c32ac5f178baa52dc87ccfaa558bfb43b1` is present and an ancestor of `HEAD`, restore/rerun D83 artifacts if missing, and write `d83_upstream_manifest.json`. D84 must not silently assume D83 was pushed.

## D83 handoff

D83 confirmed:

- `decision=low_cost_pressure_repair_scale_confirmed`
- `next=D84_LOW_COST_PRESSURE_REPAIR_STRESS_MAP`
- `best_arm=D82_LOW_COST_REPAIR_COST_AWARE_REPLAY`
- `low_cost_pressure_breakpoint=0.751`
- top1 guard preserved, not weakened, and ablation remained worse.

D84 is not required to improve support; it must produce a reliable stress map of the repaired low-cost pressure behavior and keep D83 core routing/safety gates visible.

## Stress axes

1. `LOW_COST_PRESSURE_EXTENDED_SWEEP`
2. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY`
3. `OOD_SUPPORT_DISTRIBUTION_SHIFT`
4. `JOINT_REQUIRED_NEAR_BOUNDARY`
5. `HARD_CORRELATED_JOINT_RECALL`
6. `HARD_ADVERSARIAL_JOINT_RECALL`
7. `EXTERNAL_REQUIRED_PRESSURE`
8. `INDISTINGUISHABLE_BOUNDARY`
9. `TOP1_GUARD_CORRUPTION_OR_ABLATION`
10. `COMBINED_LOW_COST_PLUS_OOD`
11. `COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY`
12. `RUST_INVOCATION_FALLBACK_GUARD`

## Arms

1. `D83_LOW_COST_REPAIR_REPLAY`
2. `D83_HIGH_RECALL_VARIANT`
3. `D83_LOW_COST_VARIANT`
4. `D79_INTEGRATED_ROUTER_REPLAY`
5. `TOP1_GUARD_ABLATION_CONTROL`
6. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
7. `LOW_COST_ONLY_CONTROL`
8. `OOD_SHIFT_CONTROL`
9. `RANDOM_ROUTER_CONTROL`
10. `NEVER_JOINT_CONTROL`
11. `ALWAYS_JOINT_CONTROL`
12. `CONCRETE_ORACLE_REFERENCE_ONLY`
13. `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`

## Required reports

Artifacts are written under `target/pilot_wave/d84_low_cost_pressure_repair_stress_map/`:

- `d83_upstream_manifest.json`
- `stress_axis_summary_report.json`
- `low_cost_pressure_extended_sweep_report.json`
- `top1_top2_ambiguity_stress_report.json`
- `ood_support_shift_stress_report.json`
- `joint_required_boundary_stress_report.json`
- `correlated_echo_stress_report.json`
- `adversarial_distractor_stress_report.json`
- `external_required_pressure_report.json`
- `indistinguishable_boundary_report.json`
- `combined_low_cost_ood_report.json`
- `combined_low_cost_top1_ambiguity_report.json`
- `top1_guard_corruption_report.json`
- `breakpoint_taxonomy_report.json`
- `safety_margin_watch_report.json`
- `D68_loss_repair_preservation_report.json`
- `truth_leak_audit_report.json`
- `rust_invocation_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

## Decisions

- Complete stress map with D83 core holding: `decision=low_cost_pressure_repair_stress_map_completed`, `next=D85_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`.
- Specific repairable breakpoint dominates: `decision=low_cost_pressure_repair_repairable_breakpoint_identified`, `next=D85_TARGETED_BREAKPOINT_REPAIR`.
- Severe broad regression: `decision=low_cost_pressure_repair_stress_failure`, `next=D84_REPAIR`.

## Hard gates

No full brain/Raven/AGI/consciousness claims; no fake metrics; no label echo fair oracle; truth hidden from fair arms; oracle/reference arms reference-only; top1 guard must not be weakened; top1 ablation/corruption controls required; D68 loss preservation audited; safety margins audited; Rust arms must invoke the Rust path; `fallback_rows=0`; `failed_jobs` visible; no black-box long run.

## Boundary

D84 only maps stress breakpoints after low-cost pressure repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
