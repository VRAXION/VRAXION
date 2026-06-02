# D72 Support Cost Oracle Gap Bound Analysis Contract

D72 analyzes the remaining support-cost oracle gap after D71 scale-confirmed D70's oracle-gap reduction.

Upstream source of truth: D71 `support_cost_oracle_gap_scale_confirmed`, scaled arm `D70_ORACLE_GAP_TARGETED_REPLAY`, support `6.812`, remaining oracle gap `0.4925`, and next `D72_SUPPORT_COST_ORACLE_GAP_BOUND_ANALYSIS`.

D72 is an analytical bound/decomposition milestone, not a new support-cutting mechanism. It must not blindly reduce support or regress into D68's cheap top1/top2 routing failure.

Tracks: `D71_REPLAY`, `ORACLE_GAP_DECOMPOSITION`, `JOINT_RECALL_BOUND`, `EXTERNAL_RECALL_BOUND`, `FALSE_CONFIDENCE_BOUND`, `ABSTAIN_BOUND`, `LOW_COST_VARIANT_HARM_AUDIT`, `SAFE_DEESCALATION_FRONTIER`, `OOD_BOUND_ANALYSIS`, `MIN_SEED_BOUND_ANALYSIS`.

Arms: `D71_D70_REPLAY`, `D70_LOW_COST_VARIANT_REPLAY`, `CONCRETE_ORACLE_REFERENCE_ONLY`, `JOINT_RECALL_RELAXED_VARIANT`, `EXTERNAL_RECALL_RELAXED_VARIANT`, `FALSE_CONFIDENCE_RELAXED_VARIANT`, `ABSTAIN_RELAXED_VARIANT`, `SAFETY_PRESERVING_LOW_COST_VARIANT`, `ROUTING_PRESERVING_LOW_COST_VARIANT`, `ORACLE_GAP_BOUND_ESTIMATOR`, `ALWAYS_COUNTER_CONTROL`, `NEVER_COUNTER_CONTROL`, `RANDOM_COUNTER_CONTROL`, `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`.

Required analysis separates irreducible safety/routing cost, external-test cost, joint-recall cost, false-confidence cost, abstain cost, conservative margin, and still-reducible cost.

Artifacts live under `target/pilot_wave/d72_support_cost_oracle_gap_bound_analysis/` and include upstream, decomposition, cost-bound, harm, frontier, min-seed, truth-leak, Rust, aggregate, decision, summary, and Markdown reports.

Boundary: D72 only analyzes the remaining support-cost oracle gap after D71 in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
