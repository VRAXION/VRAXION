# D71 Support Cost Oracle Gap Scale Confirm Contract

D71 scale-confirms the D70 oracle-gap reduction after D69/D70 support-cost repairs.

Upstream source of truth: D70 `support_cost_oracle_gap_reduction_confirmed`, best arm `ORACLE_GAP_TARGETED_ROUTER`, support `6.8050`, remaining oracle gap `0.4855`, and next `D71_SUPPORT_COST_ORACLE_GAP_SCALE_CONFIRM`.

D71 is a scale confirmation only: it must not add a new broad architecture mechanism or regress into D68's cheap `REQUEST_COUNTER_TOP1_TOP2` failure.

Tracks: `D70_REPLAY`, `LARGER_SEED_SCALE`, `OOD_ORACLE_GAP`, `HARD_CORRELATED_JOINT_RECALL`, `HARD_ADVERSARIAL_JOINT_RECALL`, `EXTERNAL_TEST_REQUIRED`, `INDISTINGUISHABLE_ABSTAIN`, `SAFETY_MARGIN_WATCH`, `ORACLE_DISTANCE_FRONTIER`.

Arms: `D69_D68C_REPLAY`, `D70_ORACLE_GAP_TARGETED_REPLAY`, `D70_HIGH_RECALL_VARIANT`, `D70_LOW_COST_VARIANT`, `CONCRETE_ORACLE_REFERENCE_ONLY`, `ALWAYS_COUNTER_CONTROL`, `NEVER_COUNTER_CONTROL`, `RANDOM_COUNTER_CONTROL`, `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`.

Positive gate: the scaled D70 arm must keep exact accuracy at least `0.9990`, correlated/adversarial/external accuracy at least `0.995`, false confidence at most `0.01`, indistinguishable abstain at least `0.99`, wrong concrete counter and weak top1/top2 path failure at most `0.001`, joint recall at least `0.99`, external recall at least `0.995`, full D68 loss repair preservation, support saved vs D69 at least `0.15`, oracle distance at most `0.55`, min-seed exact at least `0.997`, `fallback_rows=0`, and `failed_jobs=[]`.

Artifacts live under `target/pilot_wave/d71_support_cost_oracle_gap_scale_confirm/` and include upstream, scale, frontier, routing, recall, safety, truth-leak, Rust, aggregate, decision, summary, and Markdown reports.

Boundary: D71 only scale-confirms D70 support-cost oracle-gap reduction in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
