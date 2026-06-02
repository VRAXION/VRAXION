# D70 Support Cost Oracle Gap Reduction Contract

D70 reduces the remaining support-cost oracle gap after D69 scale-confirmed D68C support-cost optimization. It is a controlled symbolic ECF/IPF support-routing milestone, not a new broad architecture milestone.

Source-of-truth upstream is D69: `decision=support_cost_optimization_scale_confirmed`, `verdict=D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRMED`, `next=D70_SUPPORT_COST_ORACLE_GAP_REDUCTION`, scaled arm `D68C_COST_OPTIMIZED_ROUTER`, support `7.025`, support saved vs D68R `0.6545`, and remaining concrete-oracle gap `0.7055`. The runner verifies whether the D69 commit/artifacts are present and bootstraps D69 if generated artifacts are absent.

D70 must reduce support only where concrete counter-action correctness, D68 loss repair preservation, joint/external recall, and safety margins remain within gate. It must not repeat D68's cheap-top1 failure mode.

Tracks: D69 replay, oracle gap taxonomy, safe deescalation, joint-counter-required rows, top1/top2-sufficient rows, external-test-required rows, indistinguishable abstain rows, low-cost stress, OOD oracle gap, and safety margin watch.

Arms: D69 replay, D68C high-recall variant, D68C low-cost variant, concrete oracle reference, safe joint deescalation, low-risk top1 escalation only, postcheck-before-joint counter, external-first when available, oracle-gap targeted router, cost-optimized router v2, safety-margin preserving router, always/never/random controls, and a truth-leak sentinel reference-only arm.

Positive gate: the best fair D70 arm must keep exact >= 0.9990, correlated/adversarial/external >= 0.995, false confidence <= 0.01, abstain >= 0.99, wrong concrete counter <= 0.001, weak top1/top2 failure <= 0.001, joint-counter recall >= 0.99, external recall >= 0.995, D68 loss repair preservation = 1.0, support saved vs D69 >= 0.20 preferred, distance to concrete oracle <= 0.50 preferred, min_seed_exact >= 0.997, fallback_rows = 0, and failed_jobs = [].

Artifacts are written only under `target/pilot_wave/d70_support_cost_oracle_gap_reduction/` and include upstream manifest, oracle-gap taxonomy/frontier, safe deescalation, routing/recall/safety reports, truth-leak audit, Rust invocation provenance, aggregate metrics, decision, summary, and markdown report.

Boundary: D70 only reduces support-cost oracle gap after D69 scale-confirmed support-cost optimization in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
