# D69 Support Cost Optimization Scale Confirm Contract

D69 scale-confirms D68C support-cost optimization after D68R concrete counter-action routing repair. It must not introduce a new architecture claim; it only checks whether the D68C cost saving remains stable across a larger seed/OOD scale while preserving routing correctness and safety.

Source-of-truth upstream is D68C: `decision=support_cost_optimization_confirmed`, `verdict=D68C_SUPPORT_COST_OPTIMIZATION_CONFIRMED`, `next=D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRM`, and `best_fair_arm=D68C_COST_OPTIMIZED_ROUTER`. If D68C generated artifacts are absent, the D69 runner bootstraps the D68C smoke artifacts first and records that in a bootstrap report.

Scale tracks: D68C replay, larger seed scale, OOD routing, hard correlated joint recall, hard adversarial joint recall, external-test-required behavior, indistinguishable abstain behavior, oracle-distance frontier, safety regression audit, and support-cost frontier.

Arms: D67 replay, D68 threshold replay, D68R concrete router replay, D68C cost-optimized router, high-recall D68C variant, low-cost D68C variant, concrete oracle reference, always/never/random controls, and a truth-leak sentinel reference-only arm. Fair arms must not use truth labels, label echo, support-regime labels, row/sample lookup, Python `hash()`, fixed fake hit sampling, or oracle reference information.

Positive gate for the scaled D68C arm: exact >= 0.9990, correlated/adversarial/external >= 0.995, false confidence <= 0.01, abstain >= 0.99, wrong concrete counter <= 0.001, weak top1/top2 failure <= 0.001, joint counter recall >= 0.99, D68 loss repair preservation = 1.0, support_saved_vs_D68R >= 0.30, distance_to_oracle <= 0.80, min_seed_exact >= 0.997, fallback_rows = 0, and failed_jobs = [].

Artifacts are written only under `target/pilot_wave/d69_support_cost_optimization_scale_confirm/` and include queue/progress instrumentation, support/routing/safety reports, truth-leak audit, Rust provenance, aggregate metrics, decision, summary, and report markdown.

Boundary: D69 only scale-confirms D68C support-cost optimization after concrete counter-action routing repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
