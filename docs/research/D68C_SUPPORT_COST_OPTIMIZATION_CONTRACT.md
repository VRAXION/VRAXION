# D68C Support Cost Optimization Contract

D68C optimizes support cost after D68R repaired concrete counter-action routing. It must preserve D68R/D67-level exact joint accuracy and concrete counter correctness while moving support use toward the concrete-oracle reference.

Required inputs are the checked-in D68A/D68R docs and, when available, restored D67/D68/D68A/D68R smoke artifacts from the 2026-06-01 handoff package. If those generated artifacts are absent, the runner restores the compact handoff artifact bundle and writes `artifact_restore_report.json`.

Tracked arms include D67, D68, D68R, concrete oracle reference, D68C cost-optimized fair routers, escalation/defer controls, always/never/random controls, and a truth-leak sentinel reference-only arm. Fair arms must not use truth labels, support-regime labels, row/sample lookup, Python `hash()`, fixed fake accuracies, or `hit=random.random()<p` sampling.

Positive gate: a best fair D68C arm must keep exact joint accuracy at least 0.9990, correlated/adversarial/external safety at least 0.995, false confidence at most 0.01, indistinguishable abstention at least 0.99, wrong concrete counter and weak top1/top2 failures at most 0.001, joint-counter recall at least 0.99, D68 loss repair preservation at 1.0, fallback rows at zero, failed jobs empty, Rust-path provenance present, and average support below 7.6795 with at least 0.30 support saved versus D68R preferred.

Artifacts are written only under `target/pilot_wave/d68c_support_cost_optimization/` and include queue/progress instrumentation, support-cost frontier reports, routing preservation reports, truth-leak audit, Rust invocation provenance, aggregate metrics, decision, summary, and a markdown report.

Boundary: D68C only optimizes support cost after concrete counter-action routing repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
