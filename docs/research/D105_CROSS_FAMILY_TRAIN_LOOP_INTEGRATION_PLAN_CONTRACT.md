# D105 Cross-Family Train-Loop Integration Plan Contract

## Purpose

D105 creates the cross-family train-loop integration plan for the D104 sparse recurrent symbolic routing core. It is a planning and integration-readiness milestone only: it defines the D106 objective, batch mix, curriculum, freeze policy, sparse-mask policy, rollback gates, evaluation harness, shortcut sentinels, and non-destructive dry-run checks needed before a cross-family train-loop prototype can begin.

## Boundary

D105 is controlled symbolic train-loop integration planning only. The formula solver remains symbolic. D105 does not perform full training, does not increase sparsity, does not destructively modify the confirmed sparse recurrent core, does not use raw visual Raven tasks, does not use natural-language pretraining, and does not train a Gemma-class model. It makes no full VRAXION brain, raw visual Raven solved, Raven solved, AGI, consciousness, DNA/genome, production-readiness, or architecture-superiority claim.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, verify whether D104 commit `8ebc82f77ac013f4d41bbbbddb905044160eed98` is present locally, verify `target/pilot_wave/d104_sparse_recurrent_generalization_and_compression_frontier_map/`, validate D104 decision `d104_sparse_recurrent_generalization_frontier_mapped`, next `D105_CROSS_FAMILY_TRAIN_LOOP_INTEGRATION_PLAN`, `d105_ready=true`, 14 families with 12 pass, 1 partial, and 1 fail, worst family `TRIG_PERIODIC_SYMBOLIC_FAMILY`, partial family `MIXED_SYMBOLIC_TRANSFER_FAMILY`, final sparse percent `8`, final pressure `light`, protected components locked, D68 and top1 guards preserved, no fallback rows, and no failed jobs. Missing or invalid upstream artifacts must trigger explicit restore/rerun, and `d104_upstream_manifest.json` must record the handoff and pushed status without silently assuming D104 was pushed.

## Scale, families, and dry-run settings

Requested main scale uses seeds `26001,26002,26003,26004,26005,26006,26007,26008` with rows `train/test/ood=520` per seed/regime/split. The family-planning extension uses seeds `26101,26102,26103,26104,26105,26106,26107,26108` with rows `train/test/ood=480` per seed/family/regime/split across the 14 D104 controlled symbolic families. The dry-run integration extension uses seeds `26201,26202,26203,26204` with rows `train/test/ood=360` per seed/family/regime/split. The stress extension uses seeds `26301,26302,26303,26304`, stress rows `640`, and train-loop, family, trig, mixed, guard, D68, mask, checkpoint, and sparse-cost frontier tails. Scale reduction must be recorded and cannot overclaim.

## Family lane policy

Lane A is the passing-family integration lane and includes the 12 D104 passing controlled symbolic families for the shared D106 train-loop plan. Lane B is the `MIXED_SYMBOLIC_TRANSFER_FAMILY` guarded lane; it remains partial/guarded and must not contaminate Lane A metrics. Lane C is the `TRIG_PERIODIC_SYMBOLIC_FAMILY` repair lane; it is a known failing frontier and must be excluded from healthy training claims while receiving a concrete loop-utility and mask-stability repair plan.

## Train-loop constraints

D105 must not start full training and must not make permanent parameter updates to the confirmed sparse candidate. Shadow-gradient or simulated update analysis is allowed only as a non-destructive dry-run. Protected components and the 8% sparse mask are frozen by default. D105 must explicitly define D106 trainable components, frozen components, rollback checkpoints, stop gates, family weighting, trig/mixed lane handling, objective schema, route-head policy, halting-head policy, recurrent-state policy, guard/D68/loop/halting preservation losses, leakage audits, and D106 contract recommendations.

## Required reports

Artifacts are written under `target/pilot_wave/d105_cross_family_train_loop_integration_plan/` and must include `d104_upstream_manifest.json`, all D105 lane, objective, batch/curriculum, freeze, update-policy, preservation-loss, trig/mixed, stop/rollback, D106 harness/checkpoint/metric, dry-run, shortcut, sentinel, split, overfit, negative-control, truth/oracle, Rust, report-schema, deterministic-replay reports, `d105_d106_contract_recommendation_report.md`, `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`.

## Positive gate

D105 passes only if D104 handoff/replay is valid, requested and actual scale are recorded with `scale_reduced=false`, all required families and stress modes execute, sparse identity is preserved as 8% light pressure, protected components and sparse mask are frozen by default, all train-loop planning elements are defined, Lane A is integration-ready with risk below thresholds, Lane B is guarded and ready for a D106 guarded probe, Lane C is explicitly excluded from healthy training and ready for a D106 repair probe, non-destructive dry-run risk is below thresholds, leak/shortcut sentinels collapse, controls remain worse, deterministic replay/schema/metric crosscheck pass, Rust path is invoked, `fallback_rows=0`, and `failed_jobs=[]`.

## Decisions

- Healthy integration plan: `decision=d105_cross_family_train_loop_integration_plan_ready`, `next=D106_CROSS_FAMILY_TRAIN_LOOP_PROTOTYPE`.
- Lane A not ready: `decision=d105_passing_family_integration_not_ready`, `next=D105A_PASSING_FAMILY_INTEGRATION_REPAIR`.
- Mixed guarded lane not ready: `decision=d105_mixed_family_guarded_lane_not_ready`, `next=D105M_MIXED_FAMILY_MARGIN_REPAIR`.
- Trig repair lane not ready: `decision=d105_trig_periodic_repair_lane_not_ready`, `next=D105T_TRIG_PERIODIC_REPAIR_PLAN`.
- Dry-run risk detected: `decision=d105_train_loop_dry_run_risk_detected`, `next=D105R_DRY_RUN_RISK_REPAIR`.
- Shortcut or leak detected: `decision=d105_shortcut_or_leak_detected`, `next=D105L_SHORTCUT_LEAK_REPAIR`.
- Sparse identity/protection violation: `decision=d105_sparse_identity_or_protection_violation`, `next=D105P_SPARSE_IDENTITY_REPAIR`.
- Rust fallback: `decision=d105_rust_fallback_detected`, `next=D105R_RUST_PATH_REPAIR`.
- Reporting inconsistency: `decision=d105_invalid_metric_or_report_inconsistency`, `next=D105_REPORTING_REPAIR`.
- Incomplete run: `decision=d105_invalid_or_incomplete_run`, `next=D105_RETRY_WITH_FULL_AUDIT`.
