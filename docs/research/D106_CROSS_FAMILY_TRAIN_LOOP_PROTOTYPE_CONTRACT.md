# D106 Cross-Family Train-Loop Prototype Contract

## Purpose

D106 performs the first controlled cross-family train-loop prototype for the D105-planned sparse recurrent symbolic routing core. Unlike D105, D106 may execute limited, checkpointed, rollback-enabled adapter-only updates on a copy of the confirmed 8% light-pressure sparse recurrent core. It must prove whether Lane A can train safely, whether Lane B can be probed under guarded inclusion, and whether Lane C can show a repair signal without contaminating the healthy training claim.

## Boundary

D106 is controlled symbolic cross-family train-loop prototyping only. The formula solver remains symbolic. D106 uses adapter-only prototype updates and does not train the full core destructively, unfreeze the sparse mask, mutate protected components, perform natural-language pretraining, train a Gemma-class model, use raw visual Raven tasks, or claim full VRAXION brain, raw visual Raven solved, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

## Phase 0 upstream audit

The runner must verify branch/HEAD, check D105 commit `263f176685771973044ac4fdc8c542bb9972c4a0`, verify `target/pilot_wave/d105_cross_family_train_loop_integration_plan/`, validate decision `d105_cross_family_train_loop_integration_plan_ready`, next `D106_CROSS_FAMILY_TRAIN_LOOP_PROTOTYPE`, `d106_ready=true`, Lane A ready, Lane B guarded-ready, Lane C repair-ready, final sparse pct `8`, pressure `light`, protected components and sparse mask frozen by default, non-destructive D105 dry-run, zero fallback rows, and zero failed jobs. Missing or invalid D105 handoff must trigger explicit restore/rerun and `d105_upstream_manifest.json` must record the handoff and pushed status.

## Scale and training settings

Requested main scale uses seeds `27001,27002,27003,27004,27005,27006,27007,27008` with rows `train/test/ood=520`. Family training uses seeds `27101,27102,27103,27104,27105,27106,27107,27108` with rows `480` per seed/family/regime/split. Adapter training uses seeds `27201,27202,27203,27204`, rows `360`, max epochs `3`, max steps per epoch `120`, early-stop patience `1`, small deterministic adapter learning rate, light weight decay, gradient clipping, and deterministic update order. Lane B and Lane C each use guarded/repair seeds with rows `320`; Lane C remains excluded from healthy training. Stress extension uses seeds `27501,27502,27503,27504`, stress rows `640`, and 31 D106 train-loop, lane, guard, D68, mask, checkpoint, adapter, and shortcut stress modes.

## Lane policy

Lane A contains the 12 D104 passing controlled symbolic families and is the only healthy train-loop integration lane. Lane B contains `MIXED_SYMBOLIC_TRANSFER_FAMILY` as a guarded probe with batch weight `0.07`; it is not a normal passing family. Lane C contains `TRIG_PERIODIC_SYMBOLIC_FAMILY` as a repair-only probe with repair weight `0.05`; it is excluded from the healthy training claim and may only contribute targeted repair-signal evidence.

## Objective, frozen components, and adapters

The train-loop objective is `route_distillation_plus_guard_D68_loop_halting_preservation`. Loss components include route distillation, guard preservation, D68 preservation, loop utility preservation, halting/convergence preservation, calibration stability, sparse mask drift penalty, protected component change penalty, Lane A forgetting penalty, Lane B margin penalty, and Lane C trig loop/mask/phase/harmonic repair losses. Frozen surfaces include the symbolic formula solver, protected symbolic router, dense baseline, 8% sparse mask, protected components, base top1/top2, OOD, boundary, joint evidence, recurrent update, halting, route logits, threshold, and Rust invocation paths. Only `route_head_adapter`, `halting_head_adapter`, `recurrent_state_adapter`, and `calibration_scalar_adapter` may update.

## Checkpoints and rollback

Required checkpoints are `pre_d106`, `post_lane_a_epoch1`, `post_lane_a_epoch2`, `post_lane_a_epoch3_if_executed`, `post_lane_b_guarded_probe`, `post_lane_c_repair_probe`, and `final_candidate_or_rollback`. Rollback must trigger on top1/D68 regression, sparse mask drift above `0.002`, protected component modification, loop/halting/convergence failure, routing failures, false confidence over gate, Lane A forgetting, Lane B stop failure, Lane C interference, shortcut/leak, Rust fallback, failed jobs, or report/metric inconsistency.

## Required reports and decisions

Artifacts are written under `target/pilot_wave/d106_cross_family_train_loop_prototype/` and must include `d105_upstream_manifest.json`, all D106 scale, identity, baseline, adapter/frozen/mask/protected, objective/loss, checkpoint/rollback, Lane A/B/C, integrated-eval, post-train, guard/D68/halting/loop/calibration/mask/protected/Rust, sentinel, split, overfit, negative-control, truth/oracle, schema, deterministic-replay reports, plus `aggregate_metrics.json`, `decision.json`, `summary.json`, and `report.md`. Healthy decision is `d106_cross_family_train_loop_prototype_confirmed` with next `D107_CROSS_FAMILY_TRAIN_LOOP_SCALE_CONFIRM`; failures must route to the specific D106 repair decision without overclaiming.
