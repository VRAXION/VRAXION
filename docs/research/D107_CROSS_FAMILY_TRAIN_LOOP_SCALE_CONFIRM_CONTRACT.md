# D107 Cross-Family Train Loop Scale Confirm Contract

## Purpose
D107 scale-confirms the D106 adapter-only controlled cross-family train-loop prototype for controlled symbolic formula-discovery tasks. It is not a new sparsity run or a new training strategy.

## Boundary
D107 preserves the symbolic formula solver, dense baseline, protected symbolic router, protected components, and the frozen 8% sparse mask. It does not perform natural-language pretraining, does not train a Gemma-class model, and does not use raw visual Raven tasks.

## Upstream audit
The run must validate or restore/rerun D106 and write `d106_upstream_manifest.json` with commit presence, artifact presence, replayed D106 decision, D107 readiness, Lane A/B/C replay status, sparse identity replay, failed jobs, and observed push status.

## Scale settings
- Main seeds: `28001..28012`; train/test/ood rows: 640 per seed.
- Family seeds: `28101..28110`; family rows: 560 per seed/family/regime/split.
- Adapter train seeds: `28201..28206`; adapter rows: 420 per seed/family/regime/split.
- Lane B seeds: `28301..28303`; rows: 400; max steps: 90; batch weight: 0.07.
- Lane C seeds: `28401..28403`; rows: 400; max steps: 90; repair-probe weight: 0.05.
- Stress seeds: `28501..28506`; rows: 760 per seed/regime/split.

## Lanes
Lane A contains the 12 D106 passing controlled symbolic families. Lane B is `MIXED_SYMBOLIC_TRANSFER_FAMILY` and remains guarded, not an ordinary passing family. Lane C is `TRIG_PERIODIC_SYMBOLIC_FAMILY`, repair-only, and excluded from the healthy training claim.

## Frozen and trainable surfaces
Frozen surfaces include the symbolic solver, protected symbolic router, dense baseline, 8% sparse mask, protected components, base halting/route heads, recurrent base weights, convergence/halting logic, and Rust sparse invocation path. Trainable surfaces are restricted to `route_head_adapter`, `halting_head_adapter`, `recurrent_state_adapter`, and `calibration_scalar_adapter`.

## Positive decision
D107 may return `d107_cross_family_train_loop_scale_confirmed` only if D106 replay is valid, full scale is recorded without reduction, all families and stress modes execute, sparse identity is preserved, adapter-only training executes, Lane A passes all scale gates, Lane B guarded probing remains beneficial without Lane A interference, Lane C repair signal remains positive without contaminating the healthy claim, integrated evaluation is non-regressive, leak/shortcut sentinels collapse to control level, deterministic replay passes, Rust path is invoked, fallback rows are zero, and failed jobs are empty.

## Next task on success
`D108_CROSS_FAMILY_TRAIN_LOOP_FRONTIER_EXPANSION_PLAN`.
