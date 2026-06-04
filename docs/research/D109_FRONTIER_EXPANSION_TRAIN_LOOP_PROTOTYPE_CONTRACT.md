# D109 Frontier Expansion Train Loop Prototype Contract

## Purpose
D109 executes the first controlled symbolic frontier-expansion train-loop prototype after D108 planning. It expands the adapter-only sparse recurrent train-loop from the 12 Lane A anchor families to a controlled 18-family audit set: Lane A anchor, Lane B provisional-normal mixed, Lane C targeted repair-only trig, and the four Lane D safe expansion families accepted by D108.

## Boundary
D109 is adapter-only, checkpointed, rollback-enabled prototype training for controlled symbolic formula-discovery tasks. It does not mutate the symbolic formula solver, dense baseline, protected symbolic router, protected components, base heads, or the frozen 8% sparse mask. It does not include symbolic-sequence or language-like bridge families, does not perform natural-language pretraining, does not train a Gemma-class model, and does not use raw visual Raven tasks.

## Required upstream audit
The runner must validate or restore/rerun the D108 handoff and write `d108_upstream_manifest.json` with commit presence, artifact presence, restore status, replayed D108 decision, replayed next task, D109 readiness, Lane A/B/C/D/E replay status, sparse identity replay, failed jobs, and observed push status.

## Training lanes
- Lane A: 12 D107/D108 core families, batch weight 0.62, preservation anchor.
- Lane B: `MIXED_SYMBOLIC_TRANSFER_FAMILY`, provisional-normal only, batch weight 0.06, guarded stop gates required.
- Lane C: `TRIG_PERIODIC_SYMBOLIC_FAMILY`, targeted repair-only, batch weight 0.04, excluded from healthy claim.
- Lane D: `PIECEWISE_SYMBOLIC_COMPOSITION_FAMILY`, `NESTED_RATIONAL_POLYNOMIAL_FAMILY`, `DISCRETE_RECURRENCE_SYMBOLIC_FAMILY`, and `MULTI_STEP_RULE_CHAIN_FAMILY`, batch weight 0.24.
- Audit sentinels: batch weight 0.04.

## Exclusions
D109 must not train `SYMBOLIC_SEQUENCE_ROUTING_FAMILY`, `LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY`, `HELDOUT_DEPTH_6_COMPOSITION_FAMILY`, or `ADVERSARIAL_FAMILY_MIX_TRANSFER_FAMILY`; these are audited for noninterference only.

## Success decision
D109 may return `d109_frontier_expansion_train_loop_prototype_confirmed` only when D108 replay is valid, scale is not reduced, sparse identity is preserved at 8% light pressure, adapter-only updates execute, Lane A is preserved, Lane B provisional-normal gates pass without stop trigger, Lane C repair signal remains positive without healthy-claim inclusion, Lane D expansion passes safety gates, held-out rejected families do not interfere, leak sentinels collapse, Rust path is invoked, fallback rows are zero, failed jobs are empty, and reports are consistent.

## Next task on success
`D110_FRONTIER_EXPANSION_SCALE_CONFIRM_OR_SYMBOLIC_SEQUENCE_BRIDGE_PLAN`.
