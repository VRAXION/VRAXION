# Changelog

This changelog is narrowed to the current E113 evidence anchor. Full historical beta, probe, Python SDK, legacy Rust, and research-output history is preserved in git history and archive tags.

## 2026-06-14 - E113 FineWeb Light Stress Hard Mutation Recycle

- Current evidence anchor: `05415f5b06a43440742715ea93a5e2ec97632f21`
  (`Add E113 FineWeb light stress recycle probe`).
- Latest GitHub release remains `v5.0.0-e79.0`.
- E113 stress-tests the E112 CoreMemoryCandidate pool on the local
  FineWeb-Edu 100k seed pack.
- Baseline result: 2,624 hard negatives across 88 operators.
- Selected recycled variants: 0 hard negatives, 0 neutral waste, 3,461,003
  selected calls/positives, and 136 recycled operators.
- Boundary: FineWeb light stress/recycle evidence only; not PermaCore,
  TrueGolden, public API readiness, final training, or open-domain assistant
  readiness.

## 2026-06-14 - E112 Gold-To-CoreMemoryCandidate Prune-Heavy Probation

- Evidence head at that point: `9de33241f637fed08451cbb054a2f70e07630ba4`
  (`Add E112 gold to core prune wave`).
- Latest GitHub release remains `v5.0.0-e79.0`.
- E112 evaluates the scoped Gold pool under prune-heavy CoreMemoryCandidate
  probation using `minimal_core_prune`, `balanced_core_prune`,
  `deep_core_prune`, and `sibling_challenger`.
- Result: 136 candidates, 136 CoreMemoryCandidate qualifications, 0 RedFlags,
  0 hard negatives, 0 wrong-scope calls, 0 false commits, 0 unsupported
  answers, and deterministic replay pass.
- Post-wave rank summary after E110 + E111 + E112: 136
  CoreMemoryCandidate, 0 Gold, 0 Silver, 0 Bronze, 0 DiamondCandidate,
  0 RedFlag, 3 Deprecated.
- Boundary: scoped CoreMemoryCandidate probation evidence only; not PermaCore,
  TrueGolden, public API readiness, or final training.

## 2026-06-14 - E111 Bronze Mutation/Prune Scoped Gold Conversion

- Evidence head at that point: `d71e365752a46b5c94d51cd16359144ed1567553` (`Add E111 bronze mutation prune wave`).
- Latest GitHub release remains `v5.0.0-e79.0`.
- E111 evaluates the remaining E109 Bronze pool under active variant pressure:
  `base_unmodified`, `scope_adapter_mutation`, `io_contract_prune`,
  `mutation_plus_prune`, and `sibling_challenger`.
- Result: 87 candidates, 87 scoped Gold variant promotions, 0 drops, 0 RedFlags,
  0 hard negatives, 0 wrong-scope calls, 0 false commits, 0 unsupported answers,
  and deterministic replay pass.
- Post-wave rank summary after E110 + E111: 136 Gold, 0 Silver, 0 Bronze,
  0 DiamondCandidate, 0 RedFlag, 3 Deprecated.
- Boundary: scoped rank/probation evidence only; not Diamond, Core, PermaCore,
  TrueGolden, public API readiness, or final training.

## 2026-06-14 - E110 Silver-To-Gold Scoped Probation Wave

- Evidence head at that point: `b378c2c5d28475409efeae97bf4bbbfce993c520`
  (`Add E110 promote or drop wave one`).
- E110 applies Silver-to-Gold pressure to the 35 E109 Silver Operators.
- Result: 35 candidates, 35 scoped Gold promotions, 0 kept Silver, 0 RedFlags,
  0 hard negatives, reload match rate `1.000000`, and deterministic replay pass.
- Boundary: scoped Gold promotion only; not Diamond, Core, PermaCore,
  TrueGolden, or final training.

## 2026-06-14 - E109 Operator Rank Ladder And GoldenWatch

- E109 locks the scoped rank ladder and GoldenWatch probation policy.
- Initial E109 rank counts: 14 Gold, 35 Silver, 87 Bronze, 0 DiamondCandidate,
  0 RedFlag, 3 Deprecated.
- Gold requirements include qualified activation, family coverage, campaign
  count, reload/challenger/prune pass, and zero hard negatives.

## 2026-06-14 - E80-E108 Evidence Build

- E80-E85 build the CALC-SCRIBE visible calculation-trace validation line:
  dataset-backed scoring, multiseed training, floor-division closure,
  LocalGolden reload, transfer/negative scope, and mixed-stream no-call routing.
- E86-E89 add LocalGolden seeded curriculum, dense potential/sparse active-set
  selection, survival gauntlet, and Operator naming/schema lock.
- E90-E106 add scoped Operator curriculum expansions for text evidence,
  temporal state, agency guards, output hygiene, active evidence requests,
  memory hygiene, routing, multi-skill execution, scheduling, grounded answers,
  clarification repair, multi-turn continuity, compression, and progress
  tracking.
- E107 verifies the E90-E106 library through survival/regression roles.
- E108 verifies external transfer and negative-scope no-harm behavior.

## 2026-06-14 - E79 Training Data Curriculum Readiness Gate

- Current GitHub release: `v5.0.0-e79.0`.
- Runtime slice: `a908a838a1119540ed88bc91e10cfcb0bdae92a8`
  (`Add training data curriculum readiness gate`).
- E79 adds `vraxion-runtime/src/training_data.rs` and the
  `training_data_readiness` CLI.
- `final_train` runs the E79 gate before global supervisor work and blocks
  fail-fast if the data/curriculum contract cannot cover the full candidate
  rotation.
- CI smokes the standalone E79 readiness command and checks the nested
  readiness artifact tree inside the E79 `final_train` smoke.

## Current Runtime And Evidence Chain

| Slice | Commit | Purpose |
|---|---|---|
| E69-E79 | `a908a838` | Released Rust runtime and training-data readiness gate |
| E80-E85 | `56a9cf03` | CALC-SCRIBE visible calculation-trace evidence |
| E86-E89 | `a6935e61` | LocalGolden curriculum, active-set selection, survival, naming lock |
| E90-E106 | `b75c64cb` | Operator curriculum expansions |
| E107 | `1fcdf954` | Operator survival/regression gauntlet |
| E108 | `0389c211` | External transfer no-harm gauntlet |
| E109 | `555c5006` | Rank ladder and GoldenWatch policy |
| E110 | `b378c2c5` | Silver-to-Gold scoped probation wave |
| E111 | `d71e3657` | Bronze mutation/prune scoped Gold conversion wave |
| E112 | `9de33241` | Gold-to-CoreMemoryCandidate prune-heavy probation wave |
| E113 | `05415f5b` | FineWeb light stress hard mutation/recycle probe |

## Historical Access

Historical release notes before E79 were removed from the active front door because they described superseded beta/grower/byte-pipeline lines. Restore or inspect them from:

```bash
git show archive/repo/pre-e74-public-surface-cleanup-2026-06-13:CHANGELOG.md
```
