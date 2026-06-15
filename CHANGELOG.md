# Changelog

This changelog is narrowed to the current v6 / E127-E129 evidence anchor. Full historical beta, probe, Python SDK, legacy Rust, and research-output history is preserved in git history and archive tags.

## 2026-06-15 - E129 Arithmetic Trace Orange/Legendary Probation

- Added `scripts/probes/run_e129_arithmetic_trace_orange_legendary_probation.py`.
- Added exact arithmetic trace training/probation evidence for scoped direct
  arithmetic operators.
- Result: 9/9 arithmetic operators reached Orange/LegendaryCandidate with
  300,000 qualified activations each and 2,700,000 qualified activations total.
- Covered operations: addition/subtraction, multiplication, exact division,
  floor division, signed integers, decimal/fraction rendering, parenthesized
  mixed precedence, invalid trace rejection, and division-by-zero rejection.
- Negative-scope check: 9,000 natural-language word-problem/no-visible-trace
  cases produced 0 wrong-scope calls.
- Confirmed: 0 hard negatives, 0 false commits, 0 wrong-scope calls, and 0
  unsupported answers.
- Boundary: exact arithmetic expression/trace compute and validation only; not
  natural-language word-problem solving, GSM8K solving, open-domain reasoning,
  neural LLM training, PermaCore, or TrueGolden.

## 2026-06-15 - E128 Assistant Text-IO Lightweight Render Training

- Added `scripts/probes/run_e128_assistant_text_io_lightweight_render_training.py`.
- Added a no-download assistant-style corpus/action-policy/template smoke on top
  of the E127 operator-library checkpoint.
- Result: 320 local prompts split into 160 train, 64 validation, and 96 heldout
  rows.
- Source mix: 40 E127 smoke seed prompts, 88 E127 operator-derived prompts, 96
  repo-doc grounded prompts, 64 adversarial boundary prompts, and 32
  FineWeb-derived local noise prompts.
- Confirmed: train/validation/heldout action accuracy 1.000, operator trace
  validity 1.000, 0 unsupported answers, 0 wrong refusals, and 0 boundary-claim
  violations.
- Boundary: deterministic corpus plus action-policy/template rendering only; not
  neural LLM training, learned general weights, open-domain chatbot behavior, or
  freeform generation.

## 2026-06-15 - v6.1.7 / E127 Text Operator Library Checkpoint

- Current GitHub release: `v6.1.7`.
- Current evidence anchor: `f32a6f4b` (`Finalize E127 cycle 40 checkpoint`).
- E127 completed 40 checkpointed overnight text-operator farming cycles.
- Result: 382 scoped Orange/Legendary text operators, 1,849,625 mutation
  attempts, 12,123 accepted mutations, 1,837,502 rollbacks, 0 hard negatives,
  0 false commits, 0 wrong-scope calls, and 0 unsupported answers.
- Added `CODEX_HANDOVER.md` as the first-read file for fresh Codex sessions.
- Added deterministic text-to-text render smoke artifacts. Boundary: operator
  selection plus guarded template rendering, not LLM/freeform generation.
- Claim boundary: scoped governed operator-library evidence only; not PermaCore,
  TrueGolden, production assistant readiness, Gemma/GPT-like generation,
  consciousness, or sentience.

## 2026-06-14 - E113 FineWeb Light Stress Hard Mutation Recycle

- Current evidence anchor: `05415f5b06a43440742715ea93a5e2ec97632f21`
  (`Add E113 FineWeb light stress recycle probe`).
- At that point, the latest GitHub release was `v5.0.0-e79.0`.
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
- At that point, the latest GitHub release was `v5.0.0-e79.0`.
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
- At that point, the latest GitHub release was `v5.0.0-e79.0`.
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

- At that point, the current GitHub release was `v5.0.0-e79.0`.
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
