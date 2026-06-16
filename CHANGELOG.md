# Changelog

This changelog is narrowed to the current v6 / E127-E136G evidence anchor. Full historical beta, probe, Python SDK, legacy Rust, and research-output history is preserved in git history and archive tags.

## 2026-06-16 - E136G Adaptive Idle Tick Budget Confirm

- Added `scripts/probes/run_e136g_adaptive_idle_tick_budget_confirm.py`.
- Tested the explicit adaptive idle-tick policy: each idle proposal carries a
  one-more-tick recommendation, progress marker, reason, and expected next-tick
  gain, while Agency makes the final continuation decision.
- Result: 24/24 cases passed, proposal continuation fields present in 33/33
  proposals, adaptive execution used 33 ticks versus a 120-tick fixed baseline,
  average adaptive ticks 1.375, 9 Agency-approved continuations, and 2 Agency
  overrides of over-eager continuation requests.
- Coverage: 8 immediate answers stopped at t+1, 3/3 chained cases completed,
  3/3 direct-write cases rejected then repaired at t+2, 4/4 no-pocket controls
  stopped at t+1, 4 unsupported guesses rejected, and 24/24 OutputTextField
  roundtrip/checksum/zero-fill checks.
- Boundary: deterministic adaptive scheduling of explicit proposals only; not
  hidden thought, autonomous background reasoning, next-token prediction,
  consciousness, or open-domain assistant behavior.

## 2026-06-16 - E136F Idle Think-Tick Heldout Series Confirm

- Added `scripts/probes/run_e136f_idle_think_tick_heldout_series_confirm.py`.
- Tested the idle/empty tick idea on a 70-case heldout series: fixed
  observation, 0 new input, `t` advances, idle proposals are checked by
  Agency, then final text commits through `OutputTextField`.
- Result: 70/70 cases passed, 36/36 arithmetic heldout cases improved, 6/6
  no-pocket controls preserved safe output, 90 proposals checked, 0 new input,
  48 total improvements, 70/70 non-degradation, 4 direct-write rejects, 6
  unsupported-claim rejects, and 70/70 OutputTextField roundtrip/checksum/
  zero-fill checks.
- Representative improvement: `I am 17 years old in 2024. When will I be 90?`
  improves from fallback to `You will be 90 years old in 2097.`
- No-pocket control: `I am 25 now, but no year is provided...` preserves a
  safe no-route answer instead of inventing a calendar year.
- Boundary: deterministic heldout proposal refinement only; not hidden thought,
  next-token prediction, background training, consciousness, or open-domain
  assistant behavior.

## 2026-06-16 - E136E Idle Think-Tick Proposal Refinement Smoke

- Added `scripts/probes/run_e136e_idle_think_tick_proposal_refinement_smoke.py`.
- Tested fixed-observation idle ticks: no new input arrives, `t` advances,
  idle operators emit proposals, Agency checks them, and final text commits
  through `OutputTextField`.
- Result: 8/8 cases passed, 10 idle proposals, 10 Agency checks, 0 new input,
  4 improvements, 8/8 non-degradation, 1 direct-write reject, and 8/8
  OutputTextField roundtrip/checksum/zero-fill checks.
- Key sample: the prior `25 years old in 2026 -> 250 years old` prompt improves
  during idle tick to `2251`.
- Boundary: deterministic proposal refinement only; not autonomous hidden
  thought, background training, next-token prediction, consciousness, or
  open-domain assistant behavior.

## 2026-06-16 - E136D OutputTextField Binary Matrix Smoke

- Added a runtime `OutputTextField` representation in `vraxion-runtime/src/text_field.rs`.
- Locked the output-side text field name: `OutputTextField`; `TextRender` is a process/operator, not the field.
- Represented committed output text as an `N x 8` bit matrix where each row is one UTF-8 byte.
- Added `scripts/probes/run_e136d_output_text_field_binary_matrix_smoke.py`.
- Result: 10/10 cases passed, 7/7 committed text roundtrips, 3/3 guarded rejects, 10/10 zero-fill checks, and 1/1 tamper detection.
- Boundary: output-field representation and Agency-gated commit semantics only; not text generation, next-token prediction, open-domain assistant behavior, Core, PermaCore, or TrueGolden.

## 2026-06-16 - E136C Assistant Text Polished Render Quick Test

- Added `scripts/probes/run_e136c_assistant_text_polished_render_quick_test.py`.
- Tested whether E136B assistant/text route stacks can render short polished
  deterministic text instead of raw route/action labels.
- Result: 12/12 quick inference samples passed.
- Covered greeting, summary, code draft without execution claim, source defer,
  JSON output, no-solve math boundary, high-stakes defer, comparison,
  translation, no-overwrite JSON, rejected-response refusal, and scoped outline
  cases.
- Render hygiene: 2/2 JSON outputs valid, 0 raw action leaks, 0 forbidden
  claims, and 0 direct-write claims.
- Boundary: deterministic polished render smoke only; not neural training,
  open-domain LLM/freeform generation, production assistant readiness, Core,
  PermaCore, or TrueGolden.

## 2026-06-16 - E136B Assistant Text Route Composition And Boundary Confirm

- Added `scripts/probes/run_e136b_assistant_text_route_composition_and_boundary_confirm.py`.
- Composed the 18 E136A scoped assistant/text lenses and guards into bounded
  assistant/text route stacks.
- Result: 18/18 route pass operators with 144,000 route cases, 53,000
  multi-route composition cases, 72,000 boundary cases, and 18,000
  negative-scope cases.
- Activation evidence: 144,000 total qualified route activations and 8,000
  minimum qualified route activations per operator.
- Accuracy minima: route accuracy 1.000, route stack accuracy 1.000, primary
  route accuracy 1.000, boundary accuracy 1.000, multi-route composition
  accuracy 1.000, boundary-case accuracy 1.000, and negative-scope accuracy
  1.000.
- Confirmed: 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0
  unsupported answers, 0 boundary-claim violations, and 0 direct Flow writes.
- Controls: overbroad chatbot, unsafe direct-write, source hallucination,
  rejected-response-reuse, and single-operator-drop controls failed as intended.
- Boundary: controlled assistant/text route composition only; not neural
  training, open-domain assistant readiness, production assistant behavior,
  Core, PermaCore, or TrueGolden.

## 2026-06-16 - E136A Assistant Text Skill Farm Mutation/Prune Orange Cycle

- Added `scripts/probes/run_e136a_assistant_text_skill_farm_mutation_prune_orange_cycle.py`.
- Farmed scoped assistant/text lenses and guards from the local E136
  assistant-text seed pack.
- Result: 447,766 rows loaded across 5 external sources and 12 external
  families.
- Promoted 18/18 scoped assistant/text operators to
  OrangeLegendaryCandidate.
- Support evidence: 1,435,199 total external support and 4,746 minimum external
  support per operator.
- Activation evidence: 5,521,276 total qualified activations and 302,123
  minimum qualified activations per operator.
- Mutation/prune evidence: 179,840 mutation attempts, 827 accepted mutations,
  179,013 rejected mutations/rollbacks, and mean selected prune ratio 0.758889.
- Negative-scope guard: 119,868 negative-scope cases, pass-rate minimum 1.000.
- Confirmed: 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0
  unsupported answers, 0 boundary-claim violations, and 0 direct Flow writes.
- Challenger/control: the overbroad chatbot control produced 25,558 wrong-scope
  calls and was rejected.
- Boundary: scoped assistant/text lenses and guards only; not neural training,
  open-domain assistant readiness, production assistant behavior, Core,
  PermaCore, or TrueGolden.

## 2026-06-16 - E135 Math Text Multi-Route Assistant Dialogue-State Gauntlet

- Added `scripts/probes/run_e135_math_text_multi_route_assistant_dialogue_state_gauntlet.py`.
- Stressed the 16 E134 math-text route operators under controlled multi-turn
  assistant dialogue state.
- Result: 16/16 E134 route operators passed the E135 dialogue-state gate.
- Dialogue evidence: 136,000 total dialogue cases and 367,400 total dialogue
  turns.
- Route-state evidence: 29,500 hidden word-problem dialogue no-solve cases,
  10,500 visible reentry cases, 22,400 stale route rejection cases, 11,200
  cross-thread rejection cases, and 76,500 counterexample dialogue cases.
- Accuracy minima: dialogue state accuracy 1.000, current-turn route accuracy
  1.000, route-state integrity 1.000, hidden word-problem dialogue no-solve
  accuracy 1.000, and counterexample dialogue accuracy 1.000.
- Confirmed: 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0
  unsupported answers, 0 boundary-claim violations, 0 direct Flow writes, 0
  stale route reuse, and 0 cross-thread contamination.
- Controls: latest-route reuse, stale-route reuse, cross-thread contamination,
  counterexample trust, and single-turn reset controls all failed as intended.
- Boundary: controlled multi-route dialogue-state only; not open-domain
  dialogue, MATH/GSM8K solving, natural-language word-problem solving, neural
  training, Core, PermaCore, or TrueGolden.

## 2026-06-16 - E134 External Math Text OOD Route Stress And Counterexample Gauntlet

- Added `scripts/probes/run_e134_external_math_text_ood_route_stress_and_counterexample_gauntlet.py`.
- Stressed the 16 E133 math-text route-composition operators under OOD
  wrappers, noisy assistant-style math surfaces, and counterexample lures.
- Result: 16/16 E133 route operators passed the E134 OOD route-stress gate.
- OOD evidence: 208,000 total OOD route cases, 11,875 visible arithmetic OOD
  cases, 153,125 structural guard OOD cases, 43,000 hidden word-problem OOD
  no-solve cases, and 48,000 counterexample cases.
- Accuracy minima: OOD route accuracy 1.000, visible arithmetic OOD accuracy
  1.000, structural guard OOD accuracy 1.000, hidden word-problem OOD no-solve
  accuracy 1.000, and counterexample accuracy 1.000.
- Confirmed: 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0
  unsupported answers, 0 boundary-claim violations, and 0 direct Flow writes.
- Controls: the E133 baseline missed 36,275 OOD cases, the overbroad solver
  control produced 19,200 wrong-scope calls, unsafe trust controls produced
  4,200 false commits, and TIR trust controls produced 2,400 direct-write
  failures.
- Boundary: OOD route stress and counterexample rejection only; not MATH/GSM8K
  solving, natural-language word-problem solving, neural training, Core,
  PermaCore, or TrueGolden.

## 2026-06-16 - E133 Math Text Route Composition And No-Solve Assistant Confirm

- Added `scripts/probes/run_e133_math_text_route_composition_and_no_solve_assistant_confirm.py`.
- Composed the 16 E132 scoped math-text lenses/guards into assistant route
  decisions on top of the E131/E129 visible arithmetic route.
- Result: 16/16 E132 math-text operators passed the E133 route-composition gate.
- Route evidence: 176,000 total route/no-solve cases, 10,000 visible
  arithmetic route cases, 118,000 structural guard cases, and 48,000 hidden
  word-problem no-solve cases.
- Accuracy minima: route accuracy 1.000, visible arithmetic route accuracy
  1.000, structural guard accuracy 1.000, and hidden word-problem no-solve
  accuracy 1.000.
- Confirmed: 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0
  unsupported answers, 0 boundary-claim violations, and 0 direct Flow writes.
- Controls: the overbroad solver control produced 24,000 wrong-scope calls,
  unsafe trust controls produced 4,125 false commits, and TIR trust controls
  produced 3,000 direct-write failures.
- Boundary: route composition only; not MATH/GSM8K solving, natural-language
  word-problem solving, neural training, Core, PermaCore, or TrueGolden.

## 2026-06-16 - E132 External Math Text Skill Farm Mutation/Prune Orange Cycle

- Added `scripts/probes/run_e132_external_math_text_skill_farm_mutation_prune_orange_cycle.py`.
- Downloaded and normalized an E132 external math-text seed pack under ignored
  `target/`: MathInstruct, NuminaMath-TIR, OpenAssistant/oasst1, Dolly 15k,
  and EleutherAI Hendrycks MATH.
- Result: 215,051 rows loaded across 5 sources and 11 families.
- Promoted 16/16 scoped math-text lenses/guards to OrangeLegendaryCandidate.
- Minimum external support per operator: 5,953.
- Activation evidence: 4,883,030 total qualified activations and 302,510
  minimum qualified activations per operator.
- Mutation/prune evidence: 146,005 mutation attempts, 650 accepted mutations,
  145,355 rejected mutations/rollbacks, 998 prune attempts, 469 challenger
  attempts, and mean selected prune ratio 0.736875.
- Negative-scope guard: 78,859 negative-scope cases, pass-rate minimum 1.000.
- Confirmed: 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0
  unsupported answers, 0 boundary-claim violations, and 0 direct Flow writes.
- Challenger/control: the overbroad solver control produced 16,703 wrong-scope
  calls and was rejected.
- Boundary: scoped math-text lenses and guards only; not GSM8K/MATH solving,
  hidden natural-language word-problem solving, neural training, Core,
  PermaCore, or TrueGolden.

## 2026-06-16 - E131 Visible Equation Extraction And Assistant Arithmetic Render

- Added `scripts/probes/run_e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet.py`.
- Routed the 9 E129/E130B scoped arithmetic trace Operators from
  assistant-style visible equation surfaces seeded by the external E131 text
  pack.
- Result: 9/9 arithmetic Operators passed the E131 visible-equation assistant
  render gate.
- Dataset seed: 130,000 normalized rows loaded from MathInstruct, Dolly 15k,
  and OpenAssistant/oasst1.
- Positive transfer: 108,000 visible-equation cases, visible equation
  extraction accuracy minimum 1.000, and 108,000 qualified visible activations.
- Negative-scope guard: 54,000 hidden prose-only word-problem cases produced
  word-problem no-call accuracy minimum 1.000.
- Confirmed: 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0
  unsupported answers, 0 boundary-claim violations, and 0 direct flow writes.
- Challenger/control: the E130B baseline missed 96,711 new visible-equation
  surfaces, and the overbroad word-problem solver control produced 18,000
  wrong-scope calls.
- Boundary: visible equation extraction and deterministic assistant arithmetic
  rendering only; not hidden natural-language word-problem solving, GSM8K
  solving, open-domain reasoning, or neural LLM training.

## 2026-06-15 - E130B Arithmetic Text-IO Transfer And Word-Problem No-Call

- Added `scripts/probes/run_e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet.py`.
- Transferred the 9 E129 scoped arithmetic trace Operators into longer
  visible-expression text-IO wrappers.
- Result: 9/9 arithmetic Operators passed the E130B transfer gate.
- Positive transfer: 270,000 visible-transfer cases, visible transfer accuracy
  minimum 1.000, and 270,000 qualified transfer activations.
- Negative-scope guard: 135,000 hidden natural-language word-problem cases
  produced word-problem no-call accuracy minimum 1.000.
- Confirmed: 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0
  unsupported answers, and 0 direct flow writes.
- Challenger/control: the overbroad word-problem solver control produced 18,000
  wrong-scope calls and was rejected.
- Boundary: visible arithmetic expression/trace text IO only; not hidden
  natural-language word-problem solving, GSM8K solving, open-domain reasoning,
  or neural LLM training.

## 2026-06-15 - E130A CoreMemoryCandidate To Orange Backfill Gauntlet

- Added `scripts/probes/run_e130a_corememory_to_orange_backfill_gauntlet.py`.
- Backfilled the 136 E112 CoreMemoryCandidate Operators through an E121-style
  Orange/Legendary probation gate.
- Result: 136/136 reached OrangeLegendaryCandidate.
- Activation evidence: 13,877,699 qualified activations before E130A,
  27,158,734 added qualified activations, 41,036,433 total qualified
  activations, and 300,623 minimum qualified activations per operator.
- Safety/probation checks: family coverage minimum 20, campaign count minimum
  8, mean selected prune ratio 0.746176, reload/negative-scope/challenger/prune
  pass rates 1.000000.
- Confirmed: 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0
  unsupported answers, 0 negative transfers, and 0 direct flow writes.
- Dashboard result: 530 total operators, 527 Orange/LegendaryCandidate scoped
  operators, 0 CoreMemoryCandidate operators, and 3 Deprecated operators.
- Boundary: scoped Operator rank backfill only; not PermaCore, TrueGolden,
  production assistant behavior, final training, or open-domain reasoning.

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
