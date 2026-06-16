# VRAXION Codex Handover

Last updated: 2026-06-16

## Start Here

```text
repo = VRAXION_anchorwiki
branch = main
latest_release_target = v6.1.7
current_evidence_anchor = E135 math-text multi-route assistant dialogue-state gauntlet on main
current_status = E135 confirmed 16/16 E134 route operators preserve current-turn route state across 136,000 dialogue cases and 367,400 turns
```

This is the first file a fresh Codex should read after cloning the repo.

## Current State

E127 completed a supervised overnight cyclic operator farm:

```text
cycles = 40
Orange/Legendary scoped operators = 382
mutation attempts = 1,849,625
accepted mutations = 12,123
rollbacks = 1,837,502

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
```

The latest text-to-text smoke is a deterministic operator + template renderer,
not an LLM/freeform generation claim. It shows proto-assistant behavior over
short prompts using the governed operator library.

E128 extends that bridge with a no-download, local assistant-style corpus:

```text
prompt corpus = 320
train / validation / heldout = 160 / 64 / 96
source mix = E127 smoke seeds, E127 operator artifacts, repo docs,
             adversarial boundary prompts, FineWeb-derived local noise
train action accuracy = 1.000
validation action accuracy = 1.000
heldout action accuracy = 1.000
operator trace validity = 1.000
unsupported answers = 0
wrong refusals = 0
boundary-claim violations = 0
```

E129 adds exact arithmetic text-IO trace/operator evidence:

```text
operator_count = 9
Orange/Legendary arithmetic operators = 9
qualified_activation_min = 300000
qualified_activation_total = 2700000
negative_scope_case_count_total = 9000

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0

covered = add/sub, multiplication, exact division, floor division,
          signed integers, decimal/fraction rendering, mixed precedence,
          invalid trace rejection, division-by-zero rejection
```

E130A backfills the prior E112 CoreMemoryCandidate pool through an E121-style
Orange/Legendary gate:

```text
candidate_count = 136
Orange/Legendary backfilled operators = 136
qualified_activation_before_total = 13,877,699
qualified_activation_add_total = 27,158,734
qualified_activation_total = 41,036,433
qualified_activation_min = 300,623
mean_selected_prune_ratio = 0.746176

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
negative transfer = 0
direct flow writes = 0

dashboard operator count = 530
dashboard Orange/LegendaryCandidate = 527
dashboard CoreMemoryCandidate = 0
dashboard Deprecated = 3
```

E130B transfers the E129 arithmetic trace operators into visible-expression
text IO:

```text
operator_count = 9
transfer_pass_operator_count = 9
visible_transfer_case_count_total = 270,000
word_problem_no_call_case_count_total = 135,000
qualified_transfer_activation_total = 270,000

visible_transfer_accuracy_min = 1.000
word_problem_no_call_accuracy_min = 1.000

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
direct flow writes = 0

overbroad_control_wrong_scope_call_total = 18,000
```

E131 routes those arithmetic operators from assistant-style visible equation
surfaces seeded by the external E131 text pack:

```text
dataset_rows_loaded = 130,000
operator_count = 9
transfer_pass_operator_count = 9
visible_equation_case_count_total = 108,000
word_problem_no_call_case_count_total = 54,000
qualified_visible_activation_total = 108,000

visible_equation_extraction_accuracy_min = 1.000
word_problem_no_call_accuracy_min = 1.000

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
boundary claim violations = 0
direct flow writes = 0

E130B baseline visible misses = 96,711
overbroad_control_wrong_scope_call_total = 18,000
```

E132 farms new scoped math-text lenses/guards from the external E132 seed pack:

```text
dataset_rows_loaded = 215,051
external_sources = 5
external_families = 11
operator_count = 16
Orange/Legendary math-text operators = 16
external_support_min = 5,953
qualified_activation_total = 4,883,030
qualified_activation_min = 302,510
negative_scope_case_count_total = 78,859
mutation_attempts_total = 146,005
accepted_mutations_total = 650
rollback_count_total = 145,355
mean_selected_prune_ratio = 0.736875

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
boundary claim violations = 0
direct flow writes = 0
overbroad_solver_control_wrong_scope_call_total = 16,703
```

E133 composes those E132 math-text lenses/guards into assistant route decisions:

```text
operator_count = 16
composition_pass_operator_count = 16
route_case_count_total = 176,000
visible_arithmetic_route_case_count_total = 10,000
structural_guard_case_count_total = 118,000
hidden_word_problem_no_solve_case_count_total = 48,000
qualified_route_activation_total = 176,000
qualified_route_activation_min = 11,000

route_accuracy_min = 1.000
visible_arithmetic_route_accuracy_min = 1.000
structural_guard_accuracy_min = 1.000
hidden_word_problem_no_solve_accuracy_min = 1.000

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
boundary claim violations = 0
direct flow writes = 0

overbroad_solver_control_wrong_scope_call_total = 24,000
trust_control_false_commit_total = 4,125
trust_control_direct_flow_write_total = 3,000
```

E134 stresses those E133 route operators under OOD wrappers and counterexample
lures:

```text
operator_count = 16
ood_pass_operator_count = 16
ood_case_count_total = 208,000
visible_arithmetic_ood_case_count_total = 11,875
structural_guard_ood_case_count_total = 153,125
hidden_word_problem_ood_no_solve_case_count_total = 43,000
counterexample_case_count_total = 48,000
qualified_ood_route_activation_total = 208,000
qualified_ood_route_activation_min = 13,000

ood_route_accuracy_min = 1.000
visible_arithmetic_ood_accuracy_min = 1.000
structural_guard_ood_accuracy_min = 1.000
hidden_word_problem_ood_no_solve_accuracy_min = 1.000
counterexample_accuracy_min = 1.000

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
boundary claim violations = 0
direct flow writes = 0

E133 baseline OOD misses = 36,275
overbroad_solver_control_wrong_scope_call_total = 19,200
trust_control_false_commit_total = 4,200
trust_control_direct_flow_write_total = 2,400
```

E135 stresses those E134 route operators under controlled multi-turn assistant
dialogue state:

```text
operator_count = 16
dialogue_pass_operator_count = 16
dialogue_case_count_total = 136,000
dialogue_turn_count_total = 367,400
hidden_word_problem_dialogue_no_solve_case_count_total = 29,500
visible_reentry_dialogue_case_count_total = 10,500
stale_route_rejection_case_count_total = 22,400
cross_thread_rejection_case_count_total = 11,200
counterexample_dialogue_case_count_total = 76,500
qualified_dialogue_route_activation_total = 136,000
qualified_dialogue_route_activation_min = 8,500

dialogue_state_accuracy_min = 1.000
current_turn_route_accuracy_min = 1.000
route_state_integrity_min = 1.000
hidden_word_problem_dialogue_no_solve_accuracy_min = 1.000
counterexample_dialogue_accuracy_min = 1.000

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
boundary claim violations = 0
direct flow writes = 0
stale route reuse = 0
cross-thread contamination = 0
```

## Claim Boundary

Allowed:

```text
VRAXION v6 has a governed operator-library checkpoint with 382 scoped
Orange/Legendary text operators, continuous checkpointing discipline, a
dashboard sample pack, a deterministic text-to-text render smoke, and an E128
lightweight assistant text-IO corpus/render smoke. E129 confirms scoped exact
arithmetic trace Operators through Orange/Legendary probation. E130A confirms
the prior CoreMemoryCandidate pool can be backfilled to Orange/Legendary under
the stricter activation, no-harm, reload, challenger, and prune gate. E130B
confirms those arithmetic operators transfer to visible-expression text IO while
hidden word problems remain no-call. E131 confirms assistant-style visible
equation extraction and deterministic arithmetic rendering while hidden
prose-only word problems remain no-call. E132 confirms scoped math-text
lenses/guards can be farmed from external math text and promoted through the
Orange mutation/prune/no-solve gate. E133 confirms those math-text
lenses/guards compose into assistant route decisions while hidden word problems
remain no-call and unsafe trust controls fail. E134 confirms those routes
survive OOD wrapper stress and counterexample lures while keeping the same
no-solve and no-direct-write boundary. E135 confirms those routes preserve
current-turn route state under controlled multi-turn assistant dialogue pressure
without stale route reuse or cross-thread contamination.
```

System-level interpretation:

```text
It is a governed Operator/Pocket runtime.
It can reject/defer unsupported or conflicting evidence in scoped tasks.
It can render guarded template-style responses in controlled smoke tests.
It can build and pass a local 320-prompt assistant-style action-policy/template
smoke without downloading a large external chat dataset.
It can compute or validate visible arithmetic expressions/traces in scoped
cases including multiplication and division.
It can handle visible arithmetic expression/trace payloads inside longer text
wrappers while no-calling hidden prose-only word problems.
It can stress those math-text assistant routes under OOD wrappers and
counterexample lures while keeping tracked hard negatives and direct writes at
zero.
It can preserve current-turn route state across controlled multi-turn
math-text assistant dialogue surfaces.
It can promote prior scoped CoreMemoryCandidate operators to Orange/Legendary
when the E121-style gate is satisfied.
It is not an open-domain LLM/chatbot.
```

Not allowed:

```text
open-domain chatbot
Gemma/GPT-level generation
production assistant
PermaCore / TrueGolden promotion
consciousness / sentience
unbounded reasoning claim
```

## Key Files

```text
README.md
docs/CURRENT_STATUS.md
docs/CURRENT_CAPABILITIES.md
docs/GETTING_STARTED.md
docs/VERSION.json
CHANGELOG.md

docs/research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md
docs/research/E127_TEXT_TO_TEXT_RENDER_SMOKE_CURRENT_RESULT.md
docs/research/E128_ASSISTANT_TEXT_IO_LIGHTWEIGHT_RENDER_TRAINING_RESULT.md
docs/research/E129_ARITHMETIC_TRACE_ORANGE_LEGENDARY_PROBATION_RESULT.md
docs/research/E130A_COREMEMORY_TO_ORANGE_BACKFILL_GAUNTLET_RESULT.md
docs/research/E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET_RESULT.md
docs/research/E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET_RESULT.md
docs/research/E132_EXTERNAL_MATH_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md
docs/research/E133_MATH_TEXT_ROUTE_COMPOSITION_AND_NO_SOLVE_ASSISTANT_CONFIRM_RESULT.md
docs/research/E134_EXTERNAL_MATH_TEXT_OOD_ROUTE_STRESS_AND_COUNTEREXAMPLE_GAUNTLET_RESULT.md
docs/research/E135_MATH_TEXT_MULTI_ROUTE_ASSISTANT_DIALOGUE_STATE_GAUNTLET_RESULT.md
docs/research/artifact_samples/e127_overnight_text_skill_farm_orange_cycle/
docs/research/artifact_samples/e127_text_to_text_render_smoke_current/
docs/research/artifact_samples/e128_assistant_text_io_lightweight_render_training/
docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation/
docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet/
docs/research/artifact_samples/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet/
docs/research/artifact_samples/e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet/
docs/research/artifact_samples/e132_external_math_text_skill_farm_mutation_prune_orange_cycle/
docs/research/artifact_samples/e133_math_text_route_composition_and_no_solve_assistant_confirm/
docs/research/artifact_samples/e134_external_math_text_ood_route_stress_and_counterexample_gauntlet/
docs/research/artifact_samples/e135_math_text_multi_route_assistant_dialogue_state_gauntlet/
```

## Legal / License

The root [`LICENSE`](LICENSE) is the controlling license text. VRAXION uses the
custom **VRAXION Community Source License 1.0**: broad free community use, but
not OSI-approved open source.

Commercial monetization of third-party access to VRAXION-powered functionality
is Royalty Use under the license unless a separate written agreement exists.
The default royalty is 19% of Attributable Net Revenue:

```text
1% Founder Allocation
18% VRAXION Forever Prize Allocation
```

Read these before changing release/legal language:

```text
LICENSE
legal/LEGAL.md
legal/COMMERCIAL_USE_GUIDE.md
legal/VRAXION_FOREVER_PRIZE_CHARTER.md
docs/legal/PRIOR_ART_AND_PROVENANCE_CHECKLIST.md
```

## Dashboard

Generate the local operator dashboard:

```powershell
python scripts/tools/generate_operator_rank_dashboard.py --e127 docs/research/artifact_samples/e127_overnight_text_skill_farm_orange_cycle --e129 docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation --e130a docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet --e130b docs/research/artifact_samples/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet --e131 docs/research/artifact_samples/e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet --e132 docs/research/artifact_samples/e132_external_math_text_skill_farm_mutation_prune_orange_cycle --e133 docs/research/artifact_samples/e133_math_text_route_composition_and_no_solve_assistant_confirm --e134 docs/research/artifact_samples/e134_external_math_text_ood_route_stress_and_counterexample_gauntlet --e135 docs/research/artifact_samples/e135_math_text_multi_route_assistant_dialogue_state_gauntlet --out target/pilot_wave/operator_rank_dashboard/index.html
```

Expected current dashboard cards:

```text
E130B TRANSFER = 9/9
E130B VISIBLE IO = 100.000%
E130B WORD NO-CALL = 100.000%
E131 VISIBLE EQ = 9/9
E131 EXTRACTION = 100.000%
E131 WORD NO-CALL = 100.000%
E132 MATH-TEXT = 16/16
E132 DATASET ROWS = 215,051
E132 SUPPORT MIN = 5,953
E133 ROUTE COMP = 16/16
E133 ROUTE CASES = 176,000
E133 HIDDEN NO-CALL = 100.000%
E134 OOD PASS = 16/16
E134 OOD CASES = 208,000
E134 COUNTEREXAMPLES = 48,000
E134 HIDDEN NO-CALL = 100.000%
E135 DIALOGUE = 16/16
E135 CASES = 136,000
E135 TURNS = 367,400
E135 STATE INTEGRITY = 100.000%
ORANGE/LEGENDARY CANDIDATE = 543
```

## Local E136 Seed Pack

After E135, a local assistant/text seed pack was downloaded and normalized for
the next LLM-ish assistant/operator farming direction:

```text
pack_id = e136_assistant_text_seed_pack
raw_root = target/datasets/e136_assistant_text_seed_pack/raw
normalized = target/datasets/e136_assistant_text_seed_pack/normalized/e136_assistant_text_skill_seed.jsonl
download_report = docs/research/E136_ASSISTANT_TEXT_SEED_PACK_DOWNLOAD_REPORT.md

raw_size = 2.726 GiB
normalized_size = 2.430 GiB
normalized_rows = 447,766

source rows:
HuggingFaceH4/ultrachat_200k = 220,000
Open-Orca/SlimOrca = 120,000
OpenAssistant/oasst2 = 37,766
Anthropic/hh-rlhf = 60,000
HuggingFaceH4/no_robots = 10,000
```

This is data readiness only. It is not a training pass, not an E136 evidence
confirmation, and not a new current evidence anchor. Raw and normalized data
live under `target/` and are intentionally not committed.

## Quick Checks

Run these after clone or before continuing:

```powershell
git status --short --branch
python -m py_compile scripts/tools/prepare_e136_assistant_text_seed_pack.py
python -m py_compile scripts/probes/run_e127_overnight_text_skill_farm_orange_cycle.py scripts/tools/generate_operator_rank_dashboard.py
python -m py_compile scripts/probes/run_e128_assistant_text_io_lightweight_render_training.py
python scripts/probes/run_e128_assistant_text_io_lightweight_render_training.py --out target/ci/e128_assistant_text_io_lightweight_render_training
python -m py_compile scripts/probes/run_e129_arithmetic_trace_orange_legendary_probation.py
python -m py_compile scripts/probes/run_e130a_corememory_to_orange_backfill_gauntlet.py
python -m py_compile scripts/probes/run_e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet.py
python -m py_compile scripts/probes/run_e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet.py
python -m py_compile scripts/probes/run_e132_external_math_text_skill_farm_mutation_prune_orange_cycle.py
python -m py_compile scripts/probes/run_e133_math_text_route_composition_and_no_solve_assistant_confirm.py
python -m py_compile scripts/probes/run_e134_external_math_text_ood_route_stress_and_counterexample_gauntlet.py
python -m py_compile scripts/probes/run_e135_math_text_multi_route_assistant_dialogue_state_gauntlet.py
python -m compileall -q scripts
cargo test --workspace
git diff --check
```

If the machine is slow or missing Rust tooling, report the dependency state
honestly instead of faking results.

## Operating Rules

1. No black-box runs. Long jobs must write partial outcomes every few minutes or
   faster.
2. Do not run tiny sequential compute if a real multi-seed/multi-worker run is
   the actual goal.
3. Do not stage unrelated dirty or untracked files.
4. Push clean successful checkpoints to `origin/main`.
5. Keep claim boundaries explicit.

## Current Untracked Local Leftovers

At the time of this handover, these old sample artifact directories may exist
locally as untracked files. They were intentionally left untouched:

```text
docs/research/artifact_samples/e31_breakpoint_ladder_and_bottleneck_localization/
docs/research/artifact_samples/e32_flow_field_capacity_and_trace_ledger_repair_confirm/
docs/research/artifact_samples/e44_abstract_payload_wire_capacity_smoke/
docs/research/artifact_samples/e44b_parallel_abstract_wire_stream_field_smoke/
docs/research/artifact_samples/e44c_reserve_wire_mask_and_noise_stress/
docs/research/artifact_samples/e63_pocket_observatory_visual_debug_dashboard/
docs/research/artifact_samples/e64_field_size_and_agency_view_sweep/
docs/research/artifact_samples/e65_locked_body_runtime_integration_preflight/
```

Do not delete or commit them without an explicit cleanup decision.

## Next Work

Recommended next steps:

```text
1. Use the local E136 assistant-text seed pack to run a scoped assistant/text
   operator farm or transfer probe.
2. Run E136 assistant math-text dialogue route transfer and latency compare if
   the goal remains route-state latency/transfer rather than new text skills.
3. Decide whether to continue E127 with a fresh candidate pack or pause farming.
4. Build the next bridge from deterministic action/template rendering,
   E132/E133/E134/E135 math-text lenses/guards/routes/OOD/dialogue-state
   evidence, and exact arithmetic trace/text-IO operators toward richer
   assistant arithmetic while keeping claims scoped.
5. Keep all new skills scoped, dashboard-visible, and rollback-safe.
```
