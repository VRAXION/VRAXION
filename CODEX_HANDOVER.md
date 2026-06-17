# VRAXION Codex Handover

Last updated: 2026-06-17

## Start Here

```text
repo = VRAXION_anchorwiki
branch = main
latest_release_target = v6.1.7
current_evidence_anchor = E136N primary/secondary variant governance on main
current_status = E136N installs a primary/secondary registry skeleton over the E136M runtime-facing overlay
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

E136A farms scoped assistant/text lenses and guards from the local E136 seed
pack:

```text
dataset_rows_loaded = 447,766
external_sources = 5
external_families = 12
operator_count = 18
Orange/Legendary assistant/text operators = 18
external_support_total = 1,435,199
external_support_min = 4,746
qualified_activation_total = 5,521,276
qualified_activation_min = 302,123
negative_scope_case_count_total = 119,868
mutation_attempts_total = 179,840
accepted_mutations_total = 827
rollback_count_total = 179,013
mean_selected_prune_ratio = 0.758889

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
boundary claim violations = 0
direct flow writes = 0

overbroad_chatbot_control_wrong_scope_call_total = 25,558
```

E136B composes those assistant/text lenses and guards into bounded route stacks:

```text
source_e136a_operator_count = 18
route_pass_operator_count = 18 / 18
dataset_rows_loaded = 447,766
route_seed_row_count = 4,096
route_case_count_total = 144,000
multi_route_composition_case_count_total = 53,000
boundary_case_count_total = 72,000
negative_scope_case_count_total = 18,000
qualified_route_activation_total = 144,000
qualified_route_activation_min = 8,000
route_family_count = 10

route_accuracy_min = 1.000
route_stack_accuracy_min = 1.000
primary_route_accuracy_min = 1.000
boundary_accuracy_min = 1.000
multi_route_composition_accuracy_min = 1.000
negative_scope_accuracy_min = 1.000

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
boundary claim violations = 0
direct flow writes = 0

overbroad_chatbot_control_wrong_scope_call_total = 14,400
unsafe_direct_write_control_direct_flow_write_total = 14,400
source_hallucination_control_unsupported_answer_total = 14,400
```

E136C renders those route stacks into short polished deterministic text:

```text
case_count = 12
pass_count = 12
fail_count = 0
mode_accuracy = 1.000
polished_render_pass_rate = 1.000
json_valid_count = 2 / 2
json_keys_pass_count = 2 / 2
average_response_words = 27.083
route_stack_covered_count = 11
greeting_fallback_count = 1

raw_action_leak_total = 0
forbidden_claim_total = 0
direct_write_claim_total = 0
```

E136D locks the output-side text field representation:

```text
field_name = OutputTextField
matrix_shape = N x 8 bit cells
row_meaning = one UTF-8 byte
case_count = 10
pass_count = 10
fail_count = 0
commit_case_count = 7
reject_case_count = 3
roundtrip_pass_count = 7
zero_fill_pass_count = 10
overflow_reject_count = 1
direct_write_reject_count = 1
nul_reject_count = 1
tamper_detect_count = 1
```

E136E tests empty/idle think ticks after an observation:

```text
case_count = 8
pass_count = 8
fail_count = 0
idle_tick_total = 10
proposal_count = 10
agency_check_count = 10
new_input_total = 0
improvement_count = 4
non_degradation_count = 8
direct_write_reject_count = 1
output_roundtrip_count = 8
output_checksum_count = 8
output_zero_fill_count = 8

sample:
"Hello, most 25 éves vagyok 2026 ban, mikor leszek 250 éves?"
-> idle proposal after t+1:
"2251-ben leszel 250 éves."
```

E136F tests the same idle mechanism on a heldout series:

```text
case_count = 70
pass_count = 70
fail_count = 0
arithmetic_case_count = 36
arithmetic_improvement_count = 36
no_pocket_case_count = 6
no_pocket_preserve_count = 6
idle_tick_total = 90
proposal_count = 90
agency_check_count = 90
new_input_total = 0
improvement_count = 48
non_degradation_count = 70
direct_write_reject_count = 4
unsupported_claim_reject_count = 6
output_roundtrip_count = 70
output_checksum_count = 70
output_zero_fill_count = 70

sample:
"I am 17 years old in 2024. When will I be 90 years old?"
-> idle proposal after t+1:
"You will be 90 years old in 2097."

no-pocket sample:
"I am 25 now, but no year is provided. What calendar year will I be 250?"
-> preserved safe no-route answer; no unsupported year is invented.
```

E136G adds an adaptive idle tick budget:

```text
case_count = 24
pass_count = 24
fail_count = 0
adaptive_tick_total = 33
fixed_baseline_tick_total = 120
tick_savings_vs_fixed = 87
average_adaptive_ticks = 1.375000
proposal_count = 33
proposal_continue_field_count = 33
agency_continue_yes_count = 9
agency_continue_override_count = 2
immediate_answer_stop_t1_count = 8
chained_case_count = 3
chained_complete_count = 3
direct_write_case_count = 3
direct_write_reject_count = 3
direct_write_repair_t2_count = 3
no_pocket_case_count = 4
no_pocket_stop_t1_count = 4
unsupported_claim_reject_count = 4
new_input_total = 0
output_roundtrip_count = 24
output_checksum_count = 24
output_zero_fill_count = 24

sample:
t+1 proposal says continue only if useful
-> Agency approves safe progress, or overrides over-eager continuation
```

E136H adds an existing-operator refinement night cycle:

```text
cycles_completed = 40
operator_count = 34
rows_seen_total = 12,480,000
current_activation_total = 3,373,788
selected_activation_total = 2,891,151
pruned_activation_total = 482,637
verified_label_count = 16
tentative_label_tighten_count = 11
abstract_but_useful_count = 7
hold_for_more_evidence_count = 0
hard_negative_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0

meaning:
Existing operators are refined first. Low human-label alignment is not an
automatic drop if the kernel still carries useful signal.
```

E136I adds a supersession and output-impact ledger for the E136H selected
variants:

```text
operator_count = 34
replacement_ready_count = 27
direct_runtime_candidate_count = 16
tightened_challenger_required_count = 11
abstract_lineage_required_count = 7
hold_for_more_evidence_count = 0
destructive_drop_count = 0

projected_current_activation_total = 3,373,788
projected_selected_activation_total = 2,891,151
projected_pruned_activation_total = 482,637
projected_output_activation_delta_total = -482,637

accepted_mutation_total = 96
mutation_attempt_total = 43,720
hard_negative_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0

meaning:
E136I does not mutate the runtime. It classifies selected variants into direct
replacement candidates, challenger-required replacements, abstract lineage
kernels, and output/prune impact before any destructive apply.
```

E136J shadow-applies the E136I replacement ledger without destructive runtime
mutation or prune:

```text
stop_reason = deadline
cycles_completed = 8,094
rows_processed = 33,153,024
elapsed_seconds = 46,317.709

operator_count = 34
replacement_ready_count = 27
direct_runtime_candidate_count = 16
tightened_challenger_required_count = 11
abstract_lineage_required_count = 7

current_activation_total = 188,597,925
selected_activation_total = 166,354,720
shadow_pruned_activation_total = 22,243,205

strict_recall_miss_total = 0
wrong_scope_proxy_total = 0
hard_negative_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0

meaning:
E136J is the non-destructive shadow-apply confirmation for E136I. It passed the
previous premature-finish failure mode, ran past six hours, and stopped only at
the explicit 19:00 deadline.
```

E136K converts that evidence into a non-destructive apply plan:

```text
operator_count = 34
direct_canary_ready_count = 16
challenger_ood_required_count = 11
abstract_lineage_required_count = 7
runtime_mutation_allowed_now_count = 0
destructive_apply_count = 0
rollback_manifest_count = 16

shadow_pruned_activation_total = 22,243,205
direct_canary_shadow_prune_ratio = 0.057159
challenger_shadow_prune_ratio = 0.351548

strict_recall_miss_total = 0
wrong_scope_proxy_total = 0
hard_negative_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0

meaning:
16 candidates can move to rollback-safe runtime canary planning. The 11
tightened-trigger replacements are promising but must pass challenger/OOD
canary before runtime replacement. The 7 abstract kernels remain lineage holds.
```

E136L tests that apply plan under rollback-safe runtime canary simulation:

```text
operator_count = 34
direct_canary_tested_count = 16
direct_canary_pass_count = 16
old_operator_removed_in_canary_count = 16
runtime_replacement_canary_allowed_count = 16
production_runtime_apply_count = 0
destructive_apply_count = 0

challenger_ood_tested_count = 11
challenger_hold_count = 11
challenger_runtime_apply_allowed_count = 0
abstract_lineage_hold_count = 7

rollback_manifest_count = 16
rollback_trigger_count = 0
direct_canary_removed_activation_total = 3,450,257
sample_direct_removed_activation_total = 1,031

strict_recall_miss_total = 0
wrong_scope_proxy_total = 0
hard_negative_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0

meaning:
The 16 direct candidates passed the "old trigger removed, selected variant
active" canary replay. This is not a destructive production apply. The 11
challenger rows and 7 abstract lineage rows stay held.
```

E136M materializes the E136L direct canary wins as a runtime-facing overlay:

```text
operator_count = 34
runtime_overlay_active_count = 16
runtime_overlay_apply_count = 16
verified_replacement_apply_count = 7
light_prune_overlay_apply_count = 9
legacy_trigger_disabled_in_overlay_count = 16
legacy_trigger_retained_for_rollback_count = 16

challenger_ood_queue_count = 11
challenger_runtime_overlay_active_count = 0
abstract_lineage_split_count = 7
abstract_runtime_overlay_active_count = 0

rollback_snapshot_count = 16
rollback_trigger_count = 0
production_destructive_delete_count = 0
runtime_mutation_allowed_now_count = 16

runtime_overlay_removed_activation_total = 3,450,257
runtime_overlay_removed_activation_ratio = 0.018294
challenger_candidate_removed_not_applied = 18,792,948

strict_recall_miss_total = 0
wrong_scope_proxy_total = 0
hard_negative_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0

meaning:
The runtime-facing apply surface is now the 16 direct canary winners only. The
larger challenger-prune opportunity is explicitly not applied until E136O
challenger/OOD evidence exists. This is still rollback-safe overlay evidence,
not destructive deletion of legacy operators.
```

E136N installs primary/secondary variant governance over the E136M overlay:

```text
operator_count = 34
variant_registry_row_count = 68
primary_variant_count = 34
secondary_variant_count = 34

primary_active_count = 16
primary_current_count = 11
primary_abstract_current_count = 7
secondary_rollback_count = 16
secondary_challenger_count = 11
secondary_lineage_hold_count = 7
retired_redundant_count = 0

retirement_lane_created_count = 16
retirement_candidate_count = 0
destructive_delete_count = 0
ambiguous_primary_operator_count = 0
missing_primary_operator_count = 0
orphan_secondary_count = 0

runtime_overlay_removed_activation_total = 3,450,257
challenger_candidate_removed_not_applied = 18,792,948
rollback_snapshot_count = 16
rollback_trigger_count = 0

strict_recall_miss_total = 0
wrong_scope_proxy_total = 0
hard_negative_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0

meaning:
All 34 operators now have exactly one primary variant and at least one
secondary variant. The 16 E136M overlay winners are primary-active; the legacy
versions are rollback secondaries. The 11 tightened candidates remain
secondary challengers and the 7 abstract kernels remain lineage-hold
secondaries. Nothing is retired or destructively deleted yet.
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
without stale route reuse or cross-thread contamination. E136A confirms scoped
assistant/text lenses and guards can be farmed from the local assistant-text
seed pack and promoted through the Orange mutation/prune/no-harm gate. E136B
confirms those assistant/text lenses and guards compose into bounded
schema-gated route stacks while overbroad chatbot, source hallucination, and
direct-write controls fail as intended.
E136C confirms a first quick polished text render layer over those routes, with
short deterministic outputs for greeting, summary, code, source-defer, JSON,
no-solve math, safety, comparison, translation, no-overwrite, rejected-response,
and outline cases.
E136D confirms the output side can hold committed text as a binary
OutputTextField, where the matrix is N x 8 bit cells and each row is one UTF-8
byte.
E136E confirms fixed observations can improve across idle ticks only through
explicit proposals checked by Agency; no new input is introduced.
E136F confirms that same mechanism on a 70-case heldout series: arithmetic
cases improve when a matching visible trace exists, while no-pocket controls
preserve safe output and unsupported guesses are rejected.
E136G confirms adaptive idle scheduling: each proposal carries a one-more-tick
recommendation, expected next-tick gain, reason, and progress marker, while
Agency decides whether to continue or stop.
E136H confirms existing-operator-first refinement over the 16 E132 math-text
and 18 E136A assistant-text operators: 16 labels verified, 11 triggers
tightened, 7 useful abstract kernels preserved, and no operator held for more
evidence.
E136I confirms an explicit supersession/output ledger over the E136H selected
variants: 27 replacement-ready variants, 16 direct runtime candidates, 11
tightened challenger-required replacements, 7 abstract lineage-required
kernels, 0 destructive drops, and projected output activation delta -482,637.
E136J confirms those selected variants under non-destructive shadow apply over
8,094 cycles and 33,153,024 replay rows: 22,243,205 shadow-pruned activations,
0 strict recall misses, 0 wrong-scope proxy calls, 0 hard negatives, and 0
direct Flow writes.
E136K confirms a rollback-safe non-destructive apply plan: 16 direct
canary-ready candidates, 11 challenger/OOD-required replacements, 7 abstract
lineage holds, 0 destructive applies, and 0 runtime mutations allowed now.
E136L confirms those 16 direct candidates pass runtime-canary
removal/replacement replay with 0 rollback triggers while the 11 challenger
rows and 7 abstract lineage rows remain held.
E136M materializes those 16 direct candidates as a runtime-facing overlay with
7 verified replacements, 9 light-prune overlays, 16 rollback snapshots, 0
destructive deletes, and 18,792,948 challenger candidate pruned activations
explicitly not applied.
E136N records the current operator set in a primary/secondary registry: 34
operators, 34 primaries, 34 secondaries, 16 primary-active overlays, 16 rollback
secondaries, 11 challenger secondaries, 7 lineage-hold secondaries, 0 retired
variants, and 0 destructive deletes.
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
It can farm scoped assistant/text request-shape and boundary operators from the
local E136 assistant-text seed pack.
It can compose those assistant/text operators into bounded route stacks in a
controlled assistant/text route-composition proxy.
It can render short polished deterministic assistant text for a 12-sample quick
set without raw route/action label leakage.
It can represent committed output text in an Agency-gated OutputTextField binary
matrix with roundtrip, reject, zero-fill, and tamper-detection checks.
It can improve some fixed observations during idle ticks through checked
proposals, while preserving source-defer/refusal controls and rejecting direct
OutputTextField writes.
It can reproduce that idle improvement behavior on a 70-case heldout series,
with no-pocket controls preserving safe output instead of inventing answers.
It can adapt idle tick count through explicit proposal continuation fields and
Agency-approved continuation/override decisions.
It can plan non-destructive operator supersession from selected variants,
including direct candidates, challenger-required candidates, abstract lineage
kernels, and projected output/prune impact.
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
docs/research/E136A_ASSISTANT_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md
docs/research/E136B_ASSISTANT_TEXT_ROUTE_COMPOSITION_AND_BOUNDARY_CONFIRM_RESULT.md
docs/research/E136C_ASSISTANT_TEXT_POLISHED_RENDER_QUICK_TEST_RESULT.md
docs/research/E136D_OUTPUT_TEXT_FIELD_BINARY_MATRIX_SMOKE_RESULT.md
docs/research/E136E_IDLE_THINK_TICK_PROPOSAL_REFINEMENT_SMOKE_RESULT.md
docs/research/E136F_IDLE_THINK_TICK_HELDOUT_SERIES_CONFIRM_RESULT.md
docs/research/E136G_ADAPTIVE_IDLE_TICK_BUDGET_CONFIRM_RESULT.md
docs/research/E136H_EXISTING_OPERATOR_REFINEMENT_MUTATION_PRUNE_NIGHT_CYCLE_RESULT.md
docs/research/E136I_OPERATOR_SUPERSESSION_AND_OUTPUT_LEDGER_PLANNING_RESULT.md
docs/research/E136J_SHADOW_VARIANT_APPLY_AND_RESIDUAL_PRUNE_CONFIRM_RESULT.md
docs/research/E136K_OPERATOR_REPLACEMENT_APPLY_PLAN_OR_FLOW_SCALE_TRANSFER_RESULT.md
docs/research/E136L_RUNTIME_REPLACEMENT_CANARY_AND_TIGHTENED_CHALLENGER_CONFIRM_RESULT.md
docs/research/E136M_RUNTIME_REPLACEMENT_APPLY_OR_ABSTRACT_LINEAGE_SPLIT_RESULT.md
docs/research/E136N_PRIMARY_SECONDARY_VARIANT_GOVERNANCE_RESULT.md
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
docs/research/artifact_samples/e136a_assistant_text_skill_farm_mutation_prune_orange_cycle/
docs/research/artifact_samples/e136b_assistant_text_route_composition_and_boundary_confirm/
docs/research/artifact_samples/e136c_assistant_text_polished_render_quick_test/
docs/research/artifact_samples/e136d_output_text_field_binary_matrix_smoke/
docs/research/artifact_samples/e136e_idle_think_tick_proposal_refinement_smoke/
docs/research/artifact_samples/e136f_idle_think_tick_heldout_series_confirm/
docs/research/artifact_samples/e136g_adaptive_idle_tick_budget_confirm/
docs/research/artifact_samples/e136h_existing_operator_refinement_mutation_prune_night_cycle/
docs/research/artifact_samples/e136i_operator_supersession_and_output_ledger_planning/
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
python scripts/tools/generate_operator_rank_dashboard.py --e127 docs/research/artifact_samples/e127_overnight_text_skill_farm_orange_cycle --e129 docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation --e130a docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet --e130b docs/research/artifact_samples/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet --e131 docs/research/artifact_samples/e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet --e132 docs/research/artifact_samples/e132_external_math_text_skill_farm_mutation_prune_orange_cycle --e133 docs/research/artifact_samples/e133_math_text_route_composition_and_no_solve_assistant_confirm --e134 docs/research/artifact_samples/e134_external_math_text_ood_route_stress_and_counterexample_gauntlet --e135 docs/research/artifact_samples/e135_math_text_multi_route_assistant_dialogue_state_gauntlet --e136a docs/research/artifact_samples/e136a_assistant_text_skill_farm_mutation_prune_orange_cycle --e136b docs/research/artifact_samples/e136b_assistant_text_route_composition_and_boundary_confirm --out target/pilot_wave/operator_rank_dashboard/index.html
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
E136A ASSISTANT TEXT = 18/18
E136A DATASET ROWS = 447,766
E136A SUPPORT MIN = 4,746
E136A HARD NEGATIVES = 0
E136A DIRECT WRITES = 0
E136B ROUTE COMP = 18/18
E136B ROUTE CASES = 144,000
E136B STACK = 100.000%
E136B BOUNDARY = 100.000%
E136B HARD NEGATIVES = 0
E136B DIRECT WRITES = 0
ORANGE/LEGENDARY CANDIDATE = 561
```

Expected E136C render smoke:

```text
E136C RENDER = 12/12
E136C JSON VALID = 2/2
E136C RAW ACTION LEAKS = 0
E136C DIRECT-WRITE CLAIMS = 0
```

Expected E136D output-field smoke:

```text
E136D OUTPUT TEXT FIELD = 10/10
E136D ROUNDTRIPS = 7/7
E136D GUARDED REJECTS = 3/3
E136D ZERO-FILL = 10/10
E136D TAMPER DETECT = 1/1
```

Expected E136E idle think-tick smoke:

```text
E136E CASES = 8/8
E136E PROPOSALS CHECKED = 10
E136E NEW INPUT = 0
E136E IMPROVEMENTS = 4
E136E NON-DEGRADATION = 8/8
E136E OUTPUT ROUNDTRIP = 8/8
```

Expected E136F idle think-tick heldout series:

```text
E136F CASES = 70/70
E136F ARITHMETIC IMPROVEMENTS = 36/36
E136F NO-POCKET PRESERVES = 6/6
E136F PROPOSALS CHECKED = 90
E136F NEW INPUT = 0
E136F DIRECT-WRITE REJECTS = 4
E136F UNSUPPORTED-CLAIM REJECTS = 6
E136F OUTPUT ROUNDTRIP = 70/70
```

Expected E136G adaptive idle tick budget:

```text
E136G CASES = 24/24
E136G ADAPTIVE TICKS = 33
E136G FIXED BASELINE TICKS = 120
E136G PROPOSAL CONTINUE FIELDS = 33/33
E136G AGENCY CONTINUE YES = 9
E136G AGENCY OVERRIDES = 2
E136G IMMEDIATE STOPS = 8
E136G CHAINED COMPLETE = 3/3
E136G DIRECT-WRITE REPAIRS = 3/3
E136G NO-POCKET STOPS = 4/4
E136G OUTPUT ROUNDTRIP = 24/24
```

Expected E136H existing operator refinement:

```text
E136H CYCLES = 40/40
E136H OPERATORS = 34
E136H CURRENT ACTIVATIONS = 3,373,788
E136H SELECTED ACTIVATIONS = 2,891,151
E136H PRUNED ACTIVATIONS = 482,637
E136H VERIFIED LABELS = 16
E136H TIGHTENED TRIGGERS = 11
E136H ABSTRACT KERNELS = 7
E136H HOLD FOR MORE EVIDENCE = 0
E136H HARD NEGATIVES = 0
E136H DIRECT FLOW WRITES = 0
```

Expected E136I operator supersession and output ledger planning:

```text
E136I OPERATORS = 34
E136I REPLACEMENT READY = 27
E136I DIRECT RUNTIME CANDIDATES = 16
E136I TIGHTENED CHALLENGER REQUIRED = 11
E136I ABSTRACT LINEAGE REQUIRED = 7
E136I HOLD FOR MORE EVIDENCE = 0
E136I DESTRUCTIVE DROPS = 0
E136I PROJECTED CURRENT ACTIVATIONS = 3,373,788
E136I PROJECTED SELECTED ACTIVATIONS = 2,891,151
E136I PROJECTED PRUNED ACTIVATIONS = 482,637
E136I PROJECTED OUTPUT ACTIVATION DELTA = -482,637
E136I ACCEPTED MUTATIONS = 96
E136I HARD NEGATIVES = 0
E136I DIRECT FLOW WRITES = 0
```

Expected E136J shadow variant apply and residual prune confirm:

```text
E136J STOP REASON = deadline
E136J CYCLES = 8,094
E136J ROWS PROCESSED = 33,153,024
E136J ELAPSED SECONDS = 46,317.709
E136J OPERATORS = 34
E136J REPLACEMENT READY = 27
E136J DIRECT RUNTIME CANDIDATES = 16
E136J TIGHTENED CHALLENGER REQUIRED = 11
E136J ABSTRACT LINEAGE REQUIRED = 7
E136J CURRENT ACTIVATIONS = 188,597,925
E136J SELECTED ACTIVATIONS = 166,354,720
E136J SHADOW PRUNED ACTIVATIONS = 22,243,205
E136J STRICT RECALL MISSES = 0
E136J WRONG SCOPE PROXY = 0
E136J HARD NEGATIVES = 0
E136J DIRECT FLOW WRITES = 0
```

Expected E136K operator replacement apply plan:

```text
E136K OPERATORS = 34
E136K DIRECT CANARY READY = 16
E136K CHALLENGER/OOD REQUIRED = 11
E136K ABSTRACT LINEAGE REQUIRED = 7
E136K RUNTIME MUTATION ALLOWED NOW = 0
E136K DESTRUCTIVE APPLIES = 0
E136K ROLLBACK MANIFEST = 16
E136K DIRECT CANARY PRUNE RATIO = 0.057159
E136K CHALLENGER PRUNE RATIO = 0.351548
E136K STRICT RECALL MISSES = 0
E136K WRONG SCOPE PROXY = 0
E136K HARD NEGATIVES = 0
E136K DIRECT FLOW WRITES = 0
```

Expected E136L runtime replacement canary:

```text
E136L OPERATORS = 34
E136L DIRECT CANARY TESTED = 16
E136L DIRECT CANARY PASSED = 16
E136L OLD OPERATORS REMOVED IN CANARY = 16
E136L RUNTIME REPLACEMENT CANARY ALLOWED = 16
E136L PRODUCTION RUNTIME APPLIES = 0
E136L DESTRUCTIVE APPLIES = 0
E136L CHALLENGER HOLD = 11
E136L ABSTRACT LINEAGE HOLD = 7
E136L ROLLBACK MANIFEST = 16
E136L ROLLBACK TRIGGERS = 0
E136L DIRECT CANARY REMOVED ACTIVATIONS = 3,450,257
E136L SAMPLE DIRECT REMOVED ACTIVATIONS = 1,031
E136L STRICT RECALL MISSES = 0
E136L WRONG SCOPE PROXY = 0
E136L HARD NEGATIVES = 0
E136L DIRECT FLOW WRITES = 0
```

Expected E136M runtime replacement overlay:

```text
E136M OPERATORS = 34
E136M RUNTIME OVERLAY ACTIVE = 16
E136M RUNTIME OVERLAY APPLY = 16
E136M VERIFIED REPLACEMENTS = 7
E136M LIGHT PRUNE OVERLAYS = 9
E136M LEGACY TRIGGERS DISABLED IN OVERLAY = 16
E136M LEGACY TRIGGERS RETAINED FOR ROLLBACK = 16
E136M CHALLENGER/OOD QUEUE = 11
E136M ABSTRACT LINEAGE SPLIT = 7
E136M ROLLBACK SNAPSHOTS = 16
E136M ROLLBACK TRIGGERS = 0
E136M PRODUCTION DESTRUCTIVE DELETES = 0
E136M RUNTIME OVERLAY REMOVED ACTIVATIONS = 3,450,257
E136M CHALLENGER CANDIDATE REMOVED NOT APPLIED = 18,792,948
E136M STRICT RECALL MISSES = 0
E136M WRONG SCOPE PROXY = 0
E136M HARD NEGATIVES = 0
E136M DIRECT FLOW WRITES = 0
```

Expected E136N primary/secondary variant governance:

```text
E136N OPERATORS = 34
E136N VARIANT REGISTRY ROWS = 68
E136N PRIMARY VARIANTS = 34
E136N SECONDARY VARIANTS = 34
E136N PRIMARY ACTIVE = 16
E136N PRIMARY CURRENT = 11
E136N PRIMARY ABSTRACT CURRENT = 7
E136N SECONDARY ROLLBACK = 16
E136N SECONDARY CHALLENGER = 11
E136N SECONDARY LINEAGE HOLD = 7
E136N RETIRED REDUNDANT = 0
E136N RETIREMENT LANE CREATED = 16
E136N RETIREMENT CANDIDATES = 0
E136N DESTRUCTIVE DELETES = 0
E136N AMBIGUOUS PRIMARY OPERATORS = 0
E136N MISSING PRIMARY OPERATORS = 0
E136N ORPHAN SECONDARIES = 0
E136N RUNTIME OVERLAY REMOVED ACTIVATIONS = 3,450,257
E136N CHALLENGER CANDIDATE REMOVED NOT APPLIED = 18,792,948
E136N ROLLBACK TRIGGERS = 0
E136N STRICT RECALL MISSES = 0
E136N WRONG SCOPE PROXY = 0
E136N HARD NEGATIVES = 0
E136N DIRECT FLOW WRITES = 0
```

## Local E136 Seed Pack

After E135, a local assistant/text seed pack was downloaded and normalized for
the LLM-ish assistant/operator farming direction:

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

The seed pack itself is local data readiness and remains under ignored
`target/`, but E136A has now consumed it for a scoped assistant/text
operator-farm confirmation and E136B has consumed the E136A operators for a
controlled route-composition/boundary confirmation. Raw and normalized data are
intentionally not committed; the committed proof surface is the E136A/E136B
runners, docs, dashboard integration, and tracked sample artifact packs.

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
python -m py_compile scripts/probes/run_e136a_assistant_text_skill_farm_mutation_prune_orange_cycle.py
python scripts/probes/run_e136a_assistant_text_skill_farm_mutation_prune_orange_cycle.py --dataset target/datasets/missing_e136a_smoke.jsonl --dataset-manifest target/datasets/missing_e136a_manifest.json --download-manifest target/datasets/missing_e136a_download.json --allow-builtin-dataset --dataset-row-limit 120 --min-assistant-support 1 --min-dataset-rows 1 --out target/ci/e136a_assistant_text_skill_farm_mutation_prune_orange_cycle --sample-out ""
python -m py_compile scripts/probes/run_e136b_assistant_text_route_composition_and_boundary_confirm.py
python scripts/probes/run_e136b_assistant_text_route_composition_and_boundary_confirm.py --dataset target/datasets/missing_e136b_smoke.jsonl --dataset-manifest target/datasets/missing_e136b_manifest.json --download-manifest target/datasets/missing_e136b_download.json --allow-builtin-dataset --dataset-row-limit 120 --min-dataset-rows 1 --route-cases-per-operator 40 --boundary-cases-per-operator 12 --control-cases-per-operator 8 --out target/ci/e136b_assistant_text_route_composition_and_boundary_confirm --sample-out ""
python -m py_compile scripts/probes/run_e136c_assistant_text_polished_render_quick_test.py
python scripts/probes/run_e136c_assistant_text_polished_render_quick_test.py --out target/ci/e136c_assistant_text_polished_render_quick_test --sample-out ""
python -m py_compile scripts/probes/run_e136d_output_text_field_binary_matrix_smoke.py
python scripts/probes/run_e136d_output_text_field_binary_matrix_smoke.py --out target/ci/e136d_output_text_field_binary_matrix_smoke --sample-out ""
cargo test -p vraxion-runtime output_text_field
python -m py_compile scripts/probes/run_e136e_idle_think_tick_proposal_refinement_smoke.py
python scripts/probes/run_e136e_idle_think_tick_proposal_refinement_smoke.py --out target/ci/e136e_idle_think_tick_proposal_refinement_smoke --sample-out ""
python -m py_compile scripts/probes/run_e136f_idle_think_tick_heldout_series_confirm.py
python scripts/probes/run_e136f_idle_think_tick_heldout_series_confirm.py --out target/ci/e136f_idle_think_tick_heldout_series_confirm --sample-out ""
python -m py_compile scripts/probes/run_e136g_adaptive_idle_tick_budget_confirm.py
python scripts/probes/run_e136g_adaptive_idle_tick_budget_confirm.py --out target/ci/e136g_adaptive_idle_tick_budget_confirm --sample-out ""
python -m py_compile scripts/probes/run_e136h_existing_operator_refinement_mutation_prune_night_cycle.py
python scripts/probes/run_e136h_existing_operator_refinement_mutation_prune_night_cycle.py --cycles 2 --e132-rows-per-cycle 200 --e136a-rows-per-cycle 400 --out target/ci/e136h_existing_operator_refinement_mutation_prune_night_cycle --sample-out ""
python -m py_compile scripts/probes/run_e136i_operator_supersession_and_output_ledger_planning.py
python scripts/probes/run_e136i_operator_supersession_and_output_ledger_planning.py --out target/ci/e136i_operator_supersession_and_output_ledger_planning --sample-out ""
python -m py_compile scripts/probes/run_e136n_primary_secondary_variant_governance.py
python scripts/probes/run_e136n_primary_secondary_variant_governance.py --out target/ci/e136n_primary_secondary_variant_governance --sample-out ""
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
1. Run E136O challenger/OOD runtime replacement gauntlet before applying the
   11 tightened-trigger hold rows or doing broad new-operator farming.
2. Run a later assistant/open-domain boundary probe before claiming broader
   assistant readiness.
3. Decide whether to continue E127 with a fresh candidate pack or pause farming.
4. Build the next bridge from deterministic action/template rendering,
   E132/E133/E134/E135 math-text lenses/guards/routes/OOD/dialogue-state,
   E136A assistant/text lenses/guards, and exact arithmetic trace/text-IO
   operators toward richer assistant arithmetic while keeping claims scoped.
5. Keep all new skills scoped, dashboard-visible, and rollback-safe.
```
