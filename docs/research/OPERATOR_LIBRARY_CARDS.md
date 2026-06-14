# Operator Library Cards

Current source of truth:

```text
E88_LOCAL_GOLDEN_AND_SUPPORT_COMPONENT_SURVIVAL_GAUNTLET
decision = e88_local_golden_survival_gauntlet_confirmed
```

Naming lock:

```text
canonical generic term = Operator
legacy alias = Pocket
```

Boundary:

```text
These cards describe the current scoped visible calculation-trace Operator
family. They are not open-domain reasoning skills and not Core / TrueGolden
claims.
```

## Active Survivor Cards

### CALC-SCRIBE

```text
component_id = calc_scribe_v003
status       = SpecialistGoldenCandidate
type         = governed Operator artifact
family       = Scribe
scope        = visible_calc_trace_validator
```

Short description:

```text
Validates visible arithmetic trace markers.
```

What it does:

```text
reads visible calc trace forms
checks whether expression=result is mathematically valid
emits governed validation/rejection proposals
```

What it does not do:

```text
does not solve GSM8K word problems
does not infer hidden answers
does not reason over arbitrary natural language
does not directly write Flow/Ground state
```

Survival result:

```text
passed reload/import stress
passed negative-scope stress
passed long-horizon no-harm replay
no challenger replaced it
```

### Native Trace Anchor

```text
component_id = calc_scribe_native_seed
status       = LocalGoldenConfirmed
type         = seed / native trace support Operator
family       = Lens/Scribe support
```

Short description:

```text
Handles the native <<expr=result>> trace form.
```

What it does:

```text
recognizes the canonical angle-bracket trace interface
anchors the CALC-SCRIBE route for native visible markers
```

If removed:

```text
mean_action_loss = 0.392120
false_call_delta = 0.000000
false_commit_delta = 0.000000
```

Meaning:

```text
This is the strongest support fragment. It is not redundant.
```

### Square Trace Lens

```text
component_id = square_trace_adapter
status       = StableSupport
type         = format Adapter Operator
family       = Lens/Adapter
```

Short description:

```text
Adapts [calc expr=result] into the internal calc-trace route.
```

What it does:

```text
reads square-bracket visible trace forms
normalizes them toward the CALC-SCRIBE-compatible trace interface
```

If removed:

```text
mean_action_loss = 0.203491
false_call_delta = 0.000000
false_commit_delta = 0.000000
```

Meaning:

```text
Useful format support. Not standalone Golden.
```

### Arrow Trace Lens

```text
component_id = arrow_trace_adapter
status       = StableSupport
type         = format Adapter Operator
family       = Lens/Adapter
```

Short description:

```text
Adapts calc: expr -> result and trace: expr -> result forms.
```

What it does:

```text
reads arrow-style visible calc traces
maps arrow output into expression=result form
```

If removed:

```text
mean_action_loss = 0.097618
false_call_delta = 0.000000
false_commit_delta = 0.000000
```

Meaning:

```text
Useful transfer adapter for arrow trace notation.
```

### Plain Equation Lens

```text
component_id = standalone_plain_trace_adapter
status       = StableSupport
type         = format Adapter Operator
family       = Lens/Adapter
```

Short description:

```text
Handles isolated plain expr = result trace lines.
```

What it does:

```text
recognizes standalone equation lines
routes them to CALC-SCRIBE when they are scoped as visible trace evidence
```

If removed:

```text
mean_action_loss = 0.196648
false_call_delta = 0.000000
false_commit_delta = 0.000000
```

Meaning:

```text
Strong support, but must stay paired with scope hygiene.
```

### Operator Glyph Grounder

```text
component_id = unicode_operator_normalizer
status       = StableSupport
type         = notation grounding / ingress Adapter Operator
family       = α-Syncer
ascii_family = alpha_syncer
```

Short description:

```text
Grounds external operator glyphs into the internal calc notation.
```

What it does:

```text
× -> *
÷ -> /
− -> -
```

Why this is grounding:

```text
external symbol form
-> internal canonical operation token
-> CALC-SCRIBE-readable trace language
```

If removed:

```text
mean_action_loss = 0.072132
false_call_delta = 0.000000
false_commit_delta = 0.000000
```

Meaning:

```text
Small but real notation-grounding Operator.
```

### False Trace Rejector

```text
component_id = invalid_trace_rejector
status       = StableSupport
type         = safety / commit hygiene Operator
family       = Guard
```

Short description:

```text
Rejects visible calc traces whose arithmetic is wrong.
```

What it does:

```text
detects invalid expression=result claims
prevents wrong traces from becoming committed facts
```

If removed:

```text
mean_action_loss = 0.212696
false_call_delta = 0.000000
false_commit_delta = 0.215370
```

Meaning:

```text
Critical safety support. Removing it creates false commits.
```

### Long Text Scope Shield

```text
component_id = long_text_scope_guard
status       = BundleSupport
type         = scope Guard Operator
family       = Guard
```

Short description:

```text
Blocks accidental calc calls from long natural text and inactive examples.
```

What it does:

```text
guards against random numbers and quote-like calc snippets in prose
keeps CALC-SCRIBE scoped to visible active trace evidence
```

If removed:

```text
mean_action_loss = 0.002865
false_call_delta = 0.052818
false_commit_delta = 0.000000
```

Meaning:

```text
Low-frequency but important safety Guard.
It is not standalone Golden; it survives as bundle support.
```

## Non-Main / Rejected Cards

### Native Echo Clone

```text
component_id = native_seed_clone
status       = Redundant
```

Description:

```text
Duplicate native trace capability. Clean but unnecessary.
```

### Square Echo Clone

```text
component_id = square_adapter_clone
status       = Redundant
```

Description:

```text
Duplicate square trace adapter. Adds cost without unique value.
```

### Arrow Echo Clone

```text
component_id = arrow_adapter_clone
status       = Redundant
```

Description:

```text
Duplicate arrow trace adapter. Adds cost without unique value.
```

### Numeric Mirage

```text
component_id = numeric_alias_overreach
status       = Quarantine
```

Description:

```text
Unsafe overreach that treats numeric-looking text as callable evidence.
```

Failure mode:

```text
false calls on natural text with numbers
```

### Blind Full-Scan

```text
component_id = full_library_scan_overreach
status       = Quarantine
```

Description:

```text
Unsafe full-library scan behavior.
```

Failure mode:

```text
false_call_max = 1.000000
false_commit_max = 0.333374
```

### Direct Commit Hazard

```text
component_id = invalid_direct_commit
status       = Banned
```

Description:

```text
Commits invalid traces instead of rejecting them.
```

Failure mode:

```text
unsafe false commit path
```

### Long Text Overreach

```text
component_id = long_text_plain_overreach
status       = Quarantine
```

Description:

```text
Over-reads plain equation-like text inside long prose.
```

Failure mode:

```text
false calls on inactive examples and narrative text
```

### Passive Trace Observer

```text
component_id = noop_trace_observer
status       = Deprecated
```

Description:

```text
No measurable unique contribution.
```

### Expensive Debug Probe

```text
component_id = expensive_debug_probe
status       = Deprecated
```

Description:

```text
No useful contribution under E88, and too costly to keep active.
```

## Current Operator Library Summary

```text
survivors / usable main path:
  8

non-main / rejected:
  9

highest current status:
  calc_scribe_v003 -> SpecialistGoldenCandidate

support layer:
  5 StableSupport
  1 BundleSupport

blocked unsafe controls:
  3 Quarantine
  1 Banned
```

## E90 Text-Evidence Operator Candidates

Source:

```text
E90_OPERATOR_CURRICULUM_EXPANSION
decision = e90_operator_curriculum_expansion_confirmed
```

Boundary:

```text
These are controlled visible text-evidence Operator candidates.
They are not open-domain language understanding and not Core / TrueGolden.
```

### Visible Claim Binding alpha-Syncer

```text
component_id = visible_claim_binding_alpha_syncer
status       = StableOperatorCandidate
family       = alpha_syncer
```

What it does:

```text
"A means B" -> canonical binding proposal
```

If removed:

```text
mean_resolution_loss = 0.519119
mean_false_ask_delta = 0.800921
```

### Numeric Value Binding alpha-Syncer

```text
component_id = numeric_value_binding_alpha_syncer
status       = StableOperatorCandidate
family       = alpha_syncer
```

What it does:

```text
"A is N" -> canonical value-binding proposal
```

If removed:

```text
mean_resolution_loss = 0.146332
mean_false_ask_delta = 0.396338
```

### Temporal Rule-Shift T-Stab

```text
component_id = temporal_rule_shift_t_stab
status       = StableOperatorCandidate
family       = T-Stab
```

What it does:

```text
confirmed post-marker rule change -> new stable binding
```

If removed:

```text
mean_resolution_loss = 0.074432
mean_false_ask_delta = 0.201541
```

### False-Alarm Temporal T-Stab

```text
component_id = false_alarm_temporal_t_stab
status       = StableOperatorCandidate
family       = T-Stab
```

What it does:

```text
visible false alarm -> preserve the prior stable binding
```

If removed:

```text
mean_resolution_loss = 0.074371
mean_false_ask_delta = 0.201392
```

### Revoked Binding Guard

```text
component_id = revoked_binding_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
"A no longer means B" -> block stale answer commit
```

If removed:

```text
mean_resolution_loss = 0.074997
```

### Contradiction Guard

```text
component_id = contradiction_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
same-cycle conflicting claims -> reject contradiction
```

If removed:

```text
mean_resolution_loss = 0.074219
mean_wrong_confident_delta = 0.117652
```

### Unresolved-State Information Guard

```text
component_id = unresolved_state_info_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
missing/unproven evidence -> ask/search/hold instead of guessing
```

If removed:

```text
mean_resolution_loss = 0.556519
```

### Inactive Quote Scope Guard

```text
component_id = inactive_quote_scope_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
quoted/archive/example claims are not active evidence
```

If removed:

```text
mean_resolution_loss = 0.407379
```

### Evidence Span Lens

```text
component_id = evidence_span_lens
status       = StableOperatorCandidate
family       = Lens
```

What it does:

```text
preserves visible byte-span references for proposal evidence
```

If removed:

```text
mean_resolution_loss = 1.000000
mean_trace_validity_delta = -1.000000
```

### Canonical Answer Scribe

```text
component_id = canonical_answer_scribe
status       = StableOperatorCandidate
family       = Scribe
```

What it does:

```text
resolved canonical state -> external answer action
```

If removed:

```text
mean_resolution_loss = 0.369263
mean_false_ask_delta = 1.000000
```

### E90 Rejected Controls

```text
stale_binding_committer       -> Quarantine
inactive_quote_overreach      -> Quarantine
marker_only_shift_shortcut    -> Quarantine
answer_without_span_shortcut  -> Quarantine
full_text_scan_overreach      -> Quarantine
always_ask_control            -> Deprecated
passive_text_observer         -> Deprecated
claim_binding_clone           -> Redundant
```

## E91 Temporal Stream T-Stab Candidates

Source:

```text
E91_T_STAB_TEMPORAL_STREAM_EXPANSION
decision = e91_t_stab_temporal_stream_expansion_confirmed
```

Boundary:

```text
These are controlled temporal stream stabilization Operators.
They are not raw production bitstream handling and not Core / TrueGolden.
```

### Frame Sequence T-Stab

```text
component_id = frame_sequence_t_stab
status       = StableOperatorCandidate
family       = T-Stab
```

What it does:

```text
orders temporal frames by sequence/cycle
```

If removed:

```text
mean_stabilization_loss = 0.458598
mean_false_hold_delta = 0.498650
```

### CRC-Parity Frame Guard

```text
component_id = crc_parity_frame_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
rejects corrupt frames before commit
```

If removed:

```text
mean_stabilization_loss = 0.919670
mean_false_hold_delta = 1.000000
```

### Bit-Slip Resync T-Stab

```text
component_id = bit_slip_resync_t_stab
status       = StableOperatorCandidate
family       = T-Stab
```

What it does:

```text
finds valid frame start after offset/noise slip
```

If removed:

```text
mean_stabilization_loss = 0.081271
mean_false_hold_delta = 0.088374
```

### Repeat-Vote Stabilizer T-Stab

```text
component_id = repeat_vote_stabilizer_t_stab
status       = StableOperatorCandidate
family       = T-Stab
```

What it does:

```text
uses repeated frames to stabilize noisy payload bits
```

If removed:

```text
mean_stabilization_loss = 0.081235
mean_false_hold_delta = 0.088330
```

### Stale Replay Guard

```text
component_id = stale_replay_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
blocks old-cycle frames replayed as current evidence
```

If removed:

```text
mean_stabilization_loss = 0.220198
mean_false_hold_delta = 0.239432
```

### Source Trust Guard

```text
component_id = source_trust_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
prefers verified frames over rumor/untrusted frames
```

If removed:

```text
mean_stabilization_loss = 0.220198
mean_false_hold_delta = 0.239432
```

### Delayed Evidence Buffer Lens

```text
component_id = delayed_evidence_buffer_lens
status       = StableOperatorCandidate
family       = Lens
```

What it does:

```text
holds partial streams until required frames are visible
```

If removed:

```text
mean_stabilization_loss = 0.080330
```

### Temporal Commit Scribe

```text
component_id = temporal_commit_scribe
status       = StableOperatorCandidate
family       = Scribe
```

What it does:

```text
renders stabilized temporal state into answer/hold action
```

If removed:

```text
mean_stabilization_loss = 1.000000
mean_false_hold_delta = 1.000000
```

### E91 Rejected Controls

```text
first_frame_committer       -> Quarantine
no_crc_acceptor             -> Quarantine
stale_replay_committer      -> Quarantine
rumor_over_trust_committer  -> Quarantine
full_stream_overreach       -> Quarantine
always_hold_control         -> Deprecated
sequence_clone              -> Redundant
```

## E92 Alpha-Sync Lexical/Glyph Expansion Cards

Source:

```text
E92_ALPHA_SYNC_LEXICAL_GLYPH_EXPANSION
decision = e92_alpha_sync_lexical_glyph_expansion_confirmed
```

Boundary:

```text
visible lexical/glyph/unit normalization only
not open-domain language understanding
```

### Lexical Alias alpha-Syncer

```text
component_id = lexical_alias_alpha_syncer
status       = StableOperatorCandidate
family       = alpha_syncer
```

What it does:

```text
maps visible aliases/synonyms to canonical lexeme proposals
```

If removed:

```text
mean_resolution_loss = 0.343017
mean_false_hold_delta = 0.343017
```

### Negation Marker alpha-Syncer

```text
component_id = negation_marker_alpha_syncer
status       = StableOperatorCandidate
family       = alpha_syncer
```

What it does:

```text
grounds visible not/no/never markers so denied claims do not commit
```

If removed:

```text
mean_resolution_loss = 0.112229
mean_false_hold_delta = 0.112228
```

### Unit-Code alpha-Syncer

```text
component_id = unit_code_alpha_syncer
status       = StableOperatorCandidate
family       = alpha_syncer
```

What it does:

```text
normalizes visible unit/code expressions into canonical unit values
```

If removed:

```text
mean_resolution_loss = 0.107182
mean_false_hold_delta = 0.107182
```

### Multilingual Surface alpha-Syncer

```text
component_id = multilingual_surface_alpha_syncer
status       = StableOperatorCandidate
family       = alpha_syncer
```

What it does:

```text
maps visible multilingual surface forms into canonical lexemes
```

If removed:

```text
mean_resolution_loss = 0.108481
mean_false_hold_delta = 0.108481
```

### Case-Morphology alpha-Syncer

```text
component_id = case_morphology_alpha_syncer
status       = StableOperatorCandidate
family       = alpha_syncer
```

What it does:

```text
normalizes visible plural/case/morphology variants into canonical lexemes
```

If removed:

```text
mean_resolution_loss = 0.114821
mean_false_hold_delta = 0.114821
```

### Symbol Equivalence Guard

```text
component_id = symbol_equivalence_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
blocks unsafe glyph equivalence unless visible equivalence evidence supports it
```

If removed:

```text
mean_resolution_loss = 0.214492
mean_false_hold_delta = 0.101957
```

### Alias Scope Guard

```text
component_id = alias_scope_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
keeps local, quoted, or archived aliases from leaking into active scope
```

If removed:

```text
mean_resolution_loss = 0.225929
mean_false_hold_delta = 0.113922
```

### Canonical Lexeme Scribe

```text
component_id = canonical_lexeme_scribe
status       = StableOperatorCandidate
family       = Scribe
```

What it does:

```text
renders the resolved canonical lexeme or unit as the external answer action
```

If removed:

```text
mean_resolution_loss = 1.000000
mean_false_hold_delta = 0.775458
```

### E92 Rejected Controls

```text
surface_string_matcher_shortcut -> Quarantine
negation_ignoring_committer     -> Quarantine
unitless_value_committer        -> Quarantine
global_alias_overreach          -> Quarantine
glyph_similarity_overreach      -> Quarantine
always_defer_control            -> Deprecated
lexical_alias_clone             -> Redundant
```

## E93 Agency/Guard Commit-Safety Expansion Cards

Source:

```text
E93_AGENCY_GUARD_COMMIT_SAFETY_EXPANSION
decision = e93_agency_guard_commit_safety_expansion_confirmed
```

Boundary:

```text
Proposal Field -> Agency -> Flow/Ground commit safety only
direct Flow write remains disallowed
not open-domain model behavior
```

### Proposal Collision Guard

```text
component_id = proposal_collision_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
detects conflicting same-target proposals before commit
```

If removed:

```text
mean_commit_safety_loss = 0.105937
mean_missed_commit_delta = 0.000000
```

### Ground Conflict Guard

```text
component_id = ground_conflict_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
blocks proposals that contradict stable Ground without revocation evidence
```

If removed:

```text
mean_commit_safety_loss = 0.113759
mean_missed_commit_delta = 0.000000
```

### Evidence Recency Guard

```text
component_id = evidence_recency_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
prefers newer verified evidence over older active evidence
```

If removed:

```text
mean_commit_safety_loss = 0.113739
mean_missed_commit_delta = 0.113739
```

### Trace Dependency Coverage Guard

```text
component_id = trace_dependency_coverage_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
requires every proposal dependency to have visible trace support
```

If removed:

```text
mean_commit_safety_loss = 0.453563
mean_missed_commit_delta = 0.336358
```

### Cycle Freshness Guard

```text
component_id = cycle_freshness_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
rejects stale-cycle proposals replayed into the current decision cycle
```

If removed:

```text
mean_commit_safety_loss = 0.213919
mean_missed_commit_delta = 0.105245
```

### Local Scope Exit T-Stab

```text
component_id = local_scope_exit_t_stab
status       = StableOperatorCandidate
family       = T-Stab
```

What it does:

```text
stabilizes cleanup of local bindings after their scope ends
```

If removed:

```text
mean_commit_safety_loss = 0.108350
mean_missed_commit_delta = 0.000000
```

### Agency Commit Quorum Guard

```text
component_id = agency_commit_quorum_guard
status       = StableOperatorCandidate
family       = Guard
```

What it does:

```text
requires compatible evidence quorum for high-risk commits
```

If removed:

```text
mean_commit_safety_loss = 0.227090
mean_missed_commit_delta = 0.117374
```

### Safe Commit Action Scribe

```text
component_id = safe_commit_action_scribe
status       = StableOperatorCandidate
family       = Scribe
```

What it does:

```text
renders COMMIT, REJECT, DEFER, and ASK actions after guard checks
```

If removed:

```text
mean_commit_safety_loss = 1.000000
mean_missed_commit_delta = 0.336358
```

### E93 Rejected Controls

```text
first_proposal_committer          -> Quarantine
majority_without_trace_committer  -> Quarantine
stale_cycle_committer             -> Quarantine
ground_overwrite_committer        -> Quarantine
local_scope_leak_committer        -> Quarantine
always_reject_control             -> Deprecated
quorum_guard_clone                -> Redundant
```
