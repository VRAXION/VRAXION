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
