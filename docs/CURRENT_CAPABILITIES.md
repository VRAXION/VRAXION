# VRAXION Current Capabilities

_Last updated: 2026-06-21_

## One-Line Status

```text
VRAXION is currently a governed Operator runtime with scoped,
evidence-backed mechanics. It is not an open-domain LLM/chatbot.
```

## Runtime Shape

The public architecture claim is:

```text
observation
-> scoped Operator / Prismion evaluation
-> proposal field
-> Agency / governance decision
-> commit, reject, defer, ask, or render a bounded response
```

Operators do not directly overwrite stable state. They produce proposals.
Agency/governance decides whether a proposal may become committed state or
output.

## Demonstrated Public Capability Classes

The public evidence covers these scoped classes:

```text
visible arithmetic trace validation
text evidence cue routing
conflict / missing-evidence defer behavior
proposal and agency commit gating
output byte-field roundtrip checks
idle-tick proposal refinement under no-new-input controls
atomic multiwrite preview/apply guard checks
```

These are bounded mechanics, not a claim that the system understands arbitrary
natural language or solves arbitrary tasks.

## Current Non-Claims

The public repository does not claim:

```text
general language understanding
freeform LLM-style generation
production assistant readiness
internet-scale training
autonomous code evolution
PermaCore / TrueGolden memory
medical, legal, or safety-critical deployment readiness
```

## Public/Private Split

The full frontier training surface is not public. Public artifacts should be
small enough to audit and should avoid:

```text
raw datasets
full run traces
private paths
hidden oracle cells
exact frontier mutation recipes
operational dataset downloaders
```

The next intended public capability step is a clean, compileable Rust package
exported from the private α-Sync core after release-blocker review.
