# What Are We Really Testing?

Status: research direction note. This is not a product spec and not a
consciousness claim.

## Short Answer

The current Pilot/Prismion line is not testing whether a model can output the
right answer directly.

It is testing whether a system can maintain a protected decision path:

```text
input claim
-> scope/evidence decomposition
-> lexical trace
-> executable state
-> inhibition / interference
-> provisional pilot state
-> guard / hard commit
-> action
-> memory trace
```

The central question is:

```text
Can the system keep "what was mentioned" separate from
"what is allowed to execute"?
```

## Why Positive Evidence Fails

Pure positive evidence treats every cue as support:

```text
"add"                  -> ADD evidence
"do not add"           -> ADD evidence
"the word add appears" -> ADD evidence
```

That is unsafe. It cannot represent:

- negation,
- mention without command,
- weak/uncertain instruction,
- correction/refocus,
- delayed commit.

This is why keyword and raw n-gram sensors overfire. They detect cue presence,
not action authority.

## Why Phase / Interference Is Promising

Negation is not just absence of evidence. It is active opposition:

```text
ADD + NOT_ADD -> cancel ADD execution
```

Correction is not just more evidence. It is refocus:

```text
ADD actually MUL -> MUL gets authority, stale ADD loses authority
```

Weak or ambiguous evidence is not a weaker answer. It is a HOLD condition:

```text
maybe ADD -> HOLD
ADD or MUL -> HOLD
```

A phase-like representation can model cancellation more naturally than a
positive-only evidence vector.

## Lexical Trace Is Not Executable State

The system may need to remember that a cue was seen while also blocking it from
execution:

```text
"the word add appears"

lexical trace:
  ADD was seen

executable state:
  ADD must not execute
```

This distinction matters for memory and audit. If the system deletes the cue
entirely, it loses trace. If it keeps the cue as executable evidence, it may act
wrongly.

## ClaimTrace Is Not ConstraintTrace

External input should first become a claim:

```text
raw command -> ClaimTrace
```

It should become a constraint only after adoption:

```text
ClaimTrace
-> source / authority check
-> why / utility check
-> adoption gate
-> ConstraintTrace
```

For example:

```text
"do not add"
```

is not automatically truth. It is a claim that may become an adopted constraint
if the source, context, and policy justify it.

## WhyGate / AdoptionGate

A conscious-like pilot should not obey raw input by default.

It needs a gate that asks:

- Who said this?
- Why should this matter?
- Who benefits?
- Who is harmed?
- Is the cost justified?
- Does it preserve the long-term project path?
- Should the system accept, reject, negotiate, or ask for proof?

This is a later layer. The immediate probes only test the smaller primitive:

```text
can adopted constraints inhibit executable state?
```

## ProjectContinuityGuard

Some actions may feel locally justified but destroy the long-term rescue path.

The guard shape is:

```text
block local action
if it destroys higher expected long-term mission continuity
```

This should remain a guarded policy layer, not a direct reflex.

## Prismion Hypothesis

`Prismion` here means a prism-like neural unit or cell, not a magic component.
The name comes from:

```text
prisma + neuron
```

The narrow hypothesis is:

```text
phase / basis / interference primitives may represent cancellation,
trace/action separation, and refocus more naturally than a positive-only neuron.
```

What has not worked:

```text
replace every neuron with a Prismion-style hidden unit and expect the whole
factor-composition problem to disappear.
```

The useful direction is staged:

1. deterministic interference probe,
2. lexical/executable basis separation,
3. scope release / decay,
4. learned unit expressivity,
5. small stateful PrismionCell,
6. only then larger pilot/controller integration.

## Next Test Ladder

```text
PILOT_WAVE_INTERFERENCE_001
  Does phase-like cancellation beat positive evidence?

PILOT_WAVE_BASIS_SEPARATION_001
  Can lexical trace persist while executable authority is blocked?

PILOT_WAVE_BASIS_SCOPE_RELEASE_001
  Can stale HOLD/negation residue decay when a fresh valid cue arrives?

PRISMION_UNIT_BENCH_001
  Does a Prismion-like unit solve cancellation/XOR primitives more naturally
  than a single ordinary neuron?

PRISMION_CELL_SEQUENCE_001
  Can a learned stateful cell handle cue order, scope, decay, and refocus?
```

## Claim Boundary

This line does not prove:

- consciousness,
- general natural-language understanding,
- quantum physics,
- full VRAXION behavior,
- production readiness.

It tests small computational primitives for guarded agency.
