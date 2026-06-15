# VRAXION Current Capabilities

_Last updated: 2026-06-15_

This document describes what the current VRAXION mainline can and cannot claim
as a single system.

## One-line status

```text
VRAXION is currently a governed Pocket/Operator runtime with scoped,
evidence-backed skills. It is not an open-domain LLM/chatbot.
```

## What the system is

The current active object is a governed runtime:

```text
Input / observations
-> scoped Operators / Pockets
-> Proposal Field
-> Agency / governance checks
-> commit, reject, defer, ask/search, or render a scoped response
```

Operators do not directly overwrite stable state. They write proposals. The
Agency/governance layer decides whether a proposal can become committed state or
output.

## Current library scale

Current E127 checkpointed evidence:

```text
Dashboard operator count = 529
Orange/Legendary scoped operators = 382
E127 cycles = 40
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
```

Current E128 assistant text-IO bridge evidence:

```text
prompt corpus = 320
train / validation / heldout = 160 / 64 / 96
train action accuracy = 1.000
validation action accuracy = 1.000
heldout action accuracy = 1.000
operator trace validity = 1.000
unsupported answers = 0
wrong refusals = 0
boundary-claim violations = 0
```

Current E129 arithmetic trace evidence:

```text
Orange/Legendary arithmetic operators = 9
qualified activation total = 2700000
qualified activation min/operator = 300000
negative-scope no-call cases = 9000
min in-scope accuracy = 1.000
negative-scope pass rate = 1.000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
```

These are scoped operators, not general-purpose neural skills. A larger count is
not automatically better; value depends on safe activation, low cost, correct
scope, reloadability, and no-harm evidence.

## What it can do now

### Governed Pocket / Operator Library

- store small scoped skills as governed Operator/Pocket artifacts;
- resolve operators through registry/token/digest governance instead of file
  names alone;
- block unsafe lifecycle states, digest mismatches, token swaps, and wrong-scope
  loads;
- rank, promote, quarantine, deprecate, and stress-test scoped operators.

### Evidence-first state handling

- distinguish committed evidence from unresolved or conflicting evidence;
- reject unsupported answers;
- defer when visible evidence is insufficient;
- handle stale state, contradiction, missing dependency, and turn-continuity
  guard cases in controlled evidence tasks.

### Proposal + Agency commit boundary

- collect operator proposals in a temporary proposal surface;
- require Agency/governance approval before stable Flow/Ground updates;
- reject stale, toxic, unsupported, wrong-scope, or conflicting proposals.

### Scoped text-evidence behavior

- run deterministic operator selection plus guarded template rendering over
  short prompts;
- build a local, auditable assistant-style prompt corpus from E127 artifacts,
  repo docs, adversarial boundary prompts, and FineWeb-derived local samples;
- select among scoped response actions such as answer, ask/defer, refuse with
  boundary, summarize, diagnose boundary, and next action in controlled smoke
  tests;
- produce proto-assistant-style responses in controlled cases such as evidence
  conflict, missing support, answerability, and citation/trace-style output;
- avoid claiming a stable answer when the scoped evidence chain is incomplete.

### Visible calculation-trace validation

- validate visible calculation markers such as `<<expression=result>>`;
- normalize arithmetic notation and common operator variants;
- validate arithmetic trace markers without claiming open-domain math reasoning.

### Exact arithmetic trace operators

- compute or validate scoped arithmetic expressions and traces;
- cover plus/minus, multiplication, exact division, floor division, signed
  integers, decimal/fraction rendering, mixed precedence, invalid-trace
  rejection, and division-by-zero rejection;
- no-call natural-language word problems that lack a visible arithmetic
  expression or trace.

### Task/progress integrity

- track required task steps and required evidence;
- block premature completion when a required step lacks evidence;
- mark blocker/waiting states in controlled task-progress probes.

## Common operator families found so far

The system tends to mine small mechanical skills first:

```text
normalizers
  map surface variants into internal forms, e.g. unicode operators

evidence lenses
  detect relevant evidence spans or evidence availability

conflict guards
  block or defer when evidence contradicts itself

missing-dependency guards
  require more information when a needed support link is absent

temporal stabilizers
  prefer latest valid state over stale state

answerability guards
  decide whether a question is answerable under visible evidence

trace / citation scribes
  preserve evidence links and render scoped support

negative-scope guards
  prevent a specialist skill from being used outside its allowed scope

progress-completion guards
  prevent false DONE / COMPLETE states

calc-trace validators
  check visible arithmetic trace markers
```

## What it cannot claim yet

VRAXION currently does **not** claim:

- open-domain chatbot behavior;
- Gemma/GPT-level generation;
- natural-language world-model understanding;
- freeform long-answer assistant quality;
- neural LLM training completion;
- GSM8K or hidden word-problem solving;
- open-domain natural-language arithmetic reasoning;
- production API readiness;
- deployed service readiness;
- final training completion;
- trained general model weights;
- PermaCore / TrueGolden memory;
- consciousness, sentience, or subjective experience.

## How to interpret a response

If the current system is asked a controlled, scoped question, the strongest
expected behavior is not "chat freely." It is closer to:

```text
visible evidence exists and scope matches -> commit/render scoped answer
visible evidence conflicts -> reject/defer and request resolution
visible evidence is missing -> ask/search/hold unresolved
specialist scope does not match -> do not call that specialist
```

That is the current niche: guarded, evidence-first operator composition, not
unbounded language generation.
