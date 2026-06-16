# VRAXION Current Capabilities

_Last updated: 2026-06-16_

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

Current dashboard/rank state after E136B:

```text
Dashboard operator count = 564
Orange/Legendary scoped operators = 561
CoreMemoryCandidate operators = 0
Deprecated operators = 3
```

Current E127 checkpointed evidence:

```text
E127 cycles = 40
E127 Orange/Legendary scoped text operators = 382
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

Current E130A rank-backfill evidence:

```text
backfilled operators = 136
Orange/Legendary backfill promotions = 136
qualified activation before total = 13877699
qualified activation add total = 27158734
qualified activation total = 41036433
qualified activation min/operator = 300623
mean selected prune ratio = 0.746176
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
negative transfer = 0
direct flow writes = 0
```

Current E130B arithmetic text-IO transfer evidence:

```text
transfer pass operators = 9 / 9
visible transfer cases = 270000
word-problem no-call cases = 135000
visible transfer accuracy min = 1.000
word-problem no-call accuracy min = 1.000
qualified transfer activation total = 270000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
direct flow writes = 0
overbroad control wrong-scope calls = 18000
```

Current E131 visible equation assistant-render evidence:

```text
dataset rows loaded = 130000
transfer pass operators = 9 / 9
visible equation cases = 108000
word-problem no-call cases = 54000
visible equation extraction accuracy min = 1.000
word-problem no-call accuracy min = 1.000
qualified visible activation total = 108000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
E130B baseline visible misses = 96711
overbroad control wrong-scope calls = 18000
```

Current E132 external math-text skill-farm evidence:

```text
dataset rows loaded = 215051
external sources = 5
external families = 11
math-text operators = 16
Orange/Legendary math-text operators = 16
external support min/operator = 5953
qualified activation total = 4883030
qualified activation min/operator = 302510
negative-scope no-call cases = 78859
mutation attempts = 146005
accepted mutations = 650
rollbacks = 145355
mean selected prune ratio = 0.736875
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
overbroad solver control wrong-scope calls = 16703
```

Current E133 math-text route composition evidence:

```text
composition pass operators = 16 / 16
route cases = 176000
visible arithmetic route cases = 10000
structural guard cases = 118000
hidden word-problem no-solve cases = 48000
route accuracy min = 1.000
visible arithmetic route accuracy min = 1.000
structural guard accuracy min = 1.000
hidden word-problem no-solve accuracy min = 1.000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
overbroad solver control wrong-scope calls = 24000
trust-control false commits = 4125
trust-control direct writes = 3000
```

Current E134 external math-text OOD route stress evidence:

```text
OOD pass operators = 16 / 16
OOD route cases = 208000
visible arithmetic OOD cases = 11875
structural guard OOD cases = 153125
hidden word-problem OOD no-solve cases = 43000
counterexample cases = 48000
OOD route accuracy min = 1.000
visible arithmetic OOD accuracy min = 1.000
structural guard OOD accuracy min = 1.000
hidden word-problem OOD no-solve accuracy min = 1.000
counterexample accuracy min = 1.000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
E133 baseline OOD misses = 36275
overbroad solver control wrong-scope calls = 19200
trust-control false commits = 4200
trust-control direct writes = 2400
```

Current E135 math-text multi-route dialogue-state evidence:

```text
dialogue pass operators = 16 / 16
dialogue cases = 136000
dialogue turns = 367400
hidden word-problem dialogue no-solve cases = 29500
visible reentry dialogue cases = 10500
stale route rejection cases = 22400
cross-thread rejection cases = 11200
counterexample dialogue cases = 76500
dialogue state accuracy min = 1.000
current-turn route accuracy min = 1.000
route-state integrity min = 1.000
hidden word-problem dialogue no-solve accuracy min = 1.000
counterexample dialogue accuracy min = 1.000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
direct flow writes = 0
stale route reuse = 0
cross-thread contamination = 0
```

Current E136A assistant-text skill-farm evidence:

```text
dataset rows loaded = 447766
external sources = 5
external families = 12
assistant/text operators = 18
Orange/Legendary assistant/text operators = 18
external support total = 1435199
external support min/operator = 4746
qualified activation total = 5521276
qualified activation min/operator = 302123
negative-scope cases = 119868
mutation attempts = 179840
accepted mutations = 827
rollbacks = 179013
mean selected prune ratio = 0.758889
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
overbroad chatbot control wrong-scope calls = 25558
```

Current E136B assistant-text route-composition evidence:

```text
source E136A operators = 18
route pass operators = 18 / 18
dataset rows loaded = 447766
route seed rows = 4096
route cases = 144000
multi-route composition cases = 53000
boundary cases = 72000
negative-scope cases = 18000
qualified route activation total = 144000
qualified route activation min/operator = 8000
route families = 10
route accuracy min = 1.000
route stack accuracy min = 1.000
primary route accuracy min = 1.000
boundary accuracy min = 1.000
multi-route composition accuracy min = 1.000
negative-scope accuracy min = 1.000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
overbroad chatbot control wrong-scope calls = 14400
unsafe direct-write control direct writes = 14400
source hallucination control unsupported answers = 14400
```

Current E136C assistant-text polished-render quick evidence:

```text
case count = 12
pass count = 12
mode accuracy = 1.000
polished render pass rate = 1.000
JSON outputs valid = 2 / 2
average response words = 27.083
route stack covered samples = 11
greeting fallback samples = 1
raw action leaks = 0
forbidden claims = 0
direct-write claims = 0
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
- backfill prior scoped CoreMemoryCandidate operators through stricter
  Orange/Legendary probation without renaming or bypassing no-harm gates.

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
- farm scoped assistant/text lenses and guards from the E136 assistant-text
  seed pack for role/turn boundaries, task decomposition, summarization, code
  boundaries, refusals, source absence, response formats, preference
  boundaries, synthetic dialogue noise, reasoning instructions, safety-sensitive
  domains, longform requests, comparison/evaluation prompts, and multilingual
  task boundaries;
- compose those E136 assistant/text lenses and guards into bounded
  schema-gated route stacks while rejecting overbroad chatbot,
  source-hallucination, rejected-response-reuse, and direct-write controls;
- render short polished deterministic assistant text for a 12-sample quick set
  covering greeting, summary, code, source-defer, JSON, no-solve math, safety,
  comparison, translation, no-overwrite, rejected-response, and outline cases;
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
- transfer those visible arithmetic expression/trace skills into longer
  text-IO wrappers when the arithmetic payload is explicit;
- route assistant-style visible equation surfaces into deterministic scoped
  arithmetic renders when the arithmetic expression or trace is visible;
- no-call natural-language word problems that lack a visible arithmetic
  expression or trace.

### Math-text lenses and guards

- detect and scope LaTeX inline/display math surfaces;
- preserve boxed/final-answer boundaries without trusting them as proof;
- detect TIR/python/output/error blocks as structure, not direct Flow writes;
- ground proof-step connectors such as therefore, hence, implies, substituting,
  and equating;
- guard geometry/diagram, matrix/vector, equation-system, piecewise/function,
  fraction/probability, variable-definition, summation/sequence, unit/quantity,
  and answer-format surfaces;
- compose those math-text lenses/guards into assistant route decisions, routing
  explicit visible arithmetic to the scoped arithmetic renderer while keeping
  proof/TIR/matrix/geometry/unit/answer-format surfaces guarded;
- survive OOD wrapper stress and counterexample lures for those route decisions,
  including wrong boxed answers, spoofed TIR output, diagram/unit/proof lures,
  and conflicting final-answer surfaces;
- preserve current-turn route state across controlled multi-turn assistant
  dialogue surfaces with stale-route, cross-thread, hidden no-solve, visible
  reentry, and counterexample turns;
- keep prose-only word problems on a no-solve/no-call path until a later route
  explicitly supplies visible evidence and scoped capability.

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

math-text boundary lenses
  detect LaTeX, proof, TIR, matrix, diagram, answer-format, and quantity spans

no-solve guards
  keep hidden/prose-only word problems out of direct arithmetic routes

counterexample guards
  reject spoofed final answers, boxed answers, and tool-output trust lures

dialogue-state route guards
  keep the active/current route from being overwritten by stale or cross-thread turns

assistant-text lenses/guards
  classify scoped assistant/text request shapes and boundaries without claiming
  open-domain generation or production assistant behavior

assistant-text route composition
  compose scoped assistant/text lenses and guards into bounded route stacks
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
