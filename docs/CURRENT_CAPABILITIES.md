# VRAXION Current Capabilities

_Last updated: 2026-06-17_

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

Current E136D OutputTextField binary matrix evidence:

```text
field name = OutputTextField
matrix shape = N x 8 bit cells
row meaning = one UTF-8 byte
case count = 10
pass count = 10
committed text roundtrips = 7 / 7
guarded rejects = 3 / 3
zero-fill checks = 10 / 10
tamper detection = 1 / 1
```

Current E136E idle think-tick proposal refinement evidence:

```text
case count = 8
pass count = 8
idle tick total = 10
proposal count = 10
Agency checks = 10
new input total = 0
improvement count = 4
non-degradation count = 8
direct-write rejects = 1
OutputTextField roundtrip/checksum/zero-fill = 8 / 8
```

Current E136F idle think-tick heldout series evidence:

```text
case count = 70
pass count = 70
arithmetic cases = 36
arithmetic improvements = 36 / 36
no-pocket cases = 6
no-pocket preserves = 6 / 6
idle tick total = 90
proposal count = 90
Agency checks = 90
new input total = 0
improvement count = 48
non-degradation count = 70
direct-write rejects = 4
unsupported-claim rejects = 6
OutputTextField roundtrip/checksum/zero-fill = 70 / 70
```

Current E136N3 parallel direct-write A/B smoke evidence:

```text
case count = 123
direct-write accuracy = 0.089431
Agency-gated accuracy = 1.000000
direct-write unsafe commits = 34
Agency-gated unsafe commits = 0
direct-write conflict cases = 102
direct-write nondeterministic cases = 102
direct-write missing chunk metadata = 10
direct-write runtime writes = 602
Agency-gated runtime direct writes = 0
direct-write held variant promotions = 36
Agency-gated held variant promotions = 0
direct-write safe controls correct = 11
expected child checks = 36
Agency-gated child checks = 36
expected Flow chunks = 10
Agency-gated Flow chunks = 10
destructive deletes = 0
```

Previous E136N2 Agency Matrix arbitration smoke evidence:

```text
input E136N operators = 34
training examples = 118
training epochs completed = 2
training converged epoch = 2
training final epoch updates = 0
case count = 146
baseline accuracy = 0.232877
Agency Matrix accuracy = 1.000000
baseline unsafe commits = 34
Agency Matrix unsafe commits = 0
expected child checks = 36
Agency Matrix child checks = 36
expected Flow chunks = 10
Agency Matrix Flow chunks = 10
baseline child-call proxy = 336
Agency Matrix child-call proxy = 202
child-call proxy reduction = 134
child-call proxy reduction ratio = 0.398810
challenger promoted = 0
lineage-hold promoted = 0
destructive deletes = 0
```

Previous E136N primary/secondary variant governance evidence:

```text
operator count = 34
variant registry rows = 68
primary variants = 34
secondary variants = 34
primary active = 16
primary current = 11
primary abstract current = 7
secondary rollback = 16
secondary challenger = 11
secondary lineage hold = 7
retired redundant = 0
retirement lane created = 16
retirement candidates = 0
destructive deletes = 0
ambiguous primary operators = 0
missing primary operators = 0
orphan secondaries = 0
runtime overlay removed activations = 3,450,257
challenger candidate removed not applied = 18,792,948
rollback snapshots = 16
rollback triggers = 0
strict recall misses = 0
wrong-scope proxy calls = 0
hard negatives = 0
unsupported answers = 0
direct Flow writes = 0
```

Previous E136M runtime replacement overlay evidence:

```text
operator count = 34
runtime overlay active = 16
runtime overlay apply = 16
verified replacement applies = 7
light-prune overlay applies = 9
legacy triggers disabled in overlay = 16
legacy triggers retained for rollback = 16
challenger/OOD queue = 11
challenger runtime overlay active = 0
abstract lineage split = 7
abstract runtime overlay active = 0
rollback snapshots = 16
rollback triggers = 0
production destructive deletes = 0
runtime mutation allowed now = 16
runtime overlay activation total = 185,147,668
runtime overlay removed activations = 3,450,257
runtime overlay removed ratio = 0.018294
challenger candidate removed not applied = 18,792,948
strict recall misses = 0
wrong-scope proxy calls = 0
hard negatives = 0
unsupported answers = 0
direct Flow writes = 0
```

Previous E136L runtime replacement canary evidence:

```text
operator count = 34
direct canary tested = 16
direct canary passed = 16
old operators removed in canary = 16
runtime replacement canary allowed = 16
production runtime applies = 0
destructive applies = 0
challenger/OOD tested = 11
challenger hold = 11
abstract lineage hold = 7
rollback manifest entries = 16
rollback triggers = 0
direct canary legacy activations = 60,362,384
direct canary selected activations = 56,912,127
direct canary removed activations = 3,450,257
direct canary removed ratio = 0.057159
sample rows processed = 8,345
sample direct removed activations = 1,031
strict recall misses = 0
wrong-scope proxy calls = 0
hard negatives = 0
unsupported answers = 0
direct Flow writes = 0
```

Previous E136K operator replacement apply-plan evidence:

```text
operator count = 34
direct canary ready = 16
challenger/OOD required = 11
abstract lineage required = 7
runtime mutation allowed now = 0
destructive applies = 0
rollback manifest entries = 16
current activation total = 188,597,925
selected activation total = 166,354,720
shadow-pruned activation total = 22,243,205
shadow prune ratio = 0.117940
direct canary prune ratio = 0.057159
challenger prune ratio = 0.351548
strict recall misses = 0
wrong-scope proxy calls = 0
hard negatives = 0
unsupported answers = 0
direct Flow writes = 0
```

Previous E136J shadow-variant apply/residual-prune evidence:

```text
stop reason = deadline
cycles completed = 8,094
rows processed = 33,153,024
elapsed seconds = 46,317.709
operator count = 34
replacement ready = 27
direct runtime candidates = 16
tightened challenger required = 11
abstract lineage required = 7
current activation total = 188,597,925
selected activation total = 166,354,720
shadow-pruned activation total = 22,243,205
strict recall misses = 0
wrong-scope proxy calls = 0
hard negatives = 0
unsupported answers = 0
direct Flow writes = 0
```

Previous E136I operator supersession ledger evidence:

```text
operator count = 34
replacement ready = 27
direct runtime candidates = 16
tightened challenger required = 11
abstract lineage required = 7
destructive drops = 0
projected current activations = 3,373,788
projected selected activations = 2,891,151
projected pruned activations = 482,637
projected output activation delta = -482,637
accepted mutations = 96 / 43,720
hard negatives = 0
wrong-scope calls = 0
unsupported answers = 0
direct Flow writes = 0
```

Previous E136H existing operator refinement evidence:

```text
cycles completed = 40
operator count = 34
rows seen total = 12,480,000
current activation total = 3,373,788
selected activation total = 2,891,151
pruned activation total = 482,637
verified label count = 16
tentative tighten count = 11
abstract but useful count = 7
hold for more evidence count = 0
hard negatives = 0
wrong-scope calls = 0
unsupported answers = 0
direct Flow writes = 0
```

Previous E136G adaptive idle tick budget evidence:

```text
case count = 24
pass count = 24
adaptive tick total = 33
fixed baseline tick total = 120
tick savings vs fixed = 87
average adaptive ticks = 1.375000
proposal continue fields = 33 / 33
Agency continue yes = 9
Agency continue overrides = 2
immediate answer stop at t+1 = 8
chained complete = 3 / 3
direct-write repair at t+2 = 3 / 3
no-pocket stop at t+1 = 4 / 4
unsupported-claim rejects = 4
OutputTextField roundtrip/checksum/zero-fill = 24 / 24
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
- improve fixed observations during idle ticks when a matching scoped
  pocket/trace exists, while preserving safe output when no such pocket exists.
- adapt the idle tick budget through explicit one-more-tick recommendations
  that Agency can accept or override.

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
- store committed output text in an `OutputTextField` represented as an N x 8
  binary matrix, with UTF-8 byte roundtrip, overflow/direct-write/NUL reject,
  zero-fill, and tamper-detection smoke coverage;
- improve fixed observations across idle ticks only by emitting checked
  proposals, with no new input and no direct OutputTextField writes;
- stop immediate answers at t+1, continue chained refinements when progress is
  visible, repair direct-write attempts at t+2, and stop no-pocket controls at
  t+1 under an adaptive Agency-approved idle budget;
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

shadow operator apply / residual prune
  replay selected variants non-destructively and measure recall, wrong-scope,
  and prune impact before any runtime replacement

operator replacement apply planning
  split shadow-proven variants into rollback-safe direct canaries,
  challenger/OOD-required replacements, and abstract lineage holds

runtime replacement canary
  remove legacy triggers in canary, activate selected variants, and audit
  rollback triggers before any production runtime replacement

runtime replacement overlay
  activate the direct canary-passed selected variants in a rollback-safe
  runtime-facing overlay while holding challenger and abstract rows

primary/secondary variant governance
  keep each operator on exactly one primary variant while retaining explicit
  rollback, challenger, and lineage-hold secondary variants

Agency Matrix arbitration
  train a small arbitration matrix over primary/secondary proposal features,
  reject unsafe/direct-write proposals, hold challenger/lineage candidates for
  child checks, and commit compatible Flow chunks without a hand-written
  hierarchy registry

parallel proposal fanout with commit barrier
  let multiple operators propose in the same tick, then require Agency-gated
  commit for chunk/multi-write behavior; parallel direct Flow write is rejected
  as a default after E136N3 A/B evidence
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
