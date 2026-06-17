# VRAXION

VRAXION is building **INSTNCT**: a Rust-first, gradient-free architecture whose active object is a governed computational substrate, not a fixed backprop-trained layer stack.

This repository is consolidated around the current winning mainline. Historical beta, bounded-service, byte-pipeline, and probe-era work remains available for auditability, but it is not the active public surface unless promoted back into `main`.

## Current Source Of Truth

```text
branch = main
current_release = v6.1.7
current_evidence_anchor = E136N3 parallel direct-write A/B smoke
current_evidence_subject = parallel proposal fanout versus parallel direct Flow write over the E136N/E136N2 proposal surface
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8
latest_released_runtime_subject = Add training data curriculum readiness gate
```

The latest GitHub release is [`v6.1.7`](https://github.com/VRAXION/VRAXION/releases/tag/v6.1.7). It anchors the E127 cycle-40 governed text-operator library checkpoint.

## Current Mainline

| Slice | Commit | Purpose |
|---|---|---|
| E69-E79 | `a908a838` release line | Rust Pocket Library, curriculum, final-train supervision, global merge, and training-data/curriculum readiness gate |
| E80-E85 | `56a9cf03` | CALC-SCRIBE visible calculation-trace validation and mixed-stream no-call routing |
| E86-E89 | `a6935e61` | LocalGolden seeded curriculum, sparse active-set selection, survival gauntlet, and Operator naming/schema lock |
| E90-E106 | `b75c64cb` | Text-evidence, temporal, agency, route, memory, compression, and task-progress Operator curriculum expansions |
| E107 | `1fcdf954` | E90-E106 survival role and regression gauntlet |
| E108 | `0389c211` | External dataset transfer and negative-scope no-harm gauntlet |
| E109 | `555c5006` | Operator rank ladder and GoldenWatch probation policy |
| E110 | `b378c2c5` | Silver-to-Gold scoped probation wave: 35/35 promoted, 0 hard negatives |
| E111 | `d71e3657` | Bronze mutation/prune wave: 87/87 promoted to scoped Gold variants, 0 hard negatives |
| E112 | `9de33241` | Gold-to-CoreMemoryCandidate prune-heavy probation wave: 136/136 qualified, 0 hard negatives |
| E113 | `05415f5b` | FineWeb-Edu 100k light stress: baseline 2,624 hard negatives across 88 operators, selected recycled variants 0 hard negatives |
| E119-E126 | tracked on `main` | FineWeb/text-understanding skill mining and Orange/Legendary probation |
| E127 | `f32a6f4b` | Overnight cyclic Orange/Legendary text-operator farm: 40 cycles, 382 scoped operators, 0 hard negatives |
| E128 | tracked on `main` | Lightweight assistant text-IO render training: 320 local prompts, train/validation/heldout action accuracy 1.000, 0 unsupported answers |
| E129 | tracked on `main` | Arithmetic trace Orange/Legendary probation: 9/9 scoped arithmetic operators, 2.7M qualified activations, 0 hard negatives |
| E130A | tracked on `main` | CoreMemoryCandidate-to-Orange backfill: 136/136 promoted, 41.0M qualified activations, 0 hard negatives |
| E130B | tracked on `main` | Arithmetic text-IO transfer: 9/9 E129 operators, 270k visible-transfer cases, 135k hidden word-problem no-call cases, 0 wrong-scope calls |
| E131 | tracked on `main` | Visible equation extraction and assistant arithmetic render: 9/9 E129/E130B operators, 108k visible-equation cases, 54k hidden word-problem no-call cases, 0 hard negatives |
| E132 | tracked on `main` | External math-text skill farm: 16/16 scoped math-text lenses/guards promoted to Orange/LegendaryCandidate from a 215,051-row external seed pack, 0 hard negatives |
| E133 | tracked on `main` | Math-text route composition/no-solve assistant confirm: 16/16 E132 operators passed, 176k route cases, 10k visible arithmetic routes, 48k hidden word-problem no-call cases, 0 hard negatives |
| E134 | tracked on `main` | External math-text OOD route stress/counterexample gauntlet: 16/16 E133 route operators passed, 208k OOD cases, 48k counterexamples, 36,275 E133 baseline OOD misses covered, 0 hard negatives |
| E135 | tracked on `main` | Math-text multi-route assistant dialogue-state gauntlet: 16/16 E134 route operators passed, 136k dialogue cases, 367.4k turns, 0 stale route reuse, 0 cross-thread contamination, 0 hard negatives |
| E136A | tracked on `main` | Assistant-text skill farm mutation/prune Orange cycle: 18/18 scoped assistant/text operators promoted from a 447,766-row E136 seed pack, 0 hard negatives, 0 direct Flow writes |
| E136B | tracked on `main` | Assistant-text route composition/boundary confirm: 18/18 E136A operators compose into bounded route stacks, 144k route cases, 53k multi-route cases, 72k boundary cases, 0 hard negatives, 0 direct Flow writes |
| E136C | tracked on `main` | Assistant-text polished render quick test: 12/12 inference samples passed, 2/2 JSON outputs valid, raw action leaks 0, forbidden claims 0, direct-write claims 0 |
| E136D | tracked on `main` | OutputTextField binary matrix smoke: 10/10 cases passed, N x 8 byte matrix shape, 7/7 roundtrips, overflow/direct-write/NUL rejects, zero-fill 10/10, tamper detect 1/1 |
| E136E | tracked on `main` | Idle think-tick proposal refinement smoke: 8/8 cases passed, 10 idle proposals checked, 0 new input, 4 improvements, 8/8 non-degradation, direct-write reject 1, OutputTextField roundtrip 8/8 |
| E136F | tracked on `main` | Idle think-tick heldout series confirm: 70/70 cases passed, 36/36 arithmetic heldout improvements, 6/6 no-pocket preserves, 90 proposals checked, 0 new input, 4 direct-write rejects, 6 unsupported-claim rejects, OutputTextField roundtrip 70/70 |
| E136G | tracked on `main` | Adaptive idle tick budget confirm: 24/24 cases passed, adaptive 33 ticks vs fixed 120, 8 immediate answers stopped at t+1, 3/3 chained cases completed, 3/3 direct-write repairs at t+2, 4/4 no-pocket stops at t+1, 2 Agency continuation overrides |
| E136H | tracked on `main` | Existing operator refinement mutation/prune night cycle: 40 cycles, 34 E132/E136A operators, 3.37M current activations, 16 verified labels, 11 tightened triggers, 7 abstract-but-useful kernels, 0 hard negatives, 0 direct Flow writes |
| E136I | tracked on `main` | Operator supersession and output ledger planning: 27 replacement-ready variants, 16 direct runtime candidates, 11 tightened challenger-required replacements, 7 abstract lineage-required kernels, projected prune impact 482,637 activations, 0 destructive drops |
| E136J | tracked on `main` | Shadow variant apply and residual prune confirm: 8,094 cycles, 33.15M replay rows, 22.24M shadow-pruned activations, 0 strict recall misses, 0 wrong-scope proxy calls, 0 direct Flow writes, stopped by deadline |
| E136K | tracked on `main` | Operator replacement apply plan: 16 direct canary-ready candidates, 11 challenger/OOD-required replacements, 7 abstract lineage holds, 16 rollback entries, 0 destructive applies |
| E136L | tracked on `main` | Runtime replacement canary: 16/16 direct canaries passed with old trigger removed in canary, 11 challenger rows held, 7 abstract lineage holds, 0 rollback triggers, 0 destructive applies |
| E136M | tracked on `main` | Runtime replacement overlay: 16 active overlay applies, 7 verified replacements, 9 light-prune overlays, 11 challenger/OOD queue rows held, 7 abstract lineage split rows held, 0 destructive deletes |
| E136N | tracked on `main` | Primary/secondary variant governance: 34 operators with one primary and one secondary each, 16 primary-active overlays, 16 rollback secondaries, 11 challenger secondaries, 7 lineage-hold secondaries, 0 retired variants |
| E136N2 | tracked on `main` | Agency Matrix arbitration smoke: 118 training examples converged in 2 epochs, 146 proposal-bundle cases, baseline accuracy 0.232877, Agency Matrix accuracy 1.000000, 10 Flow chunks, 0 unsafe commits |
| E136N3 | tracked on `main` | Parallel direct-write A/B smoke: 123 cases, direct-write accuracy 0.089431, Agency-gated accuracy 1.000000, direct-write unsafe commits 34, direct-write nondeterministic cases 102, gated runtime direct writes 0 |

Current E136N3 parallel direct-write A/B smoke state:

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

Previous E136N primary/secondary variant governance state:

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

Previous E136M runtime replacement overlay state:

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

Previous E136L runtime replacement canary state:

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
challenger runtime apply allowed = 0
abstract lineage hold = 7

rollback manifest entries = 16
rollback triggers = 0

current activation total = 188,597,925
selected activation total = 166,354,720
shadow-pruned activation total = 22,243,205
shadow prune ratio = 0.117940

direct canary legacy activations = 60,362,384
direct canary selected activations = 56,912,127
direct canary removed activations = 3,450,257
direct canary removed ratio = 0.057159

sample rows processed = 8,345
sample direct legacy activations = 10,056
sample direct selected activations = 9,025
sample direct removed activations = 1,031

strict recall misses = 0
wrong-scope proxy calls = 0
hard negatives = 0
unsupported answers = 0
direct Flow writes = 0
```

Previous E136K operator replacement apply plan state:

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

Previous E136J shadow variant apply/residual prune state:

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

Previous E136I operator supersession ledger state:

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

Previous E136H existing operator refinement state:

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

Previous E136G adaptive idle tick budget state:

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
OutputTextField roundtrip = 24 / 24
```

Previous E136F idle think-tick heldout series state:

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
OutputTextField roundtrip = 70 / 70
average quality gain = 0.474286
```

Previous E136E idle think-tick proposal refinement state:

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
OutputTextField roundtrip = 8 / 8
average quality gain = 0.350000
```

Previous E136D OutputTextField binary matrix state:

```text
field name = OutputTextField
matrix shape = N x 8 bit cells
row meaning = one UTF-8 byte
case count = 10
pass count = 10
roundtrip pass = 7 / 7 committed cases
zero-fill pass = 10 / 10
overflow rejects = 1
direct-write rejects = 1
NUL-byte rejects = 1
tamper detects = 1
```

Previous E136C assistant-text polished-render state:

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

Previous E136B assistant-text route-composition state:

```text
source E136A operators = 18
route pass operators = 18 / 18
dataset rows loaded = 447,766
route seed rows = 4,096
route cases = 144,000
multi-route composition cases = 53,000
boundary cases = 72,000
negative-scope cases = 18,000
qualified route activation total = 144,000
qualified route activation min/operator = 8,000
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
overbroad chatbot control wrong-scope calls = 14,400
unsafe direct-write control direct writes = 14,400
source hallucination control unsupported answers = 14,400
```

Previous E136A assistant-text skill-farm state:

```text
dataset rows loaded = 447,766
external sources = 5
external families = 12
assistant/text operators = 18
Orange/Legendary assistant/text operators = 18
external support total = 1,435,199
external support min/operator = 4,746
qualified activation total = 5,521,276
qualified activation min/operator = 302,123
negative-scope cases = 119,868
mutation attempts = 179,840
accepted mutations = 827
rollbacks = 179,013
mean selected prune ratio = 0.758889
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
overbroad chatbot control wrong-scope calls = 25,558
```

Current E127 scoped operator state:

```text
Orange/Legendary scoped operators = 382
E127 cycles = 40
hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
```

Current E128 text-IO bridge state:

```text
prompt corpus = 320
train / validation / heldout = 160 / 64 / 96
action accuracy = 1.000 / 1.000 / 1.000
operator trace validity = 1.000
unsupported answers = 0
boundary-claim violations = 0
```

Current E129 arithmetic trace state:

```text
Orange/Legendary arithmetic operators = 9
qualified activation total = 2,700,000
qualified activation min/operator = 300,000
negative-scope no-call cases = 9,000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
```

Current E130A Orange backfill state:

```text
Orange/Legendary backfilled operators = 136
qualified activation before total = 13,877,699
qualified activation add total = 27,158,734
qualified activation total = 41,036,433
qualified activation min/operator = 300,623
mean selected prune ratio = 0.746176
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
negative transfer = 0
direct flow writes = 0
```

Current E130B arithmetic text-IO transfer state:

```text
transfer pass operators = 9 / 9
visible transfer cases = 270,000
word-problem no-call cases = 135,000
visible transfer accuracy min = 1.000
word-problem no-call accuracy min = 1.000
qualified transfer activation total = 270,000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
direct flow writes = 0
overbroad control wrong-scope calls = 18,000
```

Current E132 external math-text skill-farm state:

```text
dataset rows loaded = 215,051
external sources = 5
external families = 11
math-text operators = 16
Orange/Legendary math-text operators = 16
external support min/operator = 5,953
qualified activation total = 4,883,030
qualified activation min/operator = 302,510
negative-scope no-call cases = 78,859
mutation attempts = 146,005
accepted mutations = 650
rollbacks = 145,355
mean selected prune ratio = 0.736875
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
overbroad solver control wrong-scope calls = 16,703
```

Current E133 math-text route-composition state:

```text
composition pass operators = 16 / 16
route cases = 176,000
visible arithmetic route cases = 10,000
structural guard cases = 118,000
hidden word-problem no-solve cases = 48,000
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
overbroad solver control wrong-scope calls = 24,000
trust-control false commits = 4,125
trust-control direct writes = 3,000
```

Current E134 external math-text OOD route stress state:

```text
OOD pass operators = 16 / 16
OOD route cases = 208,000
visible arithmetic OOD cases = 11,875
structural guard OOD cases = 153,125
hidden word-problem OOD no-solve cases = 43,000
counterexample cases = 48,000
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
E133 baseline OOD misses = 36,275
overbroad solver control wrong-scope calls = 19,200
trust-control false commits = 4,200
trust-control direct writes = 2,400
```

Current E135 math-text multi-route dialogue-state:

```text
dialogue pass operators = 16 / 16
dialogue cases = 136,000
dialogue turns = 367,400
hidden word-problem dialogue no-solve cases = 29,500
visible reentry dialogue cases = 10,500
stale route rejection cases = 22,400
cross-thread rejection cases = 11,200
counterexample dialogue cases = 76,500
dialogue state accuracy min = 1.000
current-turn route accuracy min = 1.000
route-state integrity min = 1.000
hidden word-problem dialogue no-solve accuracy min = 1.000
counterexample dialogue accuracy min = 1.000
hard negatives = 0
wrong-scope calls = 0
false commits = 0
direct flow writes = 0
stale route reuse = 0
cross-thread contamination = 0
```

Current E131 visible equation assistant-render state:

```text
dataset rows loaded = 130,000
transfer pass operators = 9 / 9
visible equation cases = 108,000
word-problem no-call cases = 54,000
visible equation extraction accuracy min = 1.000
word-problem no-call accuracy min = 1.000
qualified visible activation total = 108,000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
E130B baseline visible misses = 96,711
overbroad control wrong-scope calls = 18,000
```

## What Is Current

- Active Rust runtime: [`vraxion-runtime/`](vraxion-runtime/)
- Current status: [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
- Current capabilities: [`docs/CURRENT_CAPABILITIES.md`](docs/CURRENT_CAPABILITIES.md)
- Getting started: [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)
- Validated findings: [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md)
- Operator cards: [`docs/research/OPERATOR_LIBRARY_CARDS.md`](docs/research/OPERATOR_LIBRARY_CARDS.md)
- Current result: [`docs/research/E136N3_PARALLEL_DIRECT_WRITE_AB_SMOKE_RESULT.md`](docs/research/E136N3_PARALLEL_DIRECT_WRITE_AB_SMOKE_RESULT.md)
- E136N2 Agency Matrix arbitration: [`docs/research/E136N2_AGENCY_MATRIX_ARBITRATION_SMOKE_RESULT.md`](docs/research/E136N2_AGENCY_MATRIX_ARBITRATION_SMOKE_RESULT.md)
- E136N primary/secondary variant governance: [`docs/research/E136N_PRIMARY_SECONDARY_VARIANT_GOVERNANCE_RESULT.md`](docs/research/E136N_PRIMARY_SECONDARY_VARIANT_GOVERNANCE_RESULT.md)
- E136M runtime replacement overlay: [`docs/research/E136M_RUNTIME_REPLACEMENT_APPLY_OR_ABSTRACT_LINEAGE_SPLIT_RESULT.md`](docs/research/E136M_RUNTIME_REPLACEMENT_APPLY_OR_ABSTRACT_LINEAGE_SPLIT_RESULT.md)
- E136L runtime replacement canary: [`docs/research/E136L_RUNTIME_REPLACEMENT_CANARY_AND_TIGHTENED_CHALLENGER_CONFIRM_RESULT.md`](docs/research/E136L_RUNTIME_REPLACEMENT_CANARY_AND_TIGHTENED_CHALLENGER_CONFIRM_RESULT.md)
- E136K operator replacement apply plan: [`docs/research/E136K_OPERATOR_REPLACEMENT_APPLY_PLAN_OR_FLOW_SCALE_TRANSFER_RESULT.md`](docs/research/E136K_OPERATOR_REPLACEMENT_APPLY_PLAN_OR_FLOW_SCALE_TRANSFER_RESULT.md)
- E136J shadow apply evidence: [`docs/research/E136J_SHADOW_VARIANT_APPLY_AND_RESIDUAL_PRUNE_CONFIRM_RESULT.md`](docs/research/E136J_SHADOW_VARIANT_APPLY_AND_RESIDUAL_PRUNE_CONFIRM_RESULT.md)
- E136I operator supersession ledger: [`docs/research/E136I_OPERATOR_SUPERSESSION_AND_OUTPUT_LEDGER_PLANNING_RESULT.md`](docs/research/E136I_OPERATOR_SUPERSESSION_AND_OUTPUT_LEDGER_PLANNING_RESULT.md)
- E136H existing operator refinement: [`docs/research/E136H_EXISTING_OPERATOR_REFINEMENT_MUTATION_PRUNE_NIGHT_CYCLE_RESULT.md`](docs/research/E136H_EXISTING_OPERATOR_REFINEMENT_MUTATION_PRUNE_NIGHT_CYCLE_RESULT.md)
- E136G adaptive idle tick budget: [`docs/research/E136G_ADAPTIVE_IDLE_TICK_BUDGET_CONFIRM_RESULT.md`](docs/research/E136G_ADAPTIVE_IDLE_TICK_BUDGET_CONFIRM_RESULT.md)
- E136F idle think-tick heldout series: [`docs/research/E136F_IDLE_THINK_TICK_HELDOUT_SERIES_CONFIRM_RESULT.md`](docs/research/E136F_IDLE_THINK_TICK_HELDOUT_SERIES_CONFIRM_RESULT.md)
- E136E idle think-tick proposal refinement: [`docs/research/E136E_IDLE_THINK_TICK_PROPOSAL_REFINEMENT_SMOKE_RESULT.md`](docs/research/E136E_IDLE_THINK_TICK_PROPOSAL_REFINEMENT_SMOKE_RESULT.md)
- E136D OutputTextField binary matrix: [`docs/research/E136D_OUTPUT_TEXT_FIELD_BINARY_MATRIX_SMOKE_RESULT.md`](docs/research/E136D_OUTPUT_TEXT_FIELD_BINARY_MATRIX_SMOKE_RESULT.md)
- E136C assistant-text polished render: [`docs/research/E136C_ASSISTANT_TEXT_POLISHED_RENDER_QUICK_TEST_RESULT.md`](docs/research/E136C_ASSISTANT_TEXT_POLISHED_RENDER_QUICK_TEST_RESULT.md)
- E136B assistant-text route composition: [`docs/research/E136B_ASSISTANT_TEXT_ROUTE_COMPOSITION_AND_BOUNDARY_CONFIRM_RESULT.md`](docs/research/E136B_ASSISTANT_TEXT_ROUTE_COMPOSITION_AND_BOUNDARY_CONFIRM_RESULT.md)
- E136A assistant-text skill farm: [`docs/research/E136A_ASSISTANT_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md`](docs/research/E136A_ASSISTANT_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md)
- E135 dialogue-state: [`docs/research/E135_MATH_TEXT_MULTI_ROUTE_ASSISTANT_DIALOGUE_STATE_GAUNTLET_RESULT.md`](docs/research/E135_MATH_TEXT_MULTI_ROUTE_ASSISTANT_DIALOGUE_STATE_GAUNTLET_RESULT.md)
- E134 OOD route stress: [`docs/research/E134_EXTERNAL_MATH_TEXT_OOD_ROUTE_STRESS_AND_COUNTEREXAMPLE_GAUNTLET_RESULT.md`](docs/research/E134_EXTERNAL_MATH_TEXT_OOD_ROUTE_STRESS_AND_COUNTEREXAMPLE_GAUNTLET_RESULT.md)
- E133 math-text route composition: [`docs/research/E133_MATH_TEXT_ROUTE_COMPOSITION_AND_NO_SOLVE_ASSISTANT_CONFIRM_RESULT.md`](docs/research/E133_MATH_TEXT_ROUTE_COMPOSITION_AND_NO_SOLVE_ASSISTANT_CONFIRM_RESULT.md)
- E132 external math-text skill farm: [`docs/research/E132_EXTERNAL_MATH_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md`](docs/research/E132_EXTERNAL_MATH_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md)
- E131 visible equation assistant render: [`docs/research/E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET_RESULT.md`](docs/research/E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET_RESULT.md)
- E130B arithmetic text-IO transfer: [`docs/research/E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET_RESULT.md`](docs/research/E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET_RESULT.md)
- E130A Orange backfill: [`docs/research/E130A_COREMEMORY_TO_ORANGE_BACKFILL_GAUNTLET_RESULT.md`](docs/research/E130A_COREMEMORY_TO_ORANGE_BACKFILL_GAUNTLET_RESULT.md)
- E129 arithmetic trace: [`docs/research/E129_ARITHMETIC_TRACE_ORANGE_LEGENDARY_PROBATION_RESULT.md`](docs/research/E129_ARITHMETIC_TRACE_ORANGE_LEGENDARY_PROBATION_RESULT.md)
- E128 text-IO bridge: [`docs/research/E128_ASSISTANT_TEXT_IO_LIGHTWEIGHT_RENDER_TRAINING_RESULT.md`](docs/research/E128_ASSISTANT_TEXT_IO_LIGHTWEIGHT_RENDER_TRAINING_RESULT.md)
- E127 checkpoint: [`docs/research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md`](docs/research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md)
- Handover for fresh Codex sessions: [`CODEX_HANDOVER.md`](CODEX_HANDOVER.md)
- GitHub Pages front door: <https://vraxion.github.io/VRAXION/>
- Wiki timeline: <https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive>

## Claim Boundary

Allowed current claim:

> VRAXION v6 has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, governed Operator evidence through E127, an E128 lightweight assistant text-IO render-training smoke, E129 scoped exact arithmetic trace Operators promoted through Orange/Legendary probation, an E130A CoreMemoryCandidate-to-Orange backfill, an E130B arithmetic text-IO transfer/no-call gauntlet, an E131 visible-equation assistant-render gauntlet, an E132 external math-text skill farm, E133 math-text route composition/no-solve assistant confirmation, E134 external math-text OOD route stress/counterexample confirmation, E135 controlled multi-route dialogue-state confirmation, E136A assistant-text skill-farm confirmation, E136B assistant-text route-composition/boundary confirmation, E136C assistant-text polished-render quick confirmation, E136D OutputTextField binary matrix confirmation, E136E idle think-tick proposal-refinement confirmation, E136F idle think-tick heldout-series confirmation, E136G adaptive idle tick-budget confirmation, E136H existing-operator refinement confirmation, E136I operator supersession/output-ledger planning confirmation, E136J shadow-variant apply/residual-prune confirmation, E136K operator replacement apply-plan confirmation, E136L runtime replacement canary confirmation, E136M runtime replacement overlay confirmation, E136N primary/secondary variant governance confirmation, E136N2 Agency Matrix arbitration smoke confirmation, and E136N3 parallel direct-write A/B smoke confirmation. E127 cycle 40 contains 382 scoped Orange/Legendary text operators with 0 tracked hard negatives, false commits, wrong-scope calls, or unsupported answers in the checkpointed evidence. E128 confirms a 320-prompt deterministic corpus/action-policy/template-render bridge with 0 unsupported answers and 0 boundary-claim violations. E129 confirms 9 scoped arithmetic trace operators with 2.7M qualified activations, 0 hard negatives, and 0 wrong-scope calls. E130A confirms 136 prior CoreMemoryCandidate operators reached Orange/LegendaryCandidate with 41.0M total qualified activations, 0 hard negatives, and 0 direct flow writes. E130B confirms those 9 arithmetic operators transfer to visible-expression text IO while hidden word problems remain no-call. E131 confirms those operators route from assistant-style visible equation surfaces seeded by a 130k-row external text pack while hidden prose-only word problems remain no-call. E132 confirms 16 scoped math-text lenses/guards promoted to Orange/LegendaryCandidate from a 215,051-row external math-text seed pack with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes. E133 confirms those 16 math-text lenses/guards compose into assistant route decisions over 176,000 route/no-solve cases with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes. E134 confirms those 16 route operators survive 208,000 OOD route cases and 48,000 counterexample cases with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes, while covering 36,275 E133-baseline OOD misses. E135 confirms those 16 route operators preserve current-turn route state over 136,000 controlled dialogue cases and 367,400 turns with 0 stale route reuse, 0 cross-thread contamination, and 0 direct Flow writes. E136A confirms 18 scoped assistant/text lenses and guards promoted from the 447,766-row E136 seed pack through Orange/Legendary mutation/prune probation with 5,521,276 qualified activations, 119,868 negative-scope cases, 0 hard negatives, and 0 direct Flow writes. E136B confirms those 18 operators compose into bounded assistant/text route stacks over 144,000 route cases, 53,000 multi-route composition cases, 72,000 boundary cases, and 18,000 negative-scope cases with all accuracy minima 1.000, 0 hard negatives, and 0 direct Flow writes. E136C confirms a deterministic polished text render quick test over 12/12 inference samples with 2/2 valid JSON outputs, 0 raw action leaks, 0 forbidden claims, and 0 direct-write claims. E136D confirms committed output text can be represented as an N x 8 binary OutputTextField with 7/7 roundtrips, 3/3 guarded rejects, 10/10 zero-fill checks, and 1/1 tamper detection. E136E confirms fixed observations can improve across idle ticks with 0 new input, 10/10 Agency-checked proposals, 4 response improvements, 8/8 non-degradation, and 8/8 OutputTextField commits. E136F confirms the same idle mechanism on a 70-case heldout series: 36/36 arithmetic heldout cases improved, 6/6 no-pocket controls preserved safe output, 4/4 direct writes were rejected, 6/6 unsupported guesses were rejected, and 70/70 outputs roundtripped through OutputTextField. E136G confirms adaptive idle scheduling: proposal records include one-more-tick recommendations, Agency decides continuation, 24/24 cases passed, adaptive execution used 33 ticks versus a 120-tick fixed baseline, 3/3 chained cases completed, and 4/4 no-pocket controls stopped at t+1. E136H confirms existing-operator refinement over 34 E132/E136A operators with 16 verified labels, 11 tightened triggers, 7 abstract-but-useful kernels, 482,637 pruned activations, 0 hold-for-more-evidence operators, 0 hard negatives, and 0 direct Flow writes. E136I confirms 27 replacement-ready selected variants, including 16 direct runtime candidates and 11 tightened challenger-required replacements, with 7 abstract lineage-required kernels, 0 destructive drops, and projected output activation delta -482,637. E136J confirms those selected variants under non-destructive shadow apply over 8,094 cycles and 33,153,024 replay rows with 22,243,205 shadow-pruned activations, 0 strict recall misses, 0 wrong-scope proxy calls, 0 hard negatives, and 0 direct Flow writes. E136K confirms a rollback-safe non-destructive apply plan: 16 direct canary-ready candidates, 11 challenger/OOD-required replacements, 7 abstract lineage holds, 0 destructive applies, and 0 runtime mutations allowed now. E136L confirms the 16 direct candidates pass runtime-canary removal/replacement replay with 0 rollback triggers while 11 challenger rows and 7 abstract lineage rows remain held. E136M materializes those 16 direct candidates as a runtime-facing overlay with 7 verified replacements, 9 light-prune overlays, 16 rollback snapshots, 0 destructive deletes, and 18,792,948 challenger candidate pruned activations explicitly not applied. E136N records the current operator set in a primary/secondary registry with 34 primaries, 34 secondaries, 16 primary-active overlays, 16 rollback secondaries, 11 challenger secondaries, 7 lineage-hold secondaries, 0 retired variants, and 0 destructive deletes. E136N2 confirms a trained Agency Matrix arbitration smoke over that surface with 118 training examples, 146 proposal-bundle cases, Agency Matrix accuracy 1.000000, 10 Flow chunks, 0 unsafe commits, and 0 challenger promotions. E136N3 confirms parallel proposal fanout should keep an Agency commit barrier: the direct-write arm reached only 0.089431 accuracy with 34 unsafe commits and 102 nondeterministic cases, while the Agency-gated arm reached 1.000000 accuracy with 0 runtime direct writes.

The E127/E128/E129/E130A/E130B/E131/E132/E133/E134/E135/E136A/E136B/E136C/E136D/E136E/E136F/E136G/E136H/E136I/E136J/E136K/E136L/E136M/E136N/E136N2/E136N3 finding is scoped operator-library, deterministic render-training, exact arithmetic trace, rank-backfill, visible-expression arithmetic text-IO, visible-equation assistant-render, math-text lens/guard, route-composition/no-solve, OOD route-stress/counterexample, controlled dialogue-state, assistant-text skill-farm, assistant-text route-composition/boundary, deterministic polished-render, output-field representation, idle proposal-refinement, heldout idle-refinement, adaptive idle scheduling, existing-operator refinement, supersession ledger planning, shadow-apply/residual-prune, apply-plan, runtime-canary, runtime-overlay, variant-governance, Agency-Matrix-arbitration, and parallel fanout/direct-write A/B evidence only. It includes deterministic operator+template text-to-text smoke and exact arithmetic expression/trace compute, but it does not claim PermaCore, TrueGolden, production API readiness, final training completion, open-domain assistant readiness, Gemma/GPT-like generation, GSM8K/MATH solving, natural-language word-problem solving, consciousness, or sentience.

For the current "what can it do as one system?" view, see
[`docs/CURRENT_CAPABILITIES.md`](docs/CURRENT_CAPABILITIES.md).

## Verification

```powershell
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test --workspace
python -m compileall -q scripts
```

Evidence checkers live under [`scripts/probes/`](scripts/probes/). Current front-door CI also checks tracked JSON/JSONL syntax, E89 naming/schema, E107-E112 sample artifacts, and the Operator rank dashboard smoke path. E113 has a full-artifact checker for local FineWeb stress runs.

Long or expensive runs must write partial outcomes continuously. End-only reporting is not acceptable for the current operating model.

## Archive Policy

The active branch surface is `main`. Historical branch heads from the June 13 cleanup are preserved under:

```text
archive/branches/2026-06-13/*
```

See [`ARCHIVE.md`](ARCHIVE.md) and the [Consolidation Archive wiki page](https://github.com/VRAXION/VRAXION/wiki/Consolidation-Archive-2026-06-13).

## License

VRAXION is released under the **VRAXION Community Source License 1.0**. It is a
custom source-available community license, not an OSI-approved open source
license.

Community use is free for personal, research, education, nonprofit, internal,
self-hosted, community-fork, benchmark, and non-monetized demo use.

Monetized third-party access to VRAXION-powered functionality is Royalty Use and
requires either compliance with the license royalty terms or a separate written
agreement. The default royalty is **19% of Attributable Net Revenue**:

```text
1% Founder Allocation
18% VRAXION Forever Prize Allocation
```

After a Founder Redirect Event, the Founder Allocation redirects to the Prize,
so the full 19% goes to the VRAXION Forever Prize Allocation.

Start here:

- [`LICENSE`](LICENSE)
- [`legal/LEGAL.md`](legal/LEGAL.md)
- [`legal/COMMERCIAL_USE_GUIDE.md`](legal/COMMERCIAL_USE_GUIDE.md)
- [`legal/VRAXION_FOREVER_PRIZE_CHARTER.md`](legal/VRAXION_FOREVER_PRIZE_CHARTER.md)
