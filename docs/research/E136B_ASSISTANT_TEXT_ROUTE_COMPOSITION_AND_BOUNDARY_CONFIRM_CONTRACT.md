# E136B Assistant Text Route Composition And Boundary Confirm Contract

## Purpose

E136B checks whether the 18 scoped assistant/text Operators promoted in E136A
compose into bounded assistant route decisions without becoming a general
chatbot, source hallucinator, rejected-response renderer, or direct Flow writer.

The intended bridge is:

```text
E136A assistant/text Operators
-> schema-gated assistant/text route stack
-> primary route + auxiliary route boundaries
-> negative-scope / source / rejected / direct-write controls
-> E136B route-boundary confirmation
```

## Inputs

```text
E136A artifact:
target/pilot_wave/e136a_assistant_text_skill_farm_mutation_prune_orange_cycle

normalized dataset:
target/datasets/e136_assistant_text_seed_pack/normalized/e136_assistant_text_skill_seed.jsonl

normalized manifest:
target/datasets/e136_assistant_text_seed_pack/normalized_manifest.json
```

The local dataset is intentionally under `target/` and is not committed.

## Route Families

E136B covers scoped assistant/text route families for:

```text
role / turn boundaries
multi-turn context continuity
instruction decomposition
summarization with source required
code assistance without execution claims
refusal / defer boundaries
helpful-harmless preference boundaries
rejected response contrast
source absence defer
response format constraints
human-written instruction style
synthetic dialogue noise
reasoning instruction boundary
math-text no-solve boundary
safety-sensitive domain defer
longform request form
comparison / evaluation structure
translation / multilingual boundary
```

## Gates

E136B may confirm only if:

```text
E136A source proof passed
operator count = 18
route pass operators = 18 / 18
dataset rows loaded >= 400,000 when the external dataset is present

route accuracy min = 1.000
route stack accuracy min = 1.000
primary route accuracy min = 1.000
boundary accuracy min = 1.000
multi-route composition accuracy min = 1.000
boundary-case accuracy min = 1.000
negative-scope accuracy min = 1.000

hard negatives = 0
wrong-scope calls = 0
false commits = 0
unsupported answers = 0
boundary-claim violations = 0
direct Flow writes = 0

overbroad chatbot control wrong-scope calls > 0
unsafe direct-write control direct writes > 0
source hallucination control unsupported answers > 0
```

## Boundary

This contract confirms controlled assistant/text route composition only. It does
not claim neural training, open-domain assistant readiness, production assistant
behavior, benchmark solving, Core, PermaCore, TrueGolden, consciousness, or
sentience.
