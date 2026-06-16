# E136A Assistant Text Skill Farm Mutation/Prune Orange Cycle Contract

## Purpose

E136A checks whether the local E136 assistant-text seed pack can produce new
scoped assistant/text Operator candidates and promote them through an
Orange/Legendary-style mutation/prune/no-harm gate.

The intended bridge is:

```text
E136 assistant-text seed pack
-> assistant/text support scan
-> scoped candidate cards
-> mutation / prune / challenger / reload shadow
-> negative-scope and overbroad-chatbot controls
-> Orange/LegendaryCandidate assistant-text Operators
```

## Inputs

```text
normalized dataset:
target/datasets/e136_assistant_text_seed_pack/normalized/e136_assistant_text_skill_seed.jsonl

normalized manifest:
target/datasets/e136_assistant_text_seed_pack/normalized_manifest.json

download manifest:
target/datasets/e136_assistant_text_seed_pack/download_manifest.json
```

The local dataset is intentionally under `target/` and is not committed.

## Candidate Families

E136A farms scoped assistant/text lenses and guards for:

```text
role/turn boundaries
multi-turn context continuity
instruction decomposition
summarization request binding
code instruction boundaries
refusal/source-absence boundaries
helpful/harmless preference boundaries
rejected response contrast
response format constraints
human-written instruction style
synthetic dialogue noise
reasoning instruction boundaries
math-text no-solve boundaries
safety-sensitive domain boundaries
longform generation request forms
comparison/evaluation structure
translation/multilingual boundaries
```

## Gates

E136A may confirm only if:

```text
dataset rows loaded >= 400,000
operator count = 18
Orange/LegendaryCandidate count = 18 / 18
assistant-text support min >= 4,000
qualified activation min >= 300,000

reload shadow pass rate = 1.000
negative scope pass rate = 1.000
challenger pass rate = 1.000
prune pass rate = 1.000
negative scope pass rate min = 1.000

hard negatives = 0
wrong-scope calls = 0
false commits = 0
unsupported answers = 0
boundary-claim violations = 0
direct Flow writes = 0

overbroad chatbot control wrong-scope calls > 0
```

## Boundary

This contract confirms scoped assistant/text Operator farming only. It does not
claim neural training, open-domain assistant readiness, production assistant
behavior, benchmark solving, Core, PermaCore, TrueGolden, consciousness, or
sentience.
