# ANCHOR-MINI-001 / Relational Anchor Toy Core

Status: `preregistered_toy_core_probe`

This probe tests the smallest useful AnchorCell claim:

```text
Does an explicit relational anchor variable help a tiny learner generalize a
decision rule across held-out object/site combinations?
```

This is not an LLM prompt test, not training data collection, and not an
`AnchorWeave-v1.0` export.

## Task

Each example contains a small object, two possible sites, and an intent:

```text
intent = next_step | put_away
candidate A latent role = use_site | storage_site
candidate B latent role = the other role
```

Gold rule:

```text
if intent == next_step: choose the use_site candidate
if intent == put_away: choose the storage_site candidate
```

The base arm sees object/site identifiers and noisy surface-prior features, but
not the candidate roles. Held-out eval uses unseen object and site IDs, so the
base model cannot solve by memorizing IDs.

The anchor arm adds a relational feature:

```text
candidate_matches_intent
```

This is the minimal toy equivalent of an AnchorCell relational decision hook.
It is intentionally the smallest derived anchor signal, not a test of whether a
model can infer that signal from natural language or raw world state.

## Arms

```text
BASE
ANCHOR
SHUFFLED_ANCHOR
NOISE_ANCHOR
```

Controls:

- `SHUFFLED_ANCHOR` permutes anchor features across examples.
- `NOISE_ANCHOR` uses deterministic random bits.
- These controls test whether any extra feature helps, or only the correct
  relational anchor helps.

## Models

```text
logistic_regression
tiny_mlp
```

Both models run on CPU with deterministic seeds.

## Metrics

```text
train_accuracy
eval_accuracy
next_step_accuracy
put_away_accuracy
shortcut_trap_rate
anchor_ablation_accuracy
anchor_reliance_drop
```

`shortcut_trap_rate` measures how often the model chooses the stronger
surface-prior candidate when that candidate is not gold.

## Verdict

`ANCHOR_MINI_001_POSITIVE` only if both model families pass:

```text
ANCHOR eval_accuracy >= 0.90
ANCHOR eval_accuracy >= max(BASE, SHUFFLED_ANCHOR, NOISE_ANCHOR) + 0.25
ANCHOR anchor_reliance_drop >= 0.20
SHUFFLED_ANCHOR eval_accuracy <= BASE + 0.10
NOISE_ANCHOR eval_accuracy <= BASE + 0.10
```

Otherwise:

```text
ANCHOR_MINI_001_NEGATIVE
```

## Claim Boundary

Positive evidence means the relational anchor variable is a learnable,
generalizing signal in a toy setting. It does not prove natural-language
AnchorCells, LLM prompt injection, consciousness, or grounding at scale.
It also does not prove that a model can derive `candidate_matches_intent`
without being given the relational hook.

Negative evidence means the proposed anchor signal fails even in the smallest
auditable learning setup and should not be scaled.
