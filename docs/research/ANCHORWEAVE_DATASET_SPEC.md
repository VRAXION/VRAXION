# AnchorWeave Dataset Specification

## Core Thesis

Do not ground the symbol directly. Store relational episodic anchors from which
stable abstractions can emerge; symbols attach late.

AnchorWeave exists because a concept label is usually too thin to carry the
grounding evidence. The system stores the concrete episode, the relations inside
that episode, what mattered, which actions were available, what outcomes were
predicted and observed, which counterfactuals would change the interpretation,
and what memory hook can later retrieve the episode. Only after that chain is
stable should a symbol be attached or rejected.

## Transformation

```text
episode
  -> relational graph
  -> salience
  -> action/outcome
  -> counterfactuals
  -> memory hook
  -> pre-symbol abstraction
  -> symbol attach/reject
```

## Canonical Unit: AnchorCell

An AnchorCell is one append-only JSON object. The required top-level fields are:

- `schema_version`
- `cell_id`
- `revision`
- `created_at`
- `provenance`
- `episode`
- `relational_graph`
- `salience`
- `actions`
- `outcomes`
- `counterfactuals`
- `memory_hooks`
- `abstraction`
- `symbol_binding`
- `human_annotation`
- `outcome_followup`

## Field Semantics

`provenance` records source, author, privacy tier, collection method, and whether
the cell contains private data.

`episode` records the concrete memory-like situation. It should be specific
enough to preserve the scene but sanitized before public release.

`relational_graph` records nodes and edges that make the episode usable as a
grounding anchor. The graph should distinguish causal, spatial, temporal,
affordance, comparison, and evidence relations when possible.

`salience` marks what matters at high, medium, and low levels. Low salience is
not useless; it often captures tempting but wrong cues.

`actions` records actions that were available and actions actually taken.

`outcomes` records predicted and actual outcomes so that grounding can be tied
to action/outcome structure rather than label matching.

`counterfactuals` records what would change the interpretation. Weak or missing
counterfactuals should block strong symbol attachment.

`memory_hooks` stores compact retrieval cues that can bring the episode back
without turning incidental context into the rule.

`abstraction` stores the pre-symbol statement, abstraction level, claim boundary,
and explicit non-claims.

`symbol_binding` records late attach/reject decisions. A rejected symbol is as
important as an attached symbol because it protects the claim boundary.

`human_annotation` records the grounding judgment, confidence, best summary,
positive tags, error tags, and boundary notes.

`outcome_followup` records later observations, status, and whether follow-up is
still needed.

## Append-only Rule

Canonical data is append-only JSONL. If a cell needs correction, append a new
revision with the same conceptual anchor and updated provenance. Do not silently
edit historical private cells once they have been used for training or analysis.

## Privacy Rule

The live database is private-by-default. Public examples must be synthetic or
sanitized. Private working data belongs in ignored paths.

## Derived Views

Derived views are generated from canonical cells:

- SFT: prompt/completion pairs for anchor, salience, and symbol-boundary tasks.
- DPO: preference pairs from future `candidate_outputs`.
- Reward: symbol attach/reject scores and labels.
- Eval: future held-out checks for salience, counterfactual, and claim-boundary
  failures.
- Graph: flattened relational edges for graph analytics.

Derived views must not become the source of truth. Regenerate them from the
canonical append-only cells.

## Claim Boundary

AnchorWeave supports operational symbol-grounding research through relational
episodic anchors. It does not claim to measure consciousness, subjective
experience, or inner states. Any such claim should be tagged and rejected unless
separately supported by an explicit, defensible protocol.
