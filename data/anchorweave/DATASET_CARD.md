# AnchorWeave Dataset Card

## Purpose

AnchorWeave is a continuously growing append-only dataset system for operational
symbol-grounding research through relational episodic anchors. Each AnchorCell
preserves a concrete situation and the relations, salience, actions, outcomes,
counterfactuals, memory hooks, abstractions, and human annotations needed to
study late symbol attachment.

## Non-purpose

AnchorWeave is not a direct Q/A dataset, not a dictionary of concept
definitions, and not a claim that symbols are grounded by text labels alone.

## Data Structure

Current canonical storage is rich `AnchorWeave-v1.0` JSONL. Each line is one
AnchorCell with:

- episode
- relational graph
- salience map
- actions
- predicted and actual outcomes
- counterfactuals
- memory hooks
- pre-symbol abstraction
- symbol attach/reject
- human grounding annotation
- outcome follow-up

The locked conceptual standard for future work is:

```text
AnchorCellCore
  Situation
  ImplicitJob
  RelationalModel
  SalienceMap
  Actions
  Outcomes
  MemoryHooks
  HumanSourceTrace optional/private
  DistilledPolicy
  CounterfactualChecks
  SymbolAttachReject
  ClaimBoundary
  WorldTruth private/eval-hidden

ProbeSpec derived
  PromptArms
  CandidateActions
  ChoiceOrderSeeds
  FreeResponseTaxonomy
  PairwiseTrapProbes
  ParaphraseVariants
  ScoringSpec
  PassCriteria
```

Derived SFT, DPO, reward, eval, graph, and ProbeSpec views are generated from
canonical cells.

## Privacy Boundary

The live growing database is private-by-default. Private and personal data must
stay in ignored local paths unless explicitly sanitized and reviewed for release.

## Intended Uses

- Researching relational episodic anchors for symbol grounding.
- Auditing whether symbol attachment is supported by episode, salience,
  actions, outcomes, and counterfactuals.
- Producing derived training/evaluation views after privacy review.
- Studying failure modes such as shortcut salience, premature symbol binding,
  and unsupported claim boundaries.

## Not Intended Uses

- Training on private or personal data without sanitization.
- Treating cells as final concept definitions.
- Treating high model accuracy as grounded understanding without counterfactual
  and action/outcome support.
- Making claims about consciousness or inner subjective states.

## Claim Boundary

AnchorWeave is not a dataset of consciousness. It is a dataset for operational
symbol-grounding research via relational episodic anchors.

## Known Risks

- Private data leakage if raw daily annotations are committed.
- Overgeneralization from a single episode to a broad symbol.
- Surface-pattern shortcut learning if salience and counterfactuals are weak.
- Measurement artifacts if ProbeSpec candidates, order seeds, or scoring are
  mistaken for canonical cell truth.
- Unsupported inner-state claims if the boundary field is ignored.
- Premature symbol attachment before stable abstractions are available.

## Versioning

Current schema: `AnchorWeave-v1.0`

`Core + ProbeSpec` is the locked conceptual standard, not a storage migration.
Schema changes should create a new schema file, update the dataset card, and
include migration or exporter notes. Existing JSONL cells should remain
append-only unless a new revision is appended with explicit provenance.
