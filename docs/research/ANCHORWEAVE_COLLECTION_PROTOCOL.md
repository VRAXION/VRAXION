# AnchorWeave Collection Protocol

## Goal

Collect compact human grounding annotations and convert them into canonical
AnchorCells without treating symbols as directly grounded labels.

The current storage target is `AnchorWeave-v1.0`. The locked conceptual target is
`AnchorCellCore + ProbeSpec`; use it as the authoring language until a v2 schema
or deterministic v1 exporter is approved.

## Assistant Presentation

For each candidate episode, the assistant presents:

```text
SCENE
IMPLICIT JOB
RELATIONAL BEADS
WHAT MATTERS: high / medium / low
ACTIONS
OUTCOMES
MEMORY HOOKS
COUNTERFACTUALS
ANCHOR
SYMBOL ATTACH
SYMBOL REJECT
BOUNDARY
CONFIDENCE
```

## Human Return Format

The human returns a compact annotation:

```text
GROUNDING_JUDGMENT: valid_anchor | weak_anchor | invalid_anchor | ambiguous_anchor
BEST_SUMMARY: <one or two sentences>
IMPLICIT_JOB: <what the situation is really asking the agent to do>
HIGH: <what matters most>
MEDIUM: <useful context>
LOW: <tempting but weak or wrong cue>
ACTIONS: <available and taken actions>
OUTCOMES: <predicted and actual outcomes>
COUNTERFACTUALS: <what would change the interpretation>
MEMORY_HOOK: <compact retrieval cue>
HUMAN_SOURCE_TRACE: <optional/private rugged decision trace, not model-facing>
DISTILLED_POLICY: <leak-safe reusable policy>
SYMBOL_ATTACH: <symbols that are supported>
SYMBOL_REJECT: <symbols that should not attach>
BOUNDARY: <explicit non-claim>
CONFIDENCE: <0.0-1.0>
TAGS: <positive and error tags>
FOLLOWUP: <needed or complete>
```

## Conversion Loop

1. Normalize the human annotation into `AnchorWeave-v1.0`.
2. Preserve the episode first, then graph relations, then salience.
3. Bind actions to predicted and actual outcomes.
4. Preserve memory hooks as retrieval cues, not causal rules.
5. Add counterfactuals before symbol decisions.
6. Write the implicit job, distilled policy, and claim boundary.
7. Attach supported symbols late.
8. Reject tempting unsupported symbols explicitly.
9. Keep ProbeSpec candidates, order seeds, and scoring outside canonical truth.
10. Validate locally before appending.

## Quality Checks

- Is the symbol attached only after episode, graph, salience, action/outcome,
  and counterfactual support?
- Do actions, outcomes, and memory hooks remain in the core rather than only in
  the evaluation apparatus?
- Is any raw `HumanSourceTrace` private/source material rather than polished
  rationale or model-facing chain-of-thought?
- Does the cell reject at least one plausible but unsupported overreach when
  useful?
- Is private data either absent or kept in ignored local paths?
- Does the claim boundary prevent consciousness or inner-state overclaims?
- Can the memory hook retrieve the episode without becoming the rule?

## Minimal Daily Procedure

Draft the cell locally, validate it as a single JSON file, append it to the
private JSONL database, validate the full database, then export derived views
only when needed.
