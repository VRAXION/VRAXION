# AnchorWeave AWFT-001 A/B Test

## Purpose

AWFT-001 tests whether AnchorWeave structure improves grounding-like behavior
over plain Q&A, rich prose, and shuffled structured controls on the same
synthetic navigation scenarios.

This is a local format-transfer scaffold. It is not evidence that AnchorWeave is
useful at scale, and it is not a consciousness claim.

## Hypothesis

For navigation / route-memory disambiguation, a model trained on AnchorWeave
input should choose diagnostic first actions, rank salience, reject shortcut
symbols, answer counterfactuals, and avoid premature commitment better than the
same base model trained on plain Q&A or rich prose.

The target behaviors are:

- partial landmark overlap is not the same as same place
- memory hook is not a causal rule
- vivid detail is not necessarily salience
- missing cue does not automatically reject a place
- diagnostic action should beat premature commitment
- symbols attach late

## Experimental Arms

All arms use the same structured completion target so the test measures input
representation, not output-format learning.

- `plain_qa`: compact observation and question.
- `rich_prose`: longer narrative carrying the same observable facts.
- `anchorweave_sft`: structured episode, graph, salience candidates, actions,
  counterfactual questions, and output request.
- `shuffled_anchorweave_control`: structured input that looks normal, with an
  intentionally bad target. The prompt and completion text do not disclose that
  it is a control.

## Split Design

The generator creates one domain only: navigation / route-memory
disambiguation.

Splits:

- train: 40 scenarios
- dev: 10 scenarios
- test: 40 scenarios

Families:

- exact-match
- near-miss
- viewpoint-shift
- misleading-single-landmark

Test scenarios use held-out landmark/object/detail pools so the test is not a
renaming of train cases. The underlying structures remain the same.

## Leakage Controls

Canonical scenario rows may include `hidden_truth`, but training and eval prompts
must not include `hidden_truth`, gold labels, `correct_first_action`, or any
schema field that directly reveals the target.

Counterfactual labels are structured enums:

- `strengthens_match`
- `weakens_match`
- `neutral`
- `requires_disambiguation`
- `rejects_match`
- `confirms_match`

Commitment labels are:

- `confirmed_same_place`
- `rejected_same_place`
- `defer_and_disambiguate`
- `premature_commit`

## Metrics

The evaluator reports:

- `first_action_accuracy`
- `salience_high_f1`
- `salience_low_f1`
- `symbol_attach_f1`
- `symbol_reject_f1`
- `counterfactual_accuracy`
- `commitment_accuracy`
- `overcommitment_rate`

Overcommitment is counted when the prediction is `confirmed_same_place` or
`premature_commit` while the gold commitment is not `confirmed_same_place`.

## Scoreboard

| Arm | Action acc | Salience high F1 | Salience low F1 | Symbol reject F1 | Counterfactual acc | Overcommit down |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No fine-tune | ? | ? | ? | ? | ? | ? |
| Plain Q&A | ? | ? | ? | ? | ? | ? |
| Rich prose | ? | ? | ? | ? | ? | ? |
| AnchorWeave | ? | ? | ? | ? | ? | ? |
| Shuffled AnchorWeave | ? | ? | ? | ? | ? | ? |

Desired pattern:

```text
AnchorWeave > Rich prose > Plain Q&A > No fine-tune
AnchorWeave >> Shuffled AnchorWeave
```

If rich prose matches AnchorWeave, structured prose may already carry much of
the signal. If shuffled AnchorWeave matches AnchorWeave, the model may be
learning format rather than correct relational content.

## Local Commands

Generate synthetic artifacts:

```bash
python tools/anchorweave/generate_awft001.py --out target/anchorweave/awft001 --seed 2026
```

Self-check evaluator with labels as predictions:

```bash
python tools/anchorweave/evaluate_awft001.py --labels target/anchorweave/awft001/eval_labels.jsonl --predictions target/anchorweave/awft001/eval_labels.jsonl
```

Generated artifacts live under `target/`, which is ignored by git.
