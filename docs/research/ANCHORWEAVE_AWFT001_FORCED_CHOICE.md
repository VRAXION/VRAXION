# AWFT-001-FC Forced-Choice Protocol

AWFT-001-FC is a deterministic follow-up to AWFT-001 that removes generated JSON from the measurement path. The model never produces task output. Instead, it scores natural-language candidate answers, and the candidate with the lowest average candidate-token negative log-likelihood wins.

## Goal

Test whether AnchorWeave training changes model preference toward grounded route-memory decisions, compared with plain Q&A, rich prose, and shuffled structured controls.

The test is local and narrow. It does not prove AnchorWeave is useful at scale. It checks whether one controlled navigation / route-memory task family shows a measurable format-transfer signal.

## Why Forced Choice

The JSON-generation path exposed a format-compliance failure: the model could run through the pipeline, but it did not reliably emit usable structured predictions. That blocks interpretation because invalid output can hide whether the model understood the scene.

Forced choice avoids that confound:

```text
scenario prompt + candidate answer -> candidate-token NLL
```

No parser, schema, or generated field names are involved.

## Arms

All training arms use the same base model, seed, LoRA config, example count, epochs, max sequence length, and completion-only loss.

```text
zero_shot
plain_qa_nl
rich_prose_nl
anchorweave_nl
shuffled_anchorweave_nl
```

The non-corrupted training arms share the same semantic target completion rendered as natural language. The shuffled arm uses fluent but semantically wrong completions and must not disclose that it is a control.

## Candidate Design

Each evaluation scenario has four candidate answers:

```text
correct_grounded_action
single_landmark_shortcut
premature_rejection
incidental_context_rule
```

Every candidate uses the same answer rhythm:

```text
The best first move is to ... because ...
```

Candidate order is randomized deterministically by seed. Evaluation uses `candidate_id`, never list position.

Candidate text must not contain labels such as `correct`, `wrong`, or `trap`. Evaluation wording is paraphrased relative to training completions to reduce template memorization.

## Prior Correction

Forced choice has a separate confound: a candidate sentence may be more likely because it is more common or smoother, not because it matches the scenario.

For every candidate, AWFT-001-FC records:

```text
scenario_candidate_nll
prior_candidate_nll
prior_corrected_score = scenario_candidate_nll - prior_candidate_nll
```

The prior prompt is:

```text
You are answering a navigation decision question.
```

Metrics are reported both raw and prior-corrected. A positive result must pass both verdicts.

## Metrics

Primary metrics:

```text
candidate_accuracy_raw
candidate_accuracy_prior_corrected
mean_gold_margin_raw
mean_gold_margin_prior_corrected
```

Trap and commitment metrics:

```text
exact_match_confirm_rate_raw
exact_match_confirm_rate_prior_corrected
defer_accuracy_raw
defer_accuracy_prior_corrected
reject_accuracy_raw
reject_accuracy_prior_corrected
shortcut_trap_rate_raw
shortcut_trap_rate_prior_corrected
overreject_trap_rate_raw
overreject_trap_rate_prior_corrected
incidental_context_trap_rate_raw
incidental_context_trap_rate_prior_corrected
```

If there are no gold reject rows, reject accuracy is reported as `null`.

Family breakdowns are reported for:

```text
exact_match
near_miss
viewpoint_shift
misleading_single_landmark
```

The generator also reports `candidate_length_imbalance_count` for scenarios where the longest candidate has more than 1.25 times the token count of the shortest candidate.

## Verdict

Raw verdict is positive only if all are true:

```text
anchorweave_nl candidate_accuracy_raw >= max(plain_qa_nl, rich_prose_nl) + 0.10
anchorweave_nl candidate_accuracy_raw >= shuffled_anchorweave_nl + 0.20
anchorweave_nl shortcut_trap_rate_raw <= rich_prose_nl shortcut_trap_rate_raw
anchorweave_nl exact_match_confirm_rate_raw >= 0.60
```

The same rules apply to prior-corrected metrics.

Final verdict:

```text
POSITIVE only if RAW_VERDICT and PRIOR_CORRECTED_VERDICT are both POSITIVE.
Otherwise NEGATIVE.
```

## Boundaries

AWFT-001-FC does not train or evaluate real private AnchorCells. It writes generated artifacts only under ignored `target/anchorweave/awft001_fc/` paths and must not write under `data/anchorweave/cells/`.

The test checks a narrow local format-transfer signal. A negative result means this setup did not show the required signal; it does not settle the broader AnchorWeave research question.
