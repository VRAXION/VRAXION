# E130B Arithmetic Text-IO Transfer And Word-Problem No-Call Gauntlet Contract

## Purpose

E130B takes the nine scoped E129 arithmetic trace Operators and tests whether
they transfer into longer text-IO wrappers while preserving the E129 boundary:
visible arithmetic expressions/traces are callable, hidden natural-language
word problems are no-call.

This is scoped arithmetic text-IO transfer. It is not GSM8K solving, not
natural-language word-problem solving, not open-domain reasoning, and not
neural LLM training.

## Input

Preferred source artifact:

```text
target/pilot_wave/e129_arithmetic_trace_orange_legendary_probation/
```

Sample fallback:

```text
archived_public_artifact_sample_removed
```

Required source state:

```text
source_decision = e129_arithmetic_trace_orange_legendary_probation_confirmed
source_operator_count = 9
source_orange_legendary_candidate_count = 9
source_hard_negative_total = 0
source_wrong_scope_call_total = 0
```

## Transfer Gate

Each E129 arithmetic Operator must pass:

```text
visible_transfer_accuracy = 1.000
word_problem_no_call_accuracy = 1.000
hard_negative = 0
false_commit = 0
wrong_scope_call = 0
unsupported_answer = 0
direct_flow_write = 0
reload_shadow_pass = true
negative_scope_pass = true
challenger_pass = true
prune_pass = true
```

The selected route must also beat the overbroad word-problem control:

```text
overbroad_control_wrong_scope_call > 0
```

## Output

E130B does not create a new broad math rank. It annotates the E129 arithmetic
Operators with:

```text
watch_state = E130BTextIOTransferConfirmed
selected_route = visible_expression_text_adapter
```

## Required Artifacts

```text
run_manifest.json
input_e129_report.json
transfer_report.json
word_problem_no_call_report.json
operator_transfer_results.json
variant_report.json
row_level_samples.jsonl
progress.jsonl
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
checker_summary.json
```

## Decision Labels

```text
e130b_arithmetic_text_io_transfer_word_problem_no_call_confirmed
e130b_arithmetic_text_io_transfer_word_problem_no_call_rejected
```

## Reproduce

```powershell
python private_probe_runner_removed --out target/pilot_wave/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet --sample-out archived_public_artifact_sample_removed
```

## Boundary

E130B validates explicit visible arithmetic expression/trace handling inside
longer text. It does not solve hidden prose-only word problems.
