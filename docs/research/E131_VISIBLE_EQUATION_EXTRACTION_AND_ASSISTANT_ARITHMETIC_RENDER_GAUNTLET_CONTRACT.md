# E131 Visible Equation Extraction And Assistant Arithmetic Render Gauntlet Contract

## Purpose

E131 takes the nine E129/E130B scoped arithmetic trace Operators and tests
whether they can route from assistant-style visible equation surfaces seeded by
an external text dataset while preserving the E130B boundary: visible arithmetic
expressions/traces are callable, hidden prose-only word problems are no-call.

This is visible equation extraction and deterministic assistant rendering. It is
not GSM8K solving, not natural-language word-problem solving, not open-domain
reasoning, and not neural LLM training.

## Input

Preferred source artifact:

```text
target/pilot_wave/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet/
```

Sample fallback:

```text
docs/research/artifact_samples/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet/
```

External normalized seed pack:

```text
target/datasets/e131_wifi_seed_pack/normalized/e131_mixed_skill_seed.jsonl
```

Required source state:

```text
source_decision = e130b_arithmetic_text_io_transfer_word_problem_no_call_confirmed
source_transfer_pass_operator_count = 9
source_visible_transfer_accuracy_min = 1.000
source_word_problem_no_call_accuracy_min = 1.000
source_hard_negative_total = 0
source_wrong_scope_call_total = 0
```

## Transfer Gate

Each arithmetic Operator must pass:

```text
visible_equation_extraction_accuracy = 1.000
word_problem_no_call_accuracy = 1.000
hard_negative = 0
false_commit = 0
wrong_scope_call = 0
unsupported_answer = 0
boundary_claim_violation = 0
direct_flow_write = 0
reload_shadow_pass = true
negative_scope_pass = true
challenger_pass = true
prune_pass = true
```

The selected route must also beat both controls:

```text
e130b_baseline_visible_miss > 0
overbroad_control_wrong_scope_call > 0
```

## Output

E131 annotates the E129/E130B arithmetic Operators with:

```text
watch_state = E131VisibleEquationAssistantRenderConfirmed
selected_route = e131_visible_equation_assistant_adapter
```

## Required Artifacts

```text
run_manifest.json
dataset_report.json
input_e130b_report.json
extraction_report.json
assistant_render_report.json
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
e131_visible_equation_extraction_assistant_arithmetic_render_confirmed
e131_visible_equation_extraction_assistant_arithmetic_render_rejected
```

## Reproduce

```powershell
python scripts/probes/run_e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet.py
```

## Boundary

E131 validates explicit visible arithmetic equation/trace extraction and
deterministic assistant rendering. It does not solve hidden prose-only word
problems.
