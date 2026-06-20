# E130B Arithmetic Text-IO Transfer And Word-Problem No-Call Gauntlet Result

```text
decision = e130b_arithmetic_text_io_transfer_word_problem_no_call_confirmed
next = E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET
boundary = arithmetic text-IO transfer only; not natural-language word-problem solving

operator_count = 9
transfer_pass_operator_count = 9
visible_transfer_case_count_total = 270,000
word_problem_no_call_case_count_total = 135,000
qualified_transfer_activation_total = 270,000
qualified_transfer_activation_min = 30,000
visible_transfer_accuracy_min = 1.000
word_problem_no_call_accuracy_min = 1.000

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0

overbroad_control_wrong_scope_call_total = 18,000
overbroad_control_false_commit_total = 18,000
deterministic_replay_pass = true
checker_failure_count = 0
```

## Summary

E130B confirms that the E129 scoped arithmetic trace Operators transfer into
longer text-IO wrappers when the prompt contains an explicit visible arithmetic
expression or trace.

The selected route also preserves the E129 boundary: hidden prose-only word
problems remain no-call cases. The overbroad control intentionally tried to
answer hidden word problems and produced 18,000 wrong-scope calls, so it was
rejected.

## What Was Learned

The strongest interpretation is:

```text
E129 exact arithmetic trace Operators can be used through a text-IO adapter
when the arithmetic payload is visibly present, while hidden natural-language
word problems still route to no-call.
```

The confirmed selected route is:

```text
visible_expression_text_adapter
```

It accepts explicit visible surfaces such as:

```text
direct E129 payloads
visible_arithmetic[...] spans
VISIBLE_ARITHMETIC blocks
backtick-wrapped arithmetic payloads
embedded trace markers
```

## What Is Not Claimed

E130B does not claim:

```text
natural-language word-problem solving
GSM8K solving
open-domain math reasoning
open-domain chatbot behavior
neural LLM training
production assistant readiness
PermaCore / TrueGolden
```

## Artifacts

```text
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
```

## Reproduce

```powershell
python private_probe_runner_removed --out target/pilot_wave/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet --sample-out archived_public_artifact_sample_removed
```
