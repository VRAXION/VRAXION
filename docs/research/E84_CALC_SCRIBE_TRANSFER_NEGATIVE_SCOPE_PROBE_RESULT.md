# E84 CALC-SCRIBE Transfer And Negative Scope Probe

```text
decision = e84_calc_scribe_transfer_negative_scope_confirmed
checker_failure_count = 0
seeds = 16
workers = 16
case_count = 306356
```

## Purpose

E83 promoted CALC-SCRIBE v003 as a governed LocalGolden scoped Pocket for
visible calculation-trace marker validation.

E84 tests whether that capability transfers across visible marker formats
without expanding into GSM8K / natural-language word-problem solving.

## Result

```text
primary_validation_action_min = 1.000000
primary_validation_valid_commit_min = 1.000000
primary_validation_no_marker_no_call_min = 1.000000
primary_adversarial_action_min = 1.000000
primary_adversarial_false_call_max = 0.000000

native_validation_action_min = 0.541220
overbroad_false_call_max = 1.000000
always_false_commit_max = 0.087091
```

The transfer router handled these visible trace formats:

```text
<<expr=result>>
<< expr = result >>
[calc expr=result]
calc: expr -> result
expr = result
unicode operator lines such as 48÷2 = 24
context-wrapped visible native markers
```

It also rejected malformed or wrong visible traces and no-called rows that had
only word-problem text, only a final answer, or rationale text with the calc
markers stripped.

## Interpretation

The E84 positive is a transfer/scope result, not a new reasoning claim.

```text
CALC-SCRIBE v003 native-only:
  validates the original native marker family but does not transfer formats.

CALC-SCRIBE transfer router:
  maps alternate visible marker surfaces back into the same scoped validation
  behavior.

Negative scope:
  no visible calc marker -> NO_CALL, not hidden answer inference.
```

The negative controls isolate why the guard matters:

```text
overbroad word-problem solver control:
  false_call_max = 1.000000

always-commit control:
  false_commit_max = 0.087091
```

## Boundary

Allowed:

```text
validate visible calc traces
reject invalid visible calc traces
no-call on word-problem text without a visible calc trace
```

Not claimed:

```text
GSM8K solving
open-domain reasoning
natural-language word-problem solving
Core / True Golden promotion
Gemma-level capability
production readiness
```
