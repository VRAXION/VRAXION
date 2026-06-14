# E85 CALC-SCRIBE Mixed Stream Inference Integration

```text
decision = e85_calc_scribe_mixed_stream_inference_integration_confirmed
checker_failure_count = 0
seeds = 16
workers = 16
case_count = 170366
```

## Purpose

E84 confirmed transfer across visible calc-trace marker formats.

E85 tests runtime integration in a mixed input stream:

```text
call CALC-SCRIBE when visible calc trace exists
reject invalid visible calc traces
no-call natural text / word problems without visible calc trace
```

## Result

```text
primary_validation_route_min = 1.000000
primary_validation_action_min = 1.000000
primary_validation_false_call_max = 0.000000
primary_validation_false_commit_max = 0.000000
primary_adversarial_action_min = 1.000000
primary_adversarial_false_commit_max = 0.000000
primary_mean_active_set_size = 2.818

native_validation_action_min = 0.669161
full_scan_false_call_max = 1.000000
alias_false_call_max = 0.979896
```

## Interpretation

The managed active set integration correctly routes CALC-SCRIBE only when a
visible calc trace exists. It rejects wrong visible traces and no-calls:

```text
GSM8K word problem text without trace markers
GSM8K final answer only
GSM8K rationale with markers stripped
FineWeb natural text
FineWeb numeric-looking natural text without explicit trace framing
```

The failed controls show why governance is needed:

```text
native-only active set:
  misses transfer formats

full library scan without scope guard:
  false-call max = 1.000000

alias / numeric-keyword router:
  false-call max = 0.979896
```

## Guard Detail

E85 keeps the E84 transfer formats but prevents long natural text bodies from
being treated as standalone plain equations. In long text, the router requires
explicit framing such as:

```text
<<expr=result>>
calc: expr -> result
trace: expr -> result
```

Standalone `expr = result` remains valid only as a short isolated trace payload.

## Boundary

This is visible calculation-trace validation only.

Not claimed:

```text
GSM8K solving
open-domain reasoning
natural-language word-problem solving
Core memory
True Golden
production readiness
```
