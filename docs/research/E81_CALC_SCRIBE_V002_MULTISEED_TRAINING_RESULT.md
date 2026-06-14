# E81 CALC-SCRIBE v002 Multi-Seed Training

```text
decision = e81_calc_scribe_v002_training_positive
checker_failure_count = 0
seeds = 16
workers = 16
generations = 12
population = 24
train_sample_size = 512
```

## Purpose

E80 found a useful near-miss:

```text
CALC-SCRIBE v001 / gsm8k_rationale_calc_marker_adapter
success = 0.962480
```

The failure was not GSM8K reasoning. It was narrow trace parsing for visible
GSM8K calculation markers such as:

```text
<<2*20*.01=.4>>
<<4*6*3=72>>
<<72*3/4=54>>
<<13=13>>
```

E81 runs a multi-seed mutation training probe over a small mechanical parser
genome. It validates visible `<<expression=result>>` markers only. It does not
solve GSM8K questions or infer hidden answers.

## Result

```text
validation_marker_mean = 0.999705
validation_marker_min = 0.998609
validation_action_mean = 0.999708
adversarial_action_min = 1.000000
accepted = 65
rejected = 127
rollback = 127
```

The 16-seed run found stable parser genomes that support:

```text
multi-operator arithmetic expressions
fractions
decimals
identity markers
parentheses where present
unicode/operator normalization
safe defer behavior on no-marker rows
adversarial no-commit behavior
```

## Remaining Failure

The remaining validation examples are concentrated around floor division:

```text
<<180//3=60>>
<<560//10=56>>
```

This is a useful next target. E81 reduced the E80 broad parser gap to a narrow
operator-support gap.

## Boundary

E81 does not claim:

```text
GSM8K solving
open-domain text reasoning
Gemma-level capability
trained model weights
production readiness
```

It confirms that CALC-SCRIBE v002 is a strong visible calculation-trace parser
candidate and that the current mutation/search harness can produce stable
multi-seed improvement on dataset-backed evidence.
