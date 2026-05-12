# PILOT_SENSOR Nightly Summary - 2026-05-12

## Summary

Nightly target:

```text
raw command text
-> evidence/scope sensor
-> fixed strength+margin guard
-> execute / reject / hold
-> locked ADD/MUL skill
```

Main decision:

```text
The guard and locked skills are not the current blocker.
The blocker is robust raw text -> scope/evidence extraction.
```

## Runs

| Probe | Commit | Result |
|---|---|---|
| `PILOT_SENSOR_SCOPE_STACK_NIGHTLY_001` | `f0dbc0d` | Learned sensors nearly passed, but weak/ambiguous false commits remained. |
| `PILOT_SENSOR_WEAK_AMBIGUITY_CALIBRATION_001` | `1281d5d` | Augmented weak/ambiguous coverage solved the original eval. |
| `PILOT_SENSOR_AUGMENTED_ROBUSTNESS_001` | `f48c06b` | New stress forms broke the augmented sensor; negation/correction variants remained weak. |
| `PILOT_SENSOR_SYSTEMATIC_COVERAGE_001` | `4faa126` | Systematic template coverage solved the stress suite at 1.000 action / 0 false commit. |
| `PILOT_SENSOR_FACTOR_HELDOUT_001` | `489eefb` | Heldout factor combinations failed; scope-stack best action was 0.745 with 0.170 false commit. |
| `PILOT_SENSOR_STRUCTURED_FEATURES_001` | `b7f3a70` | Explicit scope features nearly solved factor-heldout; remaining issue was learned mapping on conflict evidence. Oracle deterministic mapper passed. |
| `PILOT_SENSOR_LOCKED_SKILL_INTEGRATION_001` | `74a8c81` | Structured sensor and learned systematic sensor executed frozen ADD/MUL with 1.000 result accuracy and 0 drift. |
| `PILOT_SENSOR_LEXICON_EXTENSION_001` | `1edd11c` | Strict synonym failures were solved by explicit alias lexicon coverage. |
| `PILOT_SENSOR_GUARD_COMPENSATION_001` | `883e019` | Threshold tuning could not fix factor-heldout sensor scope errors. |
| `PILOT_SENSOR_V0_REGRESSION_001` | `37a13c7` | Parser-assisted v0 baseline passed the combined stress/factor/alias execution suite. |

## Key Findings

### 1. Fixed Guard Is Useful But Not Sufficient

`PILOT_TOPK_GUARD_001` already showed that softmax argmax is a brittle commit policy, while strength+margin/top-K guards give a principled HOLD path. The new guard compensation sweep adds the next result:

```text
No strength/margin threshold setting fixed learned sensor scope errors.
```

If the sensor emits strong but wrong evidence, downstream thresholds cannot reliably recover.

### 2. Coverage Solves Template Stress, Not Factor Generalization

Systematic template generation solved the robustness stress suite:

```text
direct_evidence_char_ngram_mlp_systematic:
  action = 1.000
  false_commit = 0.000

scope_stack_char_ngram_mlp_systematic:
  action = 1.000
  false_commit = 0.000
```

But factor-heldout combinations did not generalize:

```text
scope_stack_char_ngram_mlp_factor:
  action = 0.745
  false_commit = 0.170
  negation = 0.500
  correction = 0.500
```

Interpretation:

```text
The learned n-gram sensor can memorize broad template families.
It does not yet reliably compose scope markers and operation cues.
```

### 3. Explicit Scope/Event Features Localize The Missing Layer

The structured feature probe used explicit normalized flags:

```text
add_cue / mul_cue / unknown_cue
weak_marker / ambiguity_marker
mention_only / multi_step_unsupported
negation_add / negation_mul
correction_to_add / correction_to_mul / correction_to_unknown
```

Result:

```text
structured_feature_linear_student:
  action = 0.966
  false_commit = 0.034
  weak = 1.000
  negation = 1.000
  correction = 1.000

oracle_flags_mapper:
  action = 1.000
  false_commit = 0.000
```

Interpretation:

```text
Explicit scope flags are enough.
Learned flag -> evidence mapping still has edge cases.
The safest v0 design is deterministic flags -> evidence mapping.
```

### 4. Locked Skill Execution Path Works

The integrated execution probe showed:

```text
structured_rule_sensor:
  action = 1.000
  result = 1.000
  false_execution = 0.000
  primitive_drift = 0.000

learned_systematic_char_sensor:
  action = 1.000
  result = 1.000
  false_execution = 0.000
  primitive_drift = 0.000

keyword_sensor:
  action = 0.413
  result = 0.455
  false_execution = 0.355
```

Interpretation:

```text
When the sensor produces reliable evidence, the fixed guard and locked skills compose correctly.
Keyword extraction remains unsafe.
```

### 5. Strict Synonym Failure Is Lexicon Coverage

Base structured sensor failed:

```text
increment by 9
product with 9
halve it
```

Alias normalizer passed:

```text
alias_extended_sensor:
  action = 1.000
  strict_synonym = 1.000
  scope_alias = 1.000
  false_commit = 0.000
```

Interpretation:

```text
This is lexical coverage, not semantic generalization.
Without pretrained semantics or explicit aliases, strict unseen synonyms remain unfair as a main fail.
```

## Decision

Recommended architecture for the next implementation step:

```text
raw command text
-> alias normalizer
-> explicit scope/event parser
-> deterministic flags -> evidence mapper
-> fixed strength+margin/top-K guard
-> locked skill execution
```

Do not keep trying to solve this with raw n-gram MLPs alone. The evidence says they can pass covered templates but fail factor-heldout scope composition.

The parser-assisted v0 regression now passes:

```text
PILOT_SENSOR_V0_REGRESSION_001:
  action_accuracy = 1.000
  result_accuracy = 1.000
  false_execution = 0.000
  primitive_drift = 0.000
```

This should be treated as the current command-sensor baseline, not as a learned-NLU result.

## Next Actions

1. Build `PilotSensor v0` as a parser-assisted concept bottleneck:
   ```text
   text -> flags -> evidence -> guard
   ```

2. Add a regression suite from the probes:
   ```text
   weak / ambiguous / negation / correction / mention traps / aliases / multi-step unsupported
   ```

3. Keep learned sensor research separate:
   ```text
   learned parser or pretrained encoder -> flags
   ```

4. Keep deterministic downstream logic:
   ```text
   flags -> evidence mapper
   guard thresholds
   locked skill execution
   ```

## Claim Boundary

This is toy command-text evidence only.

No claim of:

```text
general natural-language understanding
full PilotPulse
full VRAXION/INSTNCT
production readiness
consciousness
biology
quantum behavior
```
