# PILOT_SENSOR_V0_REGRESSION_001 Result

## Goal

Lock a parser-assisted PilotSensor v0 regression baseline after the nightly sensor probes.

## Pipeline

```text
raw command text
-> alias normalizer
-> structured scope-aware sensor
-> fixed evidence strength+margin guard
-> locked ADD/MUL skill execution or HOLD/REJECT
```

## Metrics

```json
{
  "action_accuracy": 1.0,
  "ambiguous_accuracy": 1.0,
  "correction_accuracy": 1.0,
  "exec_result_accuracy": 1.0,
  "false_execution_rate": 0.0,
  "known_accuracy": 1.0,
  "mention_accuracy": 1.0,
  "missed_execution_rate": 0.0,
  "negation_accuracy": 1.0,
  "primitive_drift": 0.0,
  "result_accuracy": 1.0,
  "strict_synonym_accuracy": 1.0,
  "weak_accuracy": 1.0
}
```

## Verdict

```json
[
  "PILOT_SENSOR_V0_REGRESSION_PASS"
]
```

## Failure Examples

No failures.

## Interpretation

This is the recommended v0 command sensor baseline. It is parser-assisted, not learned NLU.
Learned raw-text sensors should be measured against this baseline, especially on factor-heldout scope combinations.

## Claim Boundary

No general NLU, full PilotPulse, production VRAXION/INSTNCT, or consciousness claim.
