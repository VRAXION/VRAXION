# PILOT_SENSOR_V0_COMPONENTIZATION_001 Result

## Goal

Promote PilotSensor v0 into a reusable deterministic baseline component and validate every stage with golden cases.

## Public Interface

```text
normalize_aliases(text)
extract_scope_flags(normalized_text)
flags_to_evidence(scope_flags)
guard_policy(evidence)
extract_operand(text)
execute_locked_skill(action, value, operand)
run_pilot_sensor_v0(text, value=None)
```

## Metrics

```json
{
  "action_accuracy": 1.0,
  "evidence_accuracy": 1.0,
  "false_execution_rate": 0.0,
  "normalized_text_accuracy": 1.0,
  "primitive_drift": 0.0,
  "result_accuracy": 1.0,
  "scope_flag_accuracy": 1.0
}
```

## Verdict

```json
[
  "PILOT_SENSOR_V0_COMPONENT_REGRESSION_PASS"
]
```

## Failure Examples

No failures.

## Interpretation

PilotSensor v0 is now a parser-assisted command-sensor baseline, not probe-local research glue.
Learned sensors remain research-only until they beat this regression with zero false execution.

## Learned Sensor Replacement Gate

```text
action_accuracy >= 0.95
false_execution_rate = 0.000
primitive_drift = 0.000
keyword_trap_false_commit <= 0.05
no catastrophic phenomenon-tag failure
```

## Claim Boundary

Toy command domain only. No general NLU, production VRAXION/INSTNCT, full PilotPulse, biology, quantum, or consciousness claim.
