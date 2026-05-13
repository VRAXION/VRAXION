# PILOT_SENSOR_OPERATOR_STATE_001 Result

## Goal

Test whether the deterministic PilotSensor v0 scope/evidence mapper can be represented as an operator-based uncertainty-state update over `ADD`, `MUL`, `UNKNOWN`, and `HOLD` modes.

This is classical CPU math. `Kraus-like`, `operator`, and `collapse` are used as mathematical inspiration only.

## Setup

- Source of truth: `tools.pilot_sensor_v0` and the v0 component golden cases.
- Reference arm: current `flags_to_evidence()` plus the fixed v0 guard.
- Candidate arm: diagonal operator updates over a normalized 4-mode state.
- Contrastive negation with one surviving cue reapplies that cue operator, e.g. `do not add, multiply` resolves toward `MUL`.
- Evidence passed to the fixed guard is `(p_ADD, p_MUL, p_UNKNOWN)`.

## Metrics

```json
{
  "operator_no_correction": {
    "action_accuracy": 0.8888888888888888,
    "ambiguous_action_accuracy": 1.0,
    "ambiguous_false_execution_rate": 0.0,
    "conflict_action_accuracy": 1.0,
    "conflict_false_execution_rate": 0.0,
    "correction_action_accuracy": 0.0,
    "correction_false_execution_rate": 0.0,
    "false_execution_rate": 0.0,
    "known_action_accuracy": 1.0,
    "known_false_execution_rate": 0.0,
    "mean_entropy": 0.7046260639568217,
    "mean_evidence_margin": 0.37752388756986144,
    "mean_purity": 0.619527673816669,
    "mean_state_margin": 0.5650586504884882,
    "mention_trap_action_accuracy": 1.0,
    "mention_trap_false_execution_rate": 0.0,
    "multi_step_unsupported_action_accuracy": 1.0,
    "multi_step_unsupported_false_execution_rate": 0.0,
    "negation_action_accuracy": 1.0,
    "negation_false_execution_rate": 0.0,
    "no_evidence_action_accuracy": 1.0,
    "no_evidence_false_execution_rate": 0.0,
    "result_accuracy": 0.8888888888888888,
    "strict_synonym_action_accuracy": 0.8,
    "strict_synonym_false_execution_rate": 0.0,
    "unknown_action_accuracy": 1.0,
    "unknown_false_execution_rate": 0.0,
    "weak_action_accuracy": 1.0,
    "weak_false_execution_rate": 0.0
  },
  "operator_no_mention_suppressor": {
    "action_accuracy": 0.8888888888888888,
    "ambiguous_action_accuracy": 1.0,
    "ambiguous_false_execution_rate": 0.0,
    "conflict_action_accuracy": 1.0,
    "conflict_false_execution_rate": 0.0,
    "correction_action_accuracy": 1.0,
    "correction_false_execution_rate": 0.0,
    "false_execution_rate": 0.1111111111111111,
    "known_action_accuracy": 1.0,
    "known_false_execution_rate": 0.0,
    "mean_entropy": 0.6539332377377154,
    "mean_evidence_margin": 0.5757479702600312,
    "mean_purity": 0.6574349486596442,
    "mean_state_margin": 0.6294394059888402,
    "mention_trap_action_accuracy": 0.0,
    "mention_trap_false_execution_rate": 1.0,
    "multi_step_unsupported_action_accuracy": 1.0,
    "multi_step_unsupported_false_execution_rate": 0.0,
    "negation_action_accuracy": 1.0,
    "negation_false_execution_rate": 0.0,
    "no_evidence_action_accuracy": 1.0,
    "no_evidence_false_execution_rate": 0.0,
    "result_accuracy": 0.9444444444444444,
    "strict_synonym_action_accuracy": 1.0,
    "strict_synonym_false_execution_rate": 0.0,
    "unknown_action_accuracy": 1.0,
    "unknown_false_execution_rate": 0.0,
    "weak_action_accuracy": 1.0,
    "weak_false_execution_rate": 0.0
  },
  "operator_no_negation": {
    "action_accuracy": 0.8888888888888888,
    "ambiguous_action_accuracy": 1.0,
    "ambiguous_false_execution_rate": 0.0,
    "conflict_action_accuracy": 1.0,
    "conflict_false_execution_rate": 0.0,
    "correction_action_accuracy": 1.0,
    "correction_false_execution_rate": 0.0,
    "false_execution_rate": 0.05555555555555555,
    "known_action_accuracy": 1.0,
    "known_false_execution_rate": 0.0,
    "mean_entropy": 0.6217990736927883,
    "mean_evidence_margin": 0.4827148202719003,
    "mean_purity": 0.6707572294865085,
    "mean_state_margin": 0.6222291504564568,
    "mention_trap_action_accuracy": 1.0,
    "mention_trap_false_execution_rate": 0.0,
    "multi_step_unsupported_action_accuracy": 1.0,
    "multi_step_unsupported_false_execution_rate": 0.0,
    "negation_action_accuracy": 0.0,
    "negation_false_execution_rate": 0.5,
    "no_evidence_action_accuracy": 1.0,
    "no_evidence_false_execution_rate": 0.0,
    "result_accuracy": 0.8888888888888888,
    "strict_synonym_action_accuracy": 1.0,
    "strict_synonym_false_execution_rate": 0.0,
    "unknown_action_accuracy": 1.0,
    "unknown_false_execution_rate": 0.0,
    "weak_action_accuracy": 1.0,
    "weak_false_execution_rate": 0.0
  },
  "operator_no_weak_ambiguity": {
    "action_accuracy": 0.9444444444444444,
    "ambiguous_action_accuracy": 1.0,
    "ambiguous_false_execution_rate": 0.0,
    "conflict_action_accuracy": 1.0,
    "conflict_false_execution_rate": 0.0,
    "correction_action_accuracy": 1.0,
    "correction_false_execution_rate": 0.0,
    "false_execution_rate": 0.05555555555555555,
    "known_action_accuracy": 1.0,
    "known_false_execution_rate": 0.0,
    "mean_entropy": 0.6045698232053801,
    "mean_evidence_margin": 0.5136220513913796,
    "mean_purity": 0.6825699199683615,
    "mean_state_margin": 0.6512852334990732,
    "mention_trap_action_accuracy": 1.0,
    "mention_trap_false_execution_rate": 0.0,
    "multi_step_unsupported_action_accuracy": 1.0,
    "multi_step_unsupported_false_execution_rate": 0.0,
    "negation_action_accuracy": 1.0,
    "negation_false_execution_rate": 0.0,
    "no_evidence_action_accuracy": 1.0,
    "no_evidence_false_execution_rate": 0.0,
    "result_accuracy": 0.9444444444444444,
    "strict_synonym_action_accuracy": 1.0,
    "strict_synonym_false_execution_rate": 0.0,
    "unknown_action_accuracy": 1.0,
    "unknown_false_execution_rate": 0.0,
    "weak_action_accuracy": 0.0,
    "weak_false_execution_rate": 1.0
  },
  "operator_state_mapper": {
    "action_accuracy": 1.0,
    "ambiguous_action_accuracy": 1.0,
    "ambiguous_false_execution_rate": 0.0,
    "conflict_action_accuracy": 1.0,
    "conflict_false_execution_rate": 0.0,
    "correction_action_accuracy": 1.0,
    "correction_false_execution_rate": 0.0,
    "false_execution_rate": 0.0,
    "known_action_accuracy": 1.0,
    "known_false_execution_rate": 0.0,
    "mean_entropy": 0.5868145935041135,
    "mean_evidence_margin": 0.48733647933316243,
    "mean_purity": 0.6908237272056004,
    "mean_state_margin": 0.66144217996989,
    "mention_trap_action_accuracy": 1.0,
    "mention_trap_false_execution_rate": 0.0,
    "multi_step_unsupported_action_accuracy": 1.0,
    "multi_step_unsupported_false_execution_rate": 0.0,
    "negation_action_accuracy": 1.0,
    "negation_false_execution_rate": 0.0,
    "no_evidence_action_accuracy": 1.0,
    "no_evidence_false_execution_rate": 0.0,
    "result_accuracy": 1.0,
    "strict_synonym_action_accuracy": 1.0,
    "strict_synonym_false_execution_rate": 0.0,
    "unknown_action_accuracy": 1.0,
    "unknown_false_execution_rate": 0.0,
    "weak_action_accuracy": 1.0,
    "weak_false_execution_rate": 0.0
  },
  "v0_flags_to_evidence_reference": {
    "action_accuracy": 1.0,
    "ambiguous_action_accuracy": 1.0,
    "ambiguous_false_execution_rate": 0.0,
    "conflict_action_accuracy": 1.0,
    "conflict_false_execution_rate": 0.0,
    "correction_action_accuracy": 1.0,
    "correction_false_execution_rate": 0.0,
    "false_execution_rate": 0.0,
    "known_action_accuracy": 1.0,
    "known_false_execution_rate": 0.0,
    "mean_entropy": 0.4461157830148646,
    "mean_evidence_margin": 0.525,
    "mean_purity": 0.7407407407407407,
    "mean_state_margin": 0.6111111111111112,
    "mention_trap_action_accuracy": 1.0,
    "mention_trap_false_execution_rate": 0.0,
    "multi_step_unsupported_action_accuracy": 1.0,
    "multi_step_unsupported_false_execution_rate": 0.0,
    "negation_action_accuracy": 1.0,
    "negation_false_execution_rate": 0.0,
    "no_evidence_action_accuracy": 1.0,
    "no_evidence_false_execution_rate": 0.0,
    "result_accuracy": 1.0,
    "strict_synonym_action_accuracy": 1.0,
    "strict_synonym_false_execution_rate": 0.0,
    "unknown_action_accuracy": 1.0,
    "unknown_false_execution_rate": 0.0,
    "weak_action_accuracy": 1.0,
    "weak_false_execution_rate": 0.0
  }
}
```

## Verdict

```json
[
  "OPERATOR_STATE_POSITIVE",
  "OPERATOR_STATE_NO_BETTER_THAN_V0",
  "ABLATIONS_CAUSAL"
]
```

## Failure Examples

- `operator_no_correction` `correction_to_add` `correction`: `HOLD_ASK_RESEARCH` expected `EXEC_ADD`; failure `correction_missing_or_overhold`; operators `["ADD", "MUL", "AMBIGUITY"]`; state `[0.293716, 0.293716, 0.007808, 0.404761]`.
- `operator_no_correction` `alias_correction` `strict_synonym`: `HOLD_ASK_RESEARCH` expected `EXEC_ADD`; failure `over_hold`; operators `["ADD", "MUL", "AMBIGUITY"]`; state `[0.293716, 0.293716, 0.007808, 0.404761]`.
- `operator_no_weak_ambiguity` `weak_add` `weak`: `EXEC_ADD` expected `HOLD_ASK_RESEARCH`; failure `uncertainty_false_execution`; operators `["ADD"]`; state `[0.825806, 0.029032, 0.029032, 0.116129]`.
- `operator_no_negation` `negated_add` `negation`: `EXEC_ADD` expected `HOLD_ASK_RESEARCH`; failure `negation_false_execution`; operators `["ADD"]`; state `[0.825806, 0.029032, 0.029032, 0.116129]`.
- `operator_no_negation` `negated_add_then_mul` `negation`: `HOLD_ASK_RESEARCH` expected `EXEC_MUL`; failure `negation_missing_or_overhold`; operators `["ADD", "MUL"]`; state `[0.384962, 0.384962, 0.013534, 0.216541]`.
- `operator_no_mention_suppressor` `mention_word` `mention_trap`: `EXEC_ADD` expected `HOLD_ASK_RESEARCH`; failure `mention_false_execution`; operators `["ADD"]`; state `[0.825806, 0.029032, 0.029032, 0.116129]`.
- `operator_no_mention_suppressor` `mention_instruction` `mention_trap`: `EXEC_ADD` expected `HOLD_ASK_RESEARCH`; failure `mention_false_execution`; operators `["ADD"]`; state `[0.825806, 0.029032, 0.029032, 0.116129]`.

## Interpretation

A positive result means the current v0 evidence mapper can be reproduced as a compact uncertainty-state update and measurement/guard process.
It does not mean the system uses real quantum behavior or requires quantum hardware.

The ablations are expected to fail targeted phenomena, showing that correction, weak/ambiguity, negation, and mention-suppression operators carry causal roles in this toy mapper.

## Claim Boundary

Toy command domain only. No real quantum claim, quantum hardware requirement, general NLU, full PilotPulse, production VRAXION/INSTNCT, biology, or consciousness claim.
