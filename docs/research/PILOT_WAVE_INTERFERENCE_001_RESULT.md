# PILOT_WAVE_INTERFERENCE_001 Result

## Goal

Test whether phase-like interference is a useful primitive for command-scope decisions.

The probe compares three deterministic representations over the same scope flags:

```text
positive_evidence
signed_amplitude
complex_phase
```

## Metrics

| Arm | Action Accuracy | False Execution | Destructive | Correction | Weak Hold | Mention |
|---|---:|---:|---:|---:|---:|---:|
| `positive_evidence` | `0.375` | `0.375` | `0.000` | `0.000` | `0.000` | `0.000` |
| `signed_amplitude` | `0.875` | `0.000` | `1.000` | `0.000` | `1.000` | `1.000` |
| `complex_phase` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` |

## Verdict

```json
[
  "PILOT_WAVE_COMPLEX_PHASE_POSITIVE",
  "POSITIVE_EVIDENCE_WEAK",
  "SIGNED_AMPLITUDE_PARTIAL"
]
```

## Failure Examples

### `positive_evidence`

- `do_not_add`: expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`; tags `['destructive_interference', 'negation']`; state `{'ADD': 0.9, 'MUL': 0.0, 'UNKNOWN': 0.0, 'HOLD': 0.0}`.
- `do_not_multiply`: expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`; tags `['destructive_interference', 'negation']`; state `{'ADD': 0.0, 'MUL': 0.9, 'UNKNOWN': 0.0, 'HOLD': 0.0}`.
- `not_add_then_multiply`: expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`; tags `['destructive_interference', 'refocus']`; state `{'ADD': 0.9, 'MUL': 0.9, 'UNKNOWN': 0.0, 'HOLD': 0.0}`.
- `not_multiply_then_add`: expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`; tags `['destructive_interference', 'refocus']`; state `{'ADD': 0.9, 'MUL': 0.9, 'UNKNOWN': 0.0, 'HOLD': 0.0}`.
- `maybe_add`: expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`; tags `['weak_hold']`; state `{'ADD': 0.9, 'MUL': 0.0, 'UNKNOWN': 0.0, 'HOLD': 0.0}`.
- `maybe_multiply`: expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`; tags `['weak_hold']`; state `{'ADD': 0.0, 'MUL': 0.9, 'UNKNOWN': 0.0, 'HOLD': 0.0}`.
- `word_add_appears`: expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`; tags `['mention_suppression']`; state `{'ADD': 0.9, 'MUL': 0.0, 'UNKNOWN': 0.0, 'HOLD': 0.0}`.
- `word_multiply_appears`: expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`; tags `['mention_suppression']`; state `{'ADD': 0.0, 'MUL': 0.9, 'UNKNOWN': 0.0, 'HOLD': 0.0}`.
- `add_actually_multiply`: expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`; tags `['correction_refocus']`; state `{'ADD': 0.9, 'MUL': 0.9, 'UNKNOWN': 0.0, 'HOLD': 0.0}`.
- `mul_actually_add`: expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`; tags `['correction_refocus']`; state `{'ADD': 0.9, 'MUL': 0.9, 'UNKNOWN': 0.0, 'HOLD': 0.0}`.

### `signed_amplitude`

- `add_actually_multiply`: expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`; tags `['correction_refocus']`; state `{'ADD': 0.9, 'MUL': 0.9, 'UNKNOWN': 0.0, 'HOLD': 0.55}`.
- `mul_actually_add`: expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`; tags `['correction_refocus']`; state `{'ADD': 0.9, 'MUL': 0.9, 'UNKNOWN': 0.0, 'HOLD': 0.55}`.

## Interpretation

Positive evidence overfires or over-holds because it treats cue presence as action authority.
Signed amplitude can represent simple cancellation but does not fully model correction/refocus.
The complex-phase arm passes this toy smoke because it separates destructive interference, hold pressure, and correction reset.

Safe claim if positive:

```text
phase-like interference is a useful candidate primitive for Pilot/Prismion command-state modeling.
```

## Claim Boundary

Toy command domain only. No consciousness claim, no general NLU claim, no quantum physics claim, and no production VRAXION/INSTNCT claim.
