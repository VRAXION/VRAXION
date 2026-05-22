# Grounded Modular Self-Controller Finding

> Status: controlled toy finding. This is an integration result, not a consciousness claim.

This page records the May 2026 integrated probe that combined three previously separate toy mechanisms:

```text
semantic event
-> inferred grounding mode
-> self-anchor
-> hard committed self-state
-> learned controller over frozen modules
```

The canonical numeric report is in the repository:

- `docs/research/INTEGRATED_GROUNDED_MODULAR_SELF_CONTROLLER_PROBE.md`
- script: `scripts/run_integrated_grounded_modular_self_controller_probe.py`

## Why this test was run

Earlier probes established three separate signals:

1. **Inferred grounding mode**: the system can infer whether an event belongs to `reality`, `tv`, `game`, `dream`, or `memory` from compositional cue features.
2. **Recursive self-anchor v2**: a hard committed self-state can be required for a later hard-counterfactual action decision.
3. **Modular skill controller**: frozen primitives can be composed by a learned controller without primitive drift, while a shared end-to-end model can overwrite primitive skill identity.

The integrated probe asked whether these pieces work as one pipeline.

Core question:

> Can a grounded, committed self-state drive a modular controller's later action/program choice while preserving frozen primitive skills?

## Mechanism stack

The probe separates five layers that were previously blurred together:

| Layer | Role |
|---|---|
| Semantic event | What happened? Example: `dog bit me` means an injury/threat event. |
| Grounding mode | What reality layer is the event in? Example: `reality`, `tv`, `game`, `dream`, `memory`. |
| Self-anchor | Does this event affect the system/self, another person, or only an observed story/game/memory? |
| Committed self-state | A hard one-hot state such as `safe`, `injured`, `alert`, `other_help`, `story_observe`. |
| Modular controller | A learned selector over frozen action/program modules. |

The second-step prompt is intentionally generic. It does not reveal the original event, grounding mode, patient, or next-state label. The later controller decision must use the committed state.

## Final-test results

Main run:

```text
seeds: 5
hidden: 64
controller_hidden: 32
epochs: 350
module_epochs: 700
learning_rate: 0.005
```

### Event and self-state controller

| Arm | Semantic | Mode | State | Action | Hard CF | Margin | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `integrated_recursive_controller` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.997` | `0.001` |
| `no_committed_state_baseline` | `1.000` | `1.000` | `1.000` | `0.444` | `0.444` | `0.997` | `0.001` |
| `static_without_commit_baseline` | `1.000` | `1.000` | `1.000` | `0.444` | `0.444` | `0.997` | `0.001` |
| `oracle_committed_state_baseline` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.997` | `0.001` |
| `shuffled_committed_state_control` | `1.000` | `1.000` | `1.000` | `0.111` | `0.111` | `0.997` | `0.001` |
| `no_grounding_control` | `1.000` | `0.556` | `0.778` | `0.778` | `0.778` | `0.399` | `0.200` |

Key gaps:

```text
recursive_gap_vs_no_commit: 0.556
recursive_gap_vs_static:    0.556
oracle_gap:                 0.000
shuffled_committed_state:   0.111 hard-CF accuracy
wrong_mode_drop:            0.994
```

### Modular skill preservation

| Arm | Primitive Before | Primitive After | Drift | Composition | Program Acc |
|---|---:|---:|---:|---:|---:|
| `integrated_recursive_controller` | `1.000` | `1.000` | `0.000` | `1.000` | `1.000` |
| `shared_end_to_end_no_freeze` | `1.000` | `0.417` | `0.583` | `0.862` | `null` |
| `frozen_learned_controller_reference` | `1.000` | `1.000` | `0.000` | `1.000` | `1.000` |

The integrated controller preserved frozen primitives exactly while solving composition. The shared end-to-end baseline learned useful composition but substantially damaged primitive skill identity.

## Interpretation

The safe interpretation is narrow but important:

> In a controlled toy setting, inferred grounding can update a hard committed self-state, and that committed state can later drive a learned controller over frozen skills/actions without primitive drift.

This supports three bounded mechanism claims:

1. **Semantic event != grounded action authority.**
   - `dog bit me` can remain semantically an injury/threat event across modes.
   - Real action authority should rise only in the correct grounding/self condition.

2. **Self-state can be a causal decision bottleneck.**
   - No-commit and static baselines stayed at `0.444` hard-counterfactual action accuracy.
   - Shuffling the committed state dropped hard-counterfactual accuracy to `0.111`.
   - The recursive controller and oracle both reached `1.000`.

3. **Controller composition can preserve primitive identity.**
   - Frozen modules had `0.000` drift.
   - The shared end-to-end baseline drifted by `0.583`.

## What this does not prove

This result does not prove:

- consciousness
- biology
- quantum behavior
- natural-language understanding
- full VRAXION behavior
- production validity
- open-ended program discovery

The modules and toy domains are deliberately small. The result is evidence for a mechanism shape, not evidence that a full self-aware system has been built.

## Current implication

The result upgrades the earlier frame/refraction vocabulary into a more precise stack:

```text
semantic event
-> grounding mode
-> self-anchor
-> committed self-state
-> controller/action authority
```

For future VRAXION work, this suggests that the "pilot" is not a single skill module. It is closer to a protected controller layer that decides which already-formed routes, actions, or skills receive authority, conditioned on grounding and committed internal state.

## Read next

- [Speculative Extension - Cognitive Emergence](Cognitive-Emergence-Speculative)
- [Constructed Computation](Constructed-Computation)
- [Local Constructability Framework](Local-Constructability-Framework)
- [INSTNCT Architecture](INSTNCT-Architecture)
