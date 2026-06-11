# E24_UNSCAFFOLDED_ONLINE_RULESHIFT_DISCOVERY_VS_NEURAL_BASELINES Contract

## Purpose

E24 follows E23 and attacks its main caveat. E23 gave Flow/Pocket an explicit
state-update route on a trace-locked online ruleshift proxy. E24 removes direct
rule-change assignments from valid systems. A system must discover rule changes
from visible support/evidence rows.

Core question:

```text
Does Flow/Pocket retain its trace-locked online ruleshift advantage when the
rule shift must be inferred from support contradictions, false alarms, delayed
updates, and partial changes instead of explicit shift assignments?
```

## Systems

```text
flow_pocket_unsccaffolded_discovery_primary
flow_pocket_marker_shortcut_ablation
flow_pocket_stale_rule_retention_ablation
flow_pocket_answer_only_ablation
mlp_trace_locked_gradient_baseline
gru_trace_locked_gradient_baseline
tiny_transformer_trace_locked_gradient_baseline
tiny_transformer_curriculum_trace_locked
random_static_control
direct_rule_engine_invalid_control
```

The direct rule engine is an invalid control only. Valid systems must not use
hidden before/after rule maps, direct calculators, Python eval, SymPy, or oracle
answers.

## Task Families

```text
implicit_shift
partial_shift
false_alarm
delayed_shift
full_shift
adversarial_decoy
OOD token family
counterfactual variants
```

Each episode provides a fresh codebook and visible support/evidence rows. The
stream may include an ambiguous notice marker, but it never directly states
which binding changed or what the new binding is.

Primary metric:

```text
composition_success = answer_correct AND trace_exact
```

## Decision Labels

```text
e24_flow_pocket_unsccaffolded_discovery_confirmed
e24_neural_unsccaffolded_ruleshift_stronger
e24_answer_without_discovery_trace_failure
e24_no_clear_winner
e24_invalid_oracle_or_artifact_detected
```

## Boundary

E24 is a controlled symbolic/numeric proxy. It does not prove raw language
reasoning, production readiness, AGI, consciousness, or model-scale behavior.
