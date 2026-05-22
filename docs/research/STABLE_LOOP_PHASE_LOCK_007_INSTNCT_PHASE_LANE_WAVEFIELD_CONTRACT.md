# STABLE_LOOP_PHASE_LOCK_007_INSTNCT_PHASE_LANE_WAVEFIELD Contract

## Question

Can canonical `instnct-core` mutation-selection discover local phase-lane wavefield transport from final target probability only?

This is the full VRAXION/INSTNCT follow-up to:

```text
005: fixed wavefield/interference gives a smooth target probability signal
006: toy float-weight mutation does not acquire the local complex primitive
```

## Substrate

The runner uses `instnct-core::Network` directly.

Each grid cell owns four phase-lane neurons:

```text
0 deg
90 deg
180 deg
270 deg
```

Gate tokens are represented as local per-cell gate neurons. Source phase is injected at the source cell, and readout is only from the target cell phase lanes.

Allowed mechanisms:

```text
integer charge / threshold / channel / polarity
local graph edges
recurrent settling ticks
canonical jackpot mutation
```

Forbidden mechanisms:

```text
gate_sum
path_phase_total
true_path as model input
label as model input
direct_phase_oracle
phase_gate_compose
pred = source_phase + gate_sum
global pooling
flattening
nonlocal target readout
```

## Arms

```text
ORACLE_PHASE_LANE_WIRING
RANDOM_PHASE_LANE_NETWORK
PARTICLE_FRONTIER_004_BASELINE
INSTNCT_GROWER_STRICT_K9
INSTNCT_GROWER_TIES_K9
INSTNCT_GROWER_ZEROP_K9
NO_CHANNEL_MUTATION_ABLATION
NO_POLARITY_MUTATION_ABLATION
NO_LOOP_MUTATION_ABLATION
SEEDED_PHASE_LANE_MOTIF_GROWER
```

`NO_POLARITY_MUTATION_ABLATION` is retained as an audit row. The current canonical jackpot schedule does not expose polarity as a sampled public operator, so this arm verifies that no hidden polarity-only path is responsible.

## Fitness

The main utility is:

```text
correct_target_lane_probability_mean
```

Target probability is computed from target phase-lane support only:

```text
score[k] = final_charge[k] + 4 * max(final_activation[k], 0)
prob[k] = score[k] / sum(score)
```

If all target lane scores are zero, the readout returns the uniform distribution.

## Dataset And Audits

The runner separates:

```text
PublicCase:
  wall/free mask
  source location
  source phase lane
  target marker
  per-cell local gate bucket

PrivateCase:
  label
  true_path
  path_phase_total
  gate_sum
  family
  split
```

The network/evolution code receives only `PublicCase`. The evaluator reads only `PrivateCase.label`.

Required audits:

```text
forbidden_private_field_leak = 0
nonlocal_edge_count = 0 for valid accepted networks
direct_output_leak_rate near 0
gate_shuffle collapses performance
same_target_counterfactual_accuracy reported
candidate_log.jsonl complete
```

## Metrics

```text
phase_final_accuracy
correct_target_lane_probability_mean
correct_target_lane_probability_delta_vs_random
candidate_delta_nonzero_fraction
positive_delta_fraction
C_K_constructability
same_target_counterfactual_accuracy
gate_shuffle_collapse
destructive_interference_accuracy
constructive_interference_accuracy
operator_accept_rate_by_type
```

## Verdicts

```text
INSTNCT_PHASE_LANE_TASK_VALID
INSTNCT_MUTATION_RESCUES_PHASE_CREDIT
FIXED_PHASE_LANE_ONLY
SEEDED_PRIMITIVE_REQUIRED
CHANNEL_MUTATION_REQUIRED
POLARITY_MUTATION_REQUIRED
LOOPS_REQUIRED
C_K_TOO_LOW
DIRECT_SHORTCUT_CONTAMINATION
TASK_TOO_EASY
TASK_TOO_HARD
```

## Positive Gate

```text
ORACLE_PHASE_LANE_WIRING >= 0.95 accuracy
RANDOM_PHASE_LANE_NETWORK near chance
INSTNCT grower beats random/frontier by >= +0.05 accuracy
correct_target_lane_probability improves by >= +0.10
same_target_counterfactual_accuracy >= 0.85
gate shuffle collapses
no forbidden/private/locality leak
candidate deltas are nontrivially dense
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
candidate_log.jsonl
operator_summary.json
constructability_metrics.jsonl
counterfactual_metrics.jsonl
locality_audit.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

No black-box rule:

```text
progress.jsonl every <=30 sec
metrics.jsonl after each checkpoint
candidate_log.jsonl continuously
summary.json/report.md refreshed on heartbeat
```

## Claim Boundary

This probe can support or falsify canonical INSTNCT phase-lane constructability in this toy spatial phase-lock setting.

It does not prove consciousness, full VRAXION validity, language grounding, production sidepocket architecture, or physical quantum behavior.
