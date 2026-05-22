# STABLE_LOOP_ATTRACTOR_SWEEP_001 Contract

## Purpose

Test the stable-loop / attractor / reusable internal function hypothesis.

This is not a parser, factuality, or language probe. It asks whether a tied recurrent settling loop can learn a stable function that:

- improves with repeated settling,
- converges instead of oscillating or drifting,
- survives overrun,
- recovers from state noise,
- and retains learned function under continuation training.

The previous `HIGHWAY_SIDEQUEST_TOY_001` tested gated writeback topology. This probe tests explicit settling dynamics.

## Task

Use abstract symbolic streams only:

```text
A, B, C, D
anti_A, anti_B, anti_C, anti_D
reset
delay
mention_A, mention_B, mention_C, mention_D
quote_anti_A, quote_anti_B, quote_anti_C, quote_anti_D
actually_A, actually_B, actually_C, actually_D
instead_A, instead_B, instead_C, instead_D
create_X, remove_X, restore_X, query_count
noise
```

Training signal is final outcome only. Do not train on intermediate state labels.

Deterministic scope:

```text
anti_X creates block scope with TTL=2 token steps
delay consumes one TTL
reset clears all scopes
anti_X only blocks/cancels while TTL > 0
```

## Required Controls

`SUMMARY_DIRECT_HEAD` is mandatory. It receives the same input summary / h0 source as loop models and predicts without settling. If it matches loop models, report `SUMMARY_SOLVES_TASK`.

Settling modes:

```text
Autonomous: h_{t+1} = f(h_t)
Conditioned: h_{t+1} = f(h_t, input_summary)
```

Primary `STABLE_LOOP_POSITIVE` requires autonomous settling. If only conditioned settling works, report `CONDITIONED_LOOP_ONLY`.

Compute controls:

```text
DEEP_MLP_MATCHED_COMPUTE
GRU_EXTRA_NOOP_STEPS
```

If these match loop models, report `COMPUTE_BUDGET_CONFONDED`.

## Arms

```text
MLP_STATIC
DEEP_MLP_MATCHED_COMPUTE
SIMPLE_RNN
GRU
GRU_EXTRA_NOOP_STEPS
LSTM
SUMMARY_DIRECT_HEAD
MAIN_LOOP_MLP_AUTONOMOUS
MAIN_LOOP_MLP_CONDITIONED
MAIN_LOOP_GRU_AUTONOMOUS
MAIN_LOOP_GRU_CONDITIONED
HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS
HIGHWAY_FF_SIDEPOCKETS_CONDITIONED
HIGHWAY_RECURRENT_SIDEPOCKETS
HIGHWAY_SPARSE_SIDEPOCKETS
HIGHWAY_DENSE_SIDEPOCKETS
HIGHWAY_PRISMION_SIDEPOCKETS
VRAXION_LITE_LOOP
```

Default device is CPU. CUDA only runs when explicitly requested.

## Metrics

Report:

```text
final_answer_accuracy
heldout_composition_accuracy
length_generalization_accuracy
false_mutation_rate
false_cancellation_rate
false_refocus_rate
mention_noop_error_rate
scope_error_rate
settling_gain
final_state_delta
convergence_rate
overrun_stability
noise_recovery_accuracy
summary_direct_gap
matched_compute_gap
autonomous_vs_conditioned_gap
linear_probe_accuracy
MLP_probe_accuracy_separate
retention_after_new_training
catastrophic_interference_rate
stable_function_retention
parameter_count
epochs_to_threshold
```

Probe targets after freezing:

```text
active_symbol
blocked_symbol
current_focus
entity_count
mutation/no_mutation
scope_active
```

## Verdicts

```text
STABLE_LOOP_POSITIVE
SUMMARY_SOLVES_TASK
CONDITIONED_LOOP_ONLY
COMPUTE_BUDGET_CONFONDED
STABLE_FUNCTION_RETENTION_POSITIVE
CATASTROPHIC_INTERFERENCE_WARNING
HIGHWAY_TOPOLOGY_POSITIVE
SIDEPOCKET_SPECIALIZATION_POSITIVE
SPARSE_COORDINATION_POSITIVE
DENSE_MONOLITH_WARNING
PRISMION_UPDATE_POSITIVE
VRAXION_LITE_POSITIVE
STANDARD_RNN_SUFFICIENT
TASK_TOO_EASY
TASK_TOO_HARD
LOOP_UNSTABLE
```

## Run Stages

Smoke:

```powershell
python scripts/probes/run_stable_loop_attractor_sweep_probe.py ^
  --out target/pilot_wave/stable_loop_attractor_sweep_001/smoke ^
  --stage smoke ^
  --seeds 2026,2027 ^
  --train-examples 1024 ^
  --eval-examples 1024 ^
  --epochs 20 ^
  --jobs auto50 ^
  --device cpu ^
  --heartbeat-sec 15
```

Valid slice:

```powershell
python scripts/probes/run_stable_loop_attractor_sweep_probe.py ^
  --out target/pilot_wave/stable_loop_attractor_sweep_001/valid_slice ^
  --stage valid_slice ^
  --seeds 2026,2027,2028 ^
  --train-examples 4096 ^
  --eval-examples 4096 ^
  --epochs 80 ^
  --jobs auto50 ^
  --device cpu ^
  --heartbeat-sec 30
```

Full survivors:

```powershell
python scripts/probes/run_stable_loop_attractor_sweep_probe.py ^
  --out target/pilot_wave/stable_loop_attractor_sweep_001/full_survivors ^
  --stage full_survivors ^
  --from target/pilot_wave/stable_loop_attractor_sweep_001/valid_slice/survivor_configs.json ^
  --seeds 2026-2035 ^
  --train-examples 8192 ^
  --eval-examples 8192 ^
  --epochs 160 ^
  --jobs auto50 ^
  --device cpu ^
  --heartbeat-sec 30
```

## Run Hygiene

Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
survivor_configs.json
ablation_results.jsonl
convergence_curves.jsonl
probe_results.jsonl
continuation_results.jsonl
examples_sample.jsonl
contract_snapshot.md
job_progress/*.jsonl
```

Raw `target/` outputs are not committed.
