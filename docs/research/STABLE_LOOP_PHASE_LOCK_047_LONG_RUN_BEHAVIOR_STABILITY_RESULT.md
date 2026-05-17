# STABLE_LOOP_PHASE_LOCK_047_LONG_RUN_BEHAVIOR_STABILITY Result

Status: positive bounded smoke.

047 tests whether the 046 concrete inference scaleout behavior remains stable
over checkpoint time. Production defaults remain disabled.

## Run

```powershell
cargo run -p instnct-core --example phase_lane_long_run_behavior_stability --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_047_long_run_behavior_stability/smoke ^
  --seeds 2026,2027,2028 ^
  --train-examples 8192 ^
  --heldout-examples 2048 ^
  --ood-examples 2048 ^
  --heartbeat-sec 30
```

The run wrote 114 checkpoint-arm metric rows: 6 checkpoints times the
required arms plus extra static shortcut controls. It completed quickly, but
still wrote incremental progress per checkpoint/arm block.

Checkpoints:

```text
checkpoint_000
checkpoint_010
checkpoint_025
checkpoint_050
checkpoint_100
checkpoint_200
```

## Key Metrics

Main positive arm, every measured checkpoint:

```text
ROUTE_GRAMMAR_LONG_RUN_TRAIN_AND_INFER:
  heldout_exact_accuracy = 1.000
  ood_exact_accuracy = 1.000
  family_min_accuracy = 1.000
  unique_output_count = 36 / 36
  top_output_rate = 0.065
  majority_output_rate = 0.061
  output_entropy = 4.806
  collapse_detected = false
  max_behavior_drift_score = 0.020
  max_output_distribution_drift = 0.010
```

Rollback-gated arm:

```text
ROUTE_GRAMMAR_LONG_RUN_ROLLBACK_GATED:
  heldout_exact_accuracy = 1.000
  ood_exact_accuracy = 1.000
  family_min_accuracy = 1.000
  checkpoint_save_load_pass = true
  rollback_success = true
```

Important failing controls:

```text
CONCRETE_INFERENCE_046_REFERENCE:
  heldout_exact_accuracy = 0.156
  ood_exact_accuracy = 0.153
  family_min_accuracy = 0.000
  unique_output_count = 4
  top_output_rate = 0.906
  collapse_detected = true

NO_ROUTE_GRAMMAR_LONG_RUN_BASELINE:
  heldout_exact_accuracy = 0.064
  ood_exact_accuracy = 0.058
  family_min_accuracy = 0.000
  unique_output_count = 1
  top_output_rate = 1.000
  collapse_detected = true

RANDOM_PHASE_RULE_CONTROL:
  heldout_exact_accuracy = 0.750
  ood_exact_accuracy = 0.750
  family_min_accuracy = 0.000
```

The random phase rule control retains output entropy but fails the family-min
gate, which prevents a false positive from aggregate accuracy.

## Concrete Inference Samples

Samples were written at every checkpoint. Examples from the stable arm:

```text
checkpoint_000 route_answer:
  input: ROUTE S>B>C>D>T source_phase=2 gates=[B:+0,C:+2,D:+1,T:+1] answer_phase?
  expected: phase_2
  predicted: phase_2

checkpoint_000 context_carry:
  input: MEMORY: key=river value=black / QUERY: what color is river?
  expected: black
  predicted: black

checkpoint_100 symbolic_map:
  input: MAP y->green z->blue a->cyan / QUERY y
  expected: green
  predicted: green

checkpoint_200 non_route_control:
  input: CLASSIFY parity 164
  expected: even
  predicted: even
```

## Verdicts

```text
LONG_RUN_BEHAVIOR_STABILITY_POSITIVE
INPUT_CONDITIONING_STABLE_OVER_TIME
OUTPUT_ENTROPY_STABLE
STATIC_OUTPUT_COLLAPSE_DOES_NOT_RETURN
MAJORITY_SHORTCUT_DOES_NOT_RETURN
COPY_SHORTCUT_DOES_NOT_RETURN
OOD_RETENTION_STABLE
NON_ROUTE_REGRESSION_CLEAN
BEHAVIOR_DRIFT_ACCEPTABLE
CHECKPOINT_SAVE_LOAD_STABLE
PRODUCTION_API_NOT_READY
```

## Interpretation

The 046 input-conditioned behavior remains stable across the measured
checkpoint timeline in this bounded runner suite. The main route-grammar
train-and-infer arms keep full heldout/OOD/family-min accuracy, preserve all
36 expected output classes, and do not collapse back into static, space,
majority, or copy-output shortcuts.

The control arms confirm the gate is adversarial: the 046 reference collapses
under this long-run behavior test, no-route training collapses to a single
dominant output, and random phase rules fail family-min even when aggregate
accuracy looks nonzero.

## Boundary

This is long-run behavior stability evidence for the bounded concrete
train-to-inference suite. It is not production default training, public beta
promotion, full VRAXION, language grounding, consciousness, or a production API
readiness claim.
