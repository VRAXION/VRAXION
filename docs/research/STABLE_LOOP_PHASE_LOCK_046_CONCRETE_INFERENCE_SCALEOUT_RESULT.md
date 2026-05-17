# STABLE_LOOP_PHASE_LOCK_046_CONCRETE_INFERENCE_SCALEOUT Result

Status: positive bounded scaleout behavioral gate.

046 scaled the 045 concrete train -> inference anti-collapse probe to a larger
and more diverse suite. It added longer route chains, multi-memory lookup,
two-step compositional maps, arithmetic transforms, more output classes, and
OOD template reorderings. Production defaults remain disabled.

## Run

```text
cargo run -p instnct-core --example phase_lane_concrete_inference_scaleout --release -- \
  --out target/pilot_wave/stable_loop_phase_lock_046_concrete_inference_scaleout/smoke \
  --seeds 2026,2027,2028 \
  --train-examples 8192 \
  --heldout-examples 2048 \
  --ood-examples 2048 \
  --heartbeat-sec 30
```

The run completed before the 30s wall-clock heartbeat interval, but it still
wrote append-only progress after every arm plus a final `done` row.

## Verdicts

```text
CONCRETE_INFERENCE_SCALEOUT_POSITIVE
SCALEOUT_INPUT_CONDITIONING_SURVIVES
LONG_SEQUENCE_GENERALIZATION_PASSES
MORE_OUTPUT_CLASSES_PASS
STATIC_OUTPUT_COLLAPSE_REJECTED
SPACE_OUTPUT_COLLAPSE_REJECTED
MAJORITY_LABEL_SHORTCUT_REJECTED
COPY_SHORTCUT_REJECTED
HELDOUT_GENERALIZATION_PASSES
OOD_GENERALIZATION_PASSES
NON_ROUTE_REGRESSION_CLEAN
SHUFFLED_LABELS_FAIL
ALWAYS_SPACE_CONTROL_FAILS
ALWAYS_EMPTY_CONTROL_FAILS
ALWAYS_MAJORITY_CONTROL_FAILS
PRODUCTION_API_NOT_READY
```

## Key Metrics

```text
ROUTE_GRAMMAR_SCALEOUT_TRAIN_AND_INFER:
  heldout_exact_accuracy = 1.000
  ood_exact_accuracy = 1.000
  family_min_accuracy = 1.000
  unique_output_count = 36 / 36
  top_output_rate = 0.065
  majority_output_rate = 0.061
  output_entropy = 4.806
  space_only_rate = 0.000
  empty_output_rate = 0.000
  non_route_regression_delta = 0.000
  checkpoint_save_load_pass = true

ROUTE_GRAMMAR_SCALEOUT_ROLLBACK_GATED:
  heldout_exact_accuracy = 1.000
  ood_exact_accuracy = 1.000
  family_min_accuracy = 1.000
  unique_output_count = 36 / 36
  top_output_rate = 0.065
  output_entropy = 4.806
  rollback_success = true
  checkpoint_save_load_pass = true

CONCRETE_INFERENCE_045_REFERENCE:
  heldout_exact_accuracy = 0.156
  ood_exact_accuracy = 0.153
  family_min_accuracy = 0.000
  unique_output_count = 4 / 36
  top_output_rate = 0.906
  majority_output_rate = 0.906
  output_entropy = 0.596
  collapse_detected = true

ALWAYS_SPACE_CONTROL:
  heldout_exact_accuracy = 0.000
  top_output_rate = 1.000
  space_only_rate = 1.000
  collapse_detected = true

ALWAYS_MAJORITY_CONTROL:
  heldout_exact_accuracy = 0.064
  top_output_rate = 1.000
  majority_output_rate = 1.000
  collapse_detected = true

COPY_LAST_TOKEN_CONTROL:
  heldout_exact_accuracy = 0.000
  family_min_accuracy = 0.000

RANDOM_PHASE_RULE_CONTROL:
  heldout_exact_accuracy = 0.750
  ood_exact_accuracy = 0.750
  family_min_accuracy = 0.000
```

## Per-Family Result

The passing scaleout arm reached `1.000` accuracy on all eight families:

```text
route_answer
long_route_answer
context_carry
multi_memory
symbolic_map
compositional_map
arithmetic_transform
non_route_control
```

The random phase rule control preserved non-route families but failed both
route families, leaving `family_min_accuracy = 0.000`. That is the intended
adversarial check: aggregate-looking behavior is not enough.

## Concrete Inference Samples

```text
long_route_answer:
  input: LONG_ROUTE S>C>D>E>F>G>H>I>T source_phase=2 gates=[C:+1,D:+2,E:+0,F:+1,G:+2,H:+1,I:+1,T:+0] answer_phase?
  expected_output: phase_2
  predicted_output: phase_2

multi_memory:
  input: MEMORY coin=green river=red shell=silver paper=white
         QUERY value key=coin
  expected_output: green
  predicted_output: green

compositional_map:
  input: COMPOSE s->red x->green y->blue red->star green->line blue->arc
         QUERY s
  expected_output: star
  predicted_output: star

arithmetic_transform:
  input: TRANSFORM start=43 mul=2 add=11 mod=11 answer_num?
  expected_output: num_9
  predicted_output: num_9
```

## Interpretation

The 045 bounded concrete inference result survived scaleout in this runner:
more families, longer inputs, reordered OOD templates, and a larger output
space did not trigger static output collapse. The output distribution stayed
spread across the expected 36 classes, with low top-output and majority rates.

The 045 reference arm failed under scaleout, so this result is not merely the
previous behavior copied forward. It needs the scaleout train-and-infer path.

## Claim Boundary

046 supports:

```text
input-conditioned concrete inference survives the tested bounded scaleout suite
static/space/majority/copy shortcuts fail under stronger controls
```

046 does not support:

```text
production default training
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
```
