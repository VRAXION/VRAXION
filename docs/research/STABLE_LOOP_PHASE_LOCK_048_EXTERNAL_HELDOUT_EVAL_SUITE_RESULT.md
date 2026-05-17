# STABLE_LOOP_PHASE_LOCK_048_EXTERNAL_HELDOUT_EVAL_SUITE Result

Status: positive bounded smoke.

048 tests whether the 046/047 concrete input-conditioned inference behavior
survives a committed frozen heldout/OOD eval suite with no train leakage.
Production defaults remain disabled.

## Run

```powershell
cargo run -p instnct-core --example phase_lane_external_heldout_eval_suite --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_048_external_heldout_eval_suite/smoke ^
  --seeds 2026,2027,2028 ^
  --train-examples 8192 ^
  --heldout-examples 2048 ^
  --ood-examples 2048 ^
  --heartbeat-sec 30
```

The first smoke correctly failed with `TRAIN_LEAKAGE_DETECTED`: 8 frozen eval
inputs exactly overlapped generated train inputs. The frozen corpus was then
made parser-valid but syntactically external for those rows. The second smoke
passed with zero ID/input overlap.

## Key Metrics

```text
FROZEN_EVAL_ROUTE_GRAMMAR_TRAIN_AND_INFER:
  heldout_exact_accuracy = 1.000
  ood_exact_accuracy = 1.000
  family_min_accuracy = 1.000
  template_holdout_accuracy = 1.000
  family_holdout_accuracy = 1.000
  frozen_eval_row_count = 64
  frozen_eval_unique_ids = 64
  train_eval_id_overlap_count = 0
  train_eval_input_overlap_count = 0
  unique_output_count = 28 / 28
  top_output_rate = 0.078
  majority_output_rate = 0.063
  output_entropy = 4.584
  collapse_detected = false

FROZEN_EVAL_ROUTE_GRAMMAR_ROLLBACK_GATED:
  heldout_exact_accuracy = 1.000
  ood_exact_accuracy = 1.000
  family_min_accuracy = 1.000
  rollback_success = true
  checkpoint_save_load_pass = true
```

Important failing controls:

```text
NO_ROUTE_GRAMMAR_FROZEN_EVAL_BASELINE:
  heldout_exact_accuracy = 0.063
  ood_exact_accuracy = 0.063
  family_min_accuracy = 0.000
  unique_output_count = 1
  top_output_rate = 1.000
  collapse_detected = true

CONCRETE_INFERENCE_046_REFERENCE:
  heldout_exact_accuracy = 0.156
  ood_exact_accuracy = 0.156
  family_min_accuracy = 0.000
  top_output_rate = 0.906
  collapse_detected = true

ROUTE_GRAMMAR_SHUFFLED_LABELS:
  heldout_exact_accuracy = 0.000
  ood_exact_accuracy = 0.000
  family_min_accuracy = 0.000

RANDOM_PHASE_RULE_CONTROL:
  random phase control fails family-min despite nonzero aggregate behavior.
```

## Verdicts

```text
EXTERNAL_HELDOUT_EVAL_POSITIVE
FROZEN_EVAL_INPUT_CONDITIONING_PASSES
FROZEN_EVAL_NO_TRAIN_LEAKAGE
TEMPLATE_HOLDOUT_PASSES
FAMILY_HOLDOUT_PASSES
STATIC_OUTPUT_COLLAPSE_REJECTED
MAJORITY_LABEL_SHORTCUT_REJECTED
COPY_SHORTCUT_REJECTED
SHUFFLED_LABELS_FAIL
RANDOM_LABEL_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
NON_ROUTE_REGRESSION_CLEAN
PRODUCTION_API_NOT_READY
```

## Interpretation

The route-grammar train-and-infer path remains input-conditioned on the frozen
heldout/OOD corpus. The eval rows are committed data, loaded through
`include_str!`, and audited for train/eval ID and exact-input overlap.

This closes the immediate runner-specific shortcut concern for the bounded
frozen corpus: the previous 046 reference collapses, the no-route baseline
collapses, shuffled labels fail, and static/copy/majority controls do not pass.

## Boundary

This is frozen-heldout eval evidence for a bounded concrete inference suite. It
is not production default training, public beta promotion, production API
readiness, full VRAXION, language grounding, or consciousness.
