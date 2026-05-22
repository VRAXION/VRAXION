# STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_SCALE Result

Status: positive bounded smoke.

049 tests whether the 048 frozen-corpus behavior survives a larger adversarial
frozen eval corpus with exact, near-duplicate, and semantic leakage audits.
Production defaults remain disabled.

## Run

```powershell
cargo run -p instnct-core --example phase_lane_adversarial_frozen_eval_scale --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_049_adversarial_frozen_eval_scale/smoke ^
  --seeds 2026,2027,2028 ^
  --train-examples 8192 ^
  --heldout-examples 4096 ^
  --ood-examples 4096 ^
  --heartbeat-sec 30
```

The run wrote 20 arm rows and append-only progress after every arm. Wall-clock
runtime was 227 seconds after startup for the measured runner section.

## Corpus

```text
frozen_eval_row_count = 1024
frozen_eval_unique_ids = 1024
expected_output_class_count = 75
source = docs/research/STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_CORPUS.jsonl
```

Leakage audit:

```text
train_eval_id_overlap_count = 0
train_eval_input_overlap_count = 0
train_eval_near_duplicate_count = 0
train_eval_semantic_overlap_count = 0
max_train_eval_token_jaccard = 0.667
```

## Key Metrics

```text
ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER:
  heldout_exact_accuracy = 1.000
  ood_exact_accuracy = 1.000
  family_min_accuracy = 1.000
  template_holdout_accuracy = 1.000
  family_holdout_accuracy = 1.000
  hard_distractor_accuracy = 1.000
  long_ood_accuracy = 1.000
  unique_output_count = 75 / 75
  top_output_rate = 0.073
  majority_output_rate = 0.055
  output_entropy = 5.404
  collapse_detected = false

ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_ROLLBACK_GATED:
  heldout_exact_accuracy = 1.000
  ood_exact_accuracy = 1.000
  family_min_accuracy = 1.000
  rollback_success = true
  checkpoint_save_load_pass = true
```

Important failing controls:

```text
NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE:
  heldout_exact_accuracy = 0.061
  ood_exact_accuracy = 0.049
  family_min_accuracy = 0.000
  unique_output_count = 1
  top_output_rate = 1.000
  collapse_detected = true

FROZEN_EVAL_048_REFERENCE:
  heldout_exact_accuracy = 0.166
  ood_exact_accuracy = 0.156
  family_min_accuracy = 0.000
  top_output_rate = 0.894
  collapse_detected = true

ROUTE_GRAMMAR_SHUFFLED_LABELS:
  heldout_exact_accuracy = 0.000
  ood_exact_accuracy = 0.000
  family_min_accuracy = 0.000

RANDOM_PHASE_RULE_CONTROL:
  heldout_exact_accuracy = 0.750
  ood_exact_accuracy = 0.750
  family_min_accuracy = 0.000
  long_ood_accuracy = 0.000
```

The random phase control keeps aggregate behavior on non-route families but
fails family-min and long-OOD route gates, preventing a false positive.

## Verdicts

```text
ADVERSARIAL_FROZEN_EVAL_SCALE_POSITIVE
ADVERSARIAL_FROZEN_INPUT_CONDITIONING_PASSES
ADVERSARIAL_FROZEN_NO_TRAIN_LEAKAGE
NEAR_DUPLICATE_LEAKAGE_AUDIT_PASSES
SEMANTIC_OVERLAP_AUDIT_PASSES
TEMPLATE_HOLDOUT_PASSES
FAMILY_HOLDOUT_PASSES
HARD_DISTRACTOR_PASSES
LONG_OOD_PASSES
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

The 048 frozen-corpus result scales to a larger adversarial frozen corpus in
this bounded runner. The passing route-grammar arm remains input-conditioned
across 1024 frozen examples, covers all 75 expected output classes, and passes
hard distractor, longer OOD, template-holdout, and family-holdout gates.

The stronger leakage audit reports zero exact ID/input overlap, zero
near-duplicate rows under token-Jaccard thresholding, and zero semantic
fingerprint overlaps against generated training rows.

## Boundary

This is bounded adversarial frozen-eval scale evidence. It is not production
default training, public beta promotion, production API readiness, full
VRAXION, language grounding, or consciousness.
