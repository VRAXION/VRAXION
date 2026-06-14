# E80 Dataset-Backed Pocket Capability Scoring And Promotion Evidence

```text
decision = e80_dataset_backed_scoring_promotion_evidence_ready
checker_failure_count = 0
seed_count = 8
workers = 8
```

## Scope

E80 is the first dataset-backed scoring bridge after the E79 curriculum
readiness gate. It uses the local gitignored seed pack:

```text
data/high_quality_seed_v0/
```

Included source data:

```text
openai/gsm8k main
  train = 7473 rows
  test  = 1319 rows
  license = MIT

HuggingFaceFW/fineweb-edu sample-10BT
  sample = 2000 rows
  license = ODC-BY
```

This is not a claim that the runtime solves GSM8K or open-domain text. It tests
whether real dataset rows can produce guarded, scoreable Pocket capability
evidence with train / validation / adversarial coverage and bad-control
rejection.

## Run

```text
python scripts/probes/run_e80_dataset_backed_pocket_capability_scoring.py \
  --out target/pilot_wave/e80_dataset_backed_pocket_capability_scoring \
  --seeds 8001,8002,8003,8004,8005,8006,8007,8008 \
  --workers 8 \
  --heartbeat-seconds 5

python scripts/probes/run_e80_dataset_backed_pocket_capability_scoring_check.py \
  --out target/pilot_wave/e80_dataset_backed_pocket_capability_scoring \
  --write-summary
```

## Results

```text
promoted_candidate_count = 3
bad_promotion_count = 0
all_promoted_safe = true
```

| system | rows | success | false_commit | trace | promoted |
|---|---:|---:|---:|---:|---|
| gsm8k_answer_marker_adapter | 70336 | 1.000000 | 0.000000 | 1.000000 | true |
| gsm8k_rationale_calc_marker_adapter | 70336 | 0.962480 | 0.000000 | 0.962480 | false |
| fineweb_text_mode_selector | 16000 | 1.000000 | 0.000000 | 1.000000 | true |
| fineweb_byte_frame_boundary_adapter | 16000 | 1.000000 | 0.000000 | 1.000000 | true |
| bad_answer_first_control | 70336 | 0.766208 | 0.233792 | 0.000000 | false |
| bad_text_always_commit_control | 16000 | 0.260250 | 0.739750 | 0.000000 | false |

## Interpretation

The dataset-backed bridge is ready for the next curriculum step:

```text
raw dataset rows
-> deterministic lesson/capability adapter
-> train / validation / adversarial scoring
-> bad-control rejection
-> promotion evidence
```

The non-promoted GSM8K rationale calculation adapter is useful signal. It shows
that answer-marker extraction is stable, but calculation-trace parsing is not
yet robust enough for promotion. That should become a focused next mutation /
adapter improvement target rather than being forced into the stable set.

## Boundary

E80 does not claim:

```text
GSM8K solving
open-domain assistant behavior
Gemma-level capability
trained model weights
production readiness
```

It confirms only that a small high-quality local dataset pack can drive
dataset-backed capability scoring and safe promotion/rejection mechanics.
