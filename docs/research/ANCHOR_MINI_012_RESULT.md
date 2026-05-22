# ANCHOR-MINI-012A Result: Neural Template-Scaling Parser Diagnostic

## Verdict

```text
ANCHOR_MINI_012_TASK_OR_FORMAT_PROBLEM
```

This is a quick diagnostic result, not a publication-grade sweep. It says the
current raw text serialization did not become held-out-template robust under the
tested small neural carriers.

## Runs

Static checks:

```bash
python -m py_compile tools/anchorweave/run_anchor_mini012_neural_template_scaling.py
git diff --check
```

Smoke:

```bash
python tools/anchorweave/run_anchor_mini012_neural_template_scaling.py ^
  --out target/anchorweave/anchor_mini012_neural_template_scaling/smoke ^
  --seeds 2026 ^
  --models CHAR_BOW_MLP,CHAR_CNN ^
  --arms ANSWER_ONLY_DIRECT,PLAN_FIRST,SHUFFLED_PLAN_FIRST ^
  --template-counts 1,8 ^
  --train-examples 512 ^
  --eval-examples 512 ^
  --epochs 10 ^
  --jobs 2
```

Partial fast run:

```bash
python tools/anchorweave/run_anchor_mini012_neural_template_scaling.py ^
  --out target/anchorweave/anchor_mini012_neural_template_scaling/fast_2026_05_10 ^
  --seeds 2026-2030 ^
  --models CHAR_BOW_MLP,CHAR_CNN,CHAR_GRU ^
  --arms ANSWER_ONLY_DIRECT,AUX_PLAN_DIRECT,PLAN_FIRST,SHUFFLED_PLAN_FIRST ^
  --template-counts 1,3,8,16,32,64 ^
  --train-examples 4096 ^
  --eval-examples 2048 ^
  --eval-template-count 32 ^
  --epochs 60 ^
  --jobs 8 ^
  --budget-minutes 45
```

The partial fast run completed 48/360 jobs before being stopped as too slow for
the intended quick diagnostic. Completed jobs cover `CHAR_BOW_MLP` and
`CHAR_CNN` for seed `2026`.

GRU micro-triage:

```bash
python tools/anchorweave/run_anchor_mini012_neural_template_scaling.py ^
  --out target/anchorweave/anchor_mini012_neural_template_scaling/gru_small_decision_v2 ^
  --seeds 2026 ^
  --models CHAR_GRU ^
  --arms ANSWER_ONLY_DIRECT,PLAN_FIRST,SHUFFLED_PLAN_FIRST ^
  --template-counts 1,64 ^
  --train-examples 512 ^
  --eval-examples 512 ^
  --eval-template-count 32 ^
  --epochs 30 ^
  --jobs 3 ^
  --budget-minutes 15
```

## Key Numbers

Partial fast run, `CHAR_CNN`, seed `2026`:

| arm | train templates | eval acc | trap rate | goal acc | effect acc | policy acc | plan exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PLAN_FIRST` | 1 | 0.264 | 0.216 | 0.959 | 0.234 | 0.742 | 0.000 |
| `PLAN_FIRST` | 3 | 0.258 | 0.238 | 0.999 | 0.307 | 0.718 | 0.006 |
| `PLAN_FIRST` | 8 | 0.271 | 0.215 | 0.999 | 0.267 | 0.694 | 0.004 |
| `PLAN_FIRST` | 16 | 0.267 | 0.216 | 1.000 | 0.311 | 0.676 | 0.000 |
| `PLAN_FIRST` | 32 | 0.234 | 0.228 | 1.000 | 0.263 | 0.656 | 0.000 |
| `PLAN_FIRST` | 64 | 0.240 | 0.273 | 1.000 | 0.276 | 0.646 | 0.003 |

GRU triage, seed `2026`:

| arm | train templates | eval acc | trap rate | policy acc | plan exact |
|---|---:|---:|---:|---:|---:|
| `ANSWER_ONLY_DIRECT` | 1 | 0.266 | 0.180 | 0.438 | 0.000 |
| `ANSWER_ONLY_DIRECT` | 64 | 0.217 | 0.289 | 0.466 | 0.000 |
| `PLAN_FIRST` | 1 | 0.242 | 0.240 | 0.663 | 0.000 |
| `PLAN_FIRST` | 64 | 0.223 | 0.287 | 0.715 | 0.000 |
| `SHUFFLED_PLAN_FIRST` | 1 | 0.264 | 0.203 | 0.680 | 0.000 |
| `SHUFFLED_PLAN_FIRST` | 64 | 0.250 | 0.217 | 0.741 | 0.000 |

## Interpretation

The quick neural parser did not rescue held-out template transfer.

The strongest diagnostic detail is that `CHAR_CNN` learned the goal field on
held-out templates almost perfectly, but did not learn candidate effect parsing
or full policy construction. That means the failure is not simply "the model
saw too few situation examples." It is failing at binding each candidate's
effect field to the candidate-local policy decision under unseen templates.

The result does not prove all neural parsers fail. It only says this compact
raw text serialization plus the current small carriers is not yet enough for a
fast held-out-template pass.

## Next Step

Do not scale to 256 real AnchorCells from this state.

The next useful test should add an explicit candidate-scoped parser carrier:

```text
find candidate block
read candidate-local effect field
compare effect to goal
emit candidate-local policy bit
answer from policy bits
```

This can be tested either as a sparse relative-scanner carrier or as a more
structured neural parser with candidate-local encoders.
