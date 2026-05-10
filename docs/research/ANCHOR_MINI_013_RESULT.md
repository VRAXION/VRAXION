# ANCHOR-MINI-013 Result: Candidate-Scoped Parser A/B

## Verdict

```text
ANCHOR_MINI_013_PARTIAL_BUDGET
```

The fast diagnostic produced a clear negative signal, but it is still a partial
run, not a full sweep. It says the tested query-scoped CNN parser did not fix
candidate-local binding under held-out templates.

## Runs

Static checks:

```bash
python -m py_compile tools/anchorweave/run_anchor_mini013_candidate_scoped_parser.py
git diff --check
```

Smoke:

```bash
python tools/anchorweave/run_anchor_mini013_candidate_scoped_parser.py ^
  --out target/anchorweave/anchor_mini013_candidate_scoped_parser/smoke_v2 ^
  --seeds 2026 ^
  --models CHAR_CNN_QUERY_SCOPED ^
  --arms GLOBAL_PLAN_FIRST,QUERY_SCOPED_PLAN_FIRST,SHUFFLED_QUERY_SCOPED ^
  --template-counts 64 ^
  --train-examples 512 ^
  --eval-examples 512 ^
  --epochs 10 ^
  --jobs 2
```

Fast diagnostic:

```bash
python tools/anchorweave/run_anchor_mini013_candidate_scoped_parser.py ^
  --out target/anchorweave/anchor_mini013_candidate_scoped_parser/fast_2026_05_10 ^
  --seeds 2026-2030 ^
  --models CHAR_CNN_QUERY_SCOPED ^
  --arms ANSWER_ONLY_DIRECT,GLOBAL_PLAN_FIRST,SCALE_ONLY_GLOBAL,QUERY_SCOPED_PLAN_FIRST,SHUFFLED_QUERY_SCOPED ^
  --template-counts 64,128,256 ^
  --train-examples 4096 ^
  --eval-examples 2048 ^
  --epochs 60 ^
  --jobs 8 ^
  --budget-minutes 45
```

The fast diagnostic completed 32/75 jobs. Completed jobs cover all template
counts for seed `2026`, plus partial coverage of seeds `2027` and `2028`.

## Key Numbers

Grouped fast diagnostic means:

| arm | templates | jobs | eval acc | trap rate | goal acc | candidate effect acc | candidate policy acc | plan exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ANSWER_ONLY_DIRECT` | 64 | 3 | 0.243 | 0.257 | 0.256 | 0.249 | 0.450 | 0.000 |
| `ANSWER_ONLY_DIRECT` | 128 | 2 | 0.231 | 0.352 | 0.251 | 0.255 | 0.452 | 0.000 |
| `ANSWER_ONLY_DIRECT` | 256 | 2 | 0.217 | 0.372 | 0.238 | 0.246 | 0.419 | 0.000 |
| `GLOBAL_PLAN_FIRST` | 64 | 3 | 0.239 | 0.254 | 0.999 | 0.295 | 0.607 | 0.004 |
| `GLOBAL_PLAN_FIRST` | 128 | 2 | 0.220 | 0.332 | 1.000 | 0.310 | 0.597 | 0.001 |
| `GLOBAL_PLAN_FIRST` | 256 | 2 | 0.232 | 0.329 | 1.000 | 0.299 | 0.601 | 0.002 |
| `QUERY_SCOPED_PLAN_FIRST` | 64 | 2 | 0.247 | 0.228 | 1.000 | 0.251 | 0.543 | 0.000 |
| `QUERY_SCOPED_PLAN_FIRST` | 128 | 2 | 0.239 | 0.226 | 1.000 | 0.249 | 0.522 | 0.000 |
| `QUERY_SCOPED_PLAN_FIRST` | 256 | 2 | 0.248 | 0.219 | 1.000 | 0.252 | 0.461 | 0.000 |
| `SCALE_ONLY_GLOBAL` | 64 | 2 | 0.235 | 0.258 | 0.999 | 0.285 | 0.609 | 0.002 |
| `SCALE_ONLY_GLOBAL` | 128 | 2 | 0.223 | 0.327 | 0.998 | 0.301 | 0.595 | 0.001 |
| `SCALE_ONLY_GLOBAL` | 256 | 2 | 0.235 | 0.317 | 1.000 | 0.305 | 0.597 | 0.003 |
| `SHUFFLED_QUERY_SCOPED` | 64 | 2 | 0.254 | 0.224 | 0.000 | 0.245 | 0.516 | 0.000 |
| `SHUFFLED_QUERY_SCOPED` | 128 | 2 | 0.244 | 0.226 | 0.000 | 0.250 | 0.505 | 0.000 |
| `SHUFFLED_QUERY_SCOPED` | 256 | 2 | 0.246 | 0.225 | 0.000 | 0.255 | 0.464 | 0.000 |

## Interpretation

The result does not support the hypothesis that a simple `QUERY=A/B/C/D`
scope hint fixes candidate binding.

The model again learned the global goal under PLAN-first arms, but candidate
effect accuracy stayed near chance and exact plan rows stayed essentially zero.
Increasing global template count from 64 to 256 also did not help. Most
importantly, `SHUFFLED_QUERY_SCOPED` matched or slightly exceeded
`QUERY_SCOPED_PLAN_FIRST`, so the measured behavior is not a clean effect of
correct process labels.

This points to a deeper parser/binding failure:

```text
goal field parsing works
candidate-local effect binding does not work
query label alone is not enough
more templates alone is not enough
```

## Claim Boundary

This is not evidence against AnchorCell process supervision in general. It is a
negative diagnostic for the tested small CNN text carrier and the current raw
template serialization.

The result also does not prove natural-language grounding. It only narrows the
next engineering target: the carrier needs a stronger candidate-local parser,
not more real AnchorCells yet.

## Next Step

Do not scale to a 256-cell human dataset from this state.

The next useful step is a candidate-block/relative-position parser test where
the model is explicitly built to:

```text
locate candidate block
read that candidate's effect field
compare it to the goal field
emit candidate-local policy bit
route answer through policy bits
```

That should be tested first as a controlled parser carrier before returning to
larger DeskCache-style AnchorCell data.
