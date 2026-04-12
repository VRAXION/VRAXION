# Grower Run Contract

This document freezes the current grower engine contract on `main`. It is the canonical source for how the public grower lane is supposed to run and what can change without breaking comparability.

## Canonical code path

- Mainline builder: [`instnct-core/examples/neuron_grower.rs`](../instnct-core/examples/neuron_grower.rs)
- Canonical regression harness: [`tools/run_grower_regression.py`](../tools/run_grower_regression.py)
- Golden snapshot: [`instnct-core/tests/fixtures/grower_regression_golden.json`](../instnct-core/tests/fixtures/grower_regression_golden.json)

## Frozen defaults

- Neuron form: bias-free threshold neuron, `dot >= threshold`
- Accept gate: non-strict ensemble improvement (`reject` only if `new_val < ens_val`)
- Parent search: scout is mandatory
  - single-signal score
  - connect-all backprop probe
  - pair-lift shortlist
- Proposal policy:
  - shortlist-guided candidate sets first
  - random fallback only after scout-derived sets
- Persistence:
  - default run directory is `target/neuron_grower/<task>`
  - authoritative resume file is `state.tsv`
  - checkpoints are append-only JSON snapshots under `checkpoints/`
- Determinism:
  - `data_seed` controls dataset sampling
  - `search_seed` controls proposal / restart randomness

## Canonical regression matrix

The B0 engine-freeze matrix is:

- `four_parity`
- `four_popcount_2`
- `is_digit_gt_4`
- `diagonal_xor`
- `full_parity_4`
- `digit_parity`

All canonical regression runs use:

- `data_seed=42`
- `search_seed=42`
- per-task `max_neurons` / `stall` as defined in `tools/run_grower_regression.py`
- append-only evidence bundle under `target/grower-regression/<timestamp>/`

## Required evidence bundle

Each canonical regression run must emit:

- `run_cmd.txt`
- `env.json`
- `metrics.json`
- `summary.md`
- `golden_check.json`

These files are the minimum evidence bundle. Extra stdout logs are fine, but they do not replace the bundle.

## What counts as a regression pass

A canonical grower regression pass is green when:

- all matrix tasks complete without panic
- the evidence bundle is written
- the current metrics match the golden snapshot
- the same command with the same seeds reproduces the same result

## Promotion gates after B0

B0 only freezes the engine contract. Public beta still requires:

- byte + opcode v1 benchmark
- structured output head freeze
- adversarial truth-table suite
- frozen export / reload path for the chosen output stack
