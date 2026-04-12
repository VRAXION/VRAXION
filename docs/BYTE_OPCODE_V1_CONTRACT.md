# Byte + Opcode v1 Contract

Canonical byte/opcode acceptance surface for the grower-based `v5.0.0 Public Beta`.

## Domain

- input: `1 byte data + 4 opcode one-hot`
- ops: `COPY`, `NOT`, `INC`, `DEC`
- output: `1 byte`
- exact domain size: `256 x 4 = 1024`

## Canon output path

The beta-path is:

```text
input -> 8 frozen grower bit-heads -> binary latent signature -> exact LUT translator -> output byte
```

Important constraints:

- the translator key is built only from the grower heads' hidden binary activations
- the translator does not read raw input
- the translator is not trainable in the canonical beta path
- export is valid only if the full exact domain has:
  - `translator_byte_acc = 100.0`
  - `distinct_keys = 1024`
  - `conflicting_keys = 0`

## Acceptance command

```powershell
python tools/run_byte_opcode_acceptance.py
```

This must:

- build the example
- train/export the frozen stack
- reload the exported stack
- replay the full `1024`-sample domain
- compare against the golden fixture

## Required evidence

The append-only bundle under `target/byte-opcode-acceptance/<timestamp>/` must contain:

- `run_cmd.txt`
- `env.json`
- `train.stdout.txt`
- `reload.stdout.txt`
- `train_metrics.json`
- `reload_metrics.json`
- `summary.md`
- `golden_check.json`
- `reload_check.json`
- `stack.json`

## Canon expectations

- direct bitbank remains a negative control, not the winner
- exact LUT translator is the current public-beta readout contract
- learned/compressed translators remain research-lane diagnostics until they beat the exact freeze without losing determinism
