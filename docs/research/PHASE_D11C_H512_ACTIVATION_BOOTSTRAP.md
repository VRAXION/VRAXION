# Phase D11c H512 Activation Bootstrap

Date: 2026-05-01

## Verdict

`D11C_H512_OUTPUT_BOOTSTRAP_BLOCKED`

D11c tested whether the H512 failure from D11a/D11b was simply an acceptance/search problem. It was not. Relaxed acceptance and bootstrap variants can move edges, but the output remains constant and inactive.

The next H512 work should not be a longer standard search. It should redesign the H512 activation/readout bootstrap so output neurons receive non-zero, input-dependent charge.

## What Ran

All runs used the same corpus/table as the H384 D7 baseline:

- corpus: `instnct-core/tests/fixtures/alice_corpus.txt`
- packed table: `output/block_c_bytepair_champion/packed.bin`
- seed: `2042`
- H: `512`

| arm | steps | key change | accept | accuracy | unique preds | alive output | result |
|---|---:|---|---:|---:|---:|---:|---|
| `H512_TICKS8_STRICT` | 1000 | ticks 8 | 0.0% | 0.0% | 1/4 | 0/158 | constant |
| `H512_TICKS10_STRICT` | 1000 | ticks 10 | 0.1% | 0.0% | 1/4 | 0/158 | constant |
| `H512_SCATTER_TICKS8_TIES` | 1000 | input scatter + ties | 100.0% | 0.0% | 1/4 | 0/158 | constant |
| `H512_SCATTER_TICKS8_J16_EPS` | 500 | input scatter + jackpot 16 + epsilon | 100.0% | 0.0% | 1/4 | 0/158 | constant |

## Interpretation

D11b already showed that relaxed acceptance can move the graph at H512. D11c confirms that graph movement is not enough: even with more ticks, input scatter, higher jackpot, and permissive acceptance, the output zone remains inactive.

The failure mode is therefore closer to:

```text
H512 output/readout ignition failure
```

not:

```text
not enough search steps
```

This means a long H512 ladder would currently waste compute. The evaluator would keep seeing a constant-output network, so mutation ranking would have little useful signal.

## Next Step

`D11d H512 readout/activation ignition`:

- add a direct activation sanity tool or mode that measures per-zone charge before search,
- compare H384 vs H512 initial charge distributions,
- tune H512 init thresholds / output-zone threshold distribution / propagation bootstrap,
- require `alive output > 0`, `unique predictions > 1`, and non-zero charge diff before any H512 search.

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 release-candidate research checkpoint
    DONE

[2] artifact/state-shuffle controls
    DONE: 16k / 30 fresh seed confirm passed

[3] H512 naive D7 scaling
    DONE: D11A_H512_BASELINE_NOT_READY

[4] H512 acceptance calibration
    DONE: graph moves, output still dead

[5] H512 activation bootstrap
    DONE: D11C_H512_OUTPUT_BOOTSTRAP_BLOCKED
        |
        v
[6] H512 readout/activation ignition
    NEXT: diagnose charge flow and threshold/readout scaling
        |
        |-- if output becomes live
        |      H512 state-anchored scout
        |
        '-- if output stays dead
               stay at H384; redesign H512 architecture/init
```

