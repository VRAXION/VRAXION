# Phase D11d H512 Readout / Activation Ignition

Date: 2026-05-01

## Verdict

`D11D_H512_EMBEDDING_TO_OUTPUT_REACH_BLOCKED`

D11d located the H512 failure more precisely. The input embeddings are valid, but the actual 32 VCBP embedding dimensions do not reach the output zone in the H512 baseline. This explains why relaxed search can move edges without producing useful output signal.

The next H512 fix should be an embedding-anchored highway/readout bootstrap, not a longer standard search.

## Evidence

`chain_diagnosis` on the H384 golden checkpoint:

- input vectors differ normally: average pairwise diff `24.8/32` dims (`78%`)
- input neurons `0..31` reaching output: `32/32`
- total output-dim changes from single input-neuron probes: `5967`
- full-pair output charge diversity: `56.0/237` dims (`23.6%`)
- alive output examples: roughly `131..141/237`

`chain_diagnosis` on the H512 D11a timing probe:

- input vectors still differ normally: average pairwise diff `24.8/32` dims (`78%`)
- input neurons `0..31` reaching output: `0/32`
- total output-dim changes from single input-neuron probes: `0`
- full-pair output charge diversity: `0.0/316` dims (`0.0%`)
- alive output: `0/316`

`chain_diagnosis` on the H512 D11c scatter/ties checkpoint:

- input neurons `0..31` reaching output: `32/32`, but total impact only `64` output-dim changes
- full-pair output charge diversity: `0.0/316` dims (`0.0%`)
- alive output: `2/316`, identical for all tested inputs

This means H512 can be forced to produce tiny output activity, but it is not input-dependent.

## Code Change

Added `--chain-count N` to `evolve_mutual_inhibition` for controlled H512 initialization tests.

Reason: `InitConfig::phi` disables chain highways at `H >= 512`. The override lets D11+ runs test whether chain highways are the missing ignition component without changing global defaults.

Smoke result:

- `H512_CHAIN50_STRICT`: still constant output, `0%` accept
- `H512_CHAIN50_TIES`: graph moves, still constant output

Conclusion: chain highways alone are not enough because they are not anchored to the 32 real VCBP embedding input dimensions.

## Interpretation

The H512 problem is not:

```text
not enough search time
```

It is:

```text
the 32 semantic input dimensions are not reliably wired into the output/readout zone
```

At H384, the random geometry plus chain/defaults happen to connect those 32 dimensions strongly enough. At H512, the larger space dilutes that path. Search then optimizes a network whose output is zero or nearly constant, so the evaluator has little useful gradient-like signal.

## Next Step

`D11e H512 embedding-anchored highway bootstrap`:

- add an explicit init option that seeds paths from each VCBP embedding dimension `0..31` into overlap/output zones,
- keep it behind a CLI flag, not a global default,
- smoke gate requires:
  - input neurons `0..31` reaching output: `32/32`
  - output charge diversity > `0`
  - unique predictions > `1`
  - no long search until those pass

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 release-candidate research checkpoint
    DONE

[2] H512 naive scaling
    DONE: baseline flat

[3] H512 acceptance/search calibration
    DONE: graph can move, output remains constant

[4] H512 readout/activation diagnosis
    DONE: embedding-to-output reach blocked
        |
        v
[5] H512 embedding-anchored highway bootstrap
    NEXT
        |
        |-- if output becomes input-dependent
        |      H512 state-anchored scout
        |
        '-- if still constant
               H512 architecture/init redesign
```

