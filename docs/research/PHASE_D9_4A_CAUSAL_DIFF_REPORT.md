# Phase D9.4a Causal Diff

Verdict: `EDGE_THRESHOLD_COADAPTATION`

D9.4a asks why the beta.8 H=384 generalist checkpoint is better than the seed2042 baseline. This is an explanatory pass, not a new search pass.

## Structural Finding

- Baseline: `output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt`
- Target: `output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt`
- Edges: `9718 -> 10140` (`+457` added, `-35` removed, net `+422`)
- Threshold changes: `143`
- Channel changes: `0`
- Polarity changes: `0`
- Projection bytes equal: `true`

The basin is therefore not a readout/projection change. It is a core-network change: edge wiring plus firing-threshold timing.

## Mixer Readout

- Bidirectional pairs: `694 -> 715`
- Triangles: `5674 -> 6425`
- Sampled 4-cycles: `102440 -> 121138`

The generalist is a denser recurrent signal mixer, not a one-neuron trick.

## Smoke Causal Result

Smoke run: `output/phase_d9_4a_causal_diff_smoke_20260429`

- Target generalist: `FULL_GENERALIST`
- `ablate_added_edges`: collapses below baseline
- `ablate_all_edges`: collapses below baseline
- `ablate_thresholds`: collapses below baseline
- `graft_added_edges`: fails
- `graft_thresholds`: fails
- `graft_all_edges_thresholds`: recovers the target

Interpretation: neither the edge package alone nor the threshold package alone explains the win. The win appears when both are present together, so the current best explanation is co-adaptation.

## Caveat

This committed report records the smoke-level causal verdict. A full 30-seed, eval_len 4000/16000 confirmation should be run before treating individual micro-ablation ranks as stable.
