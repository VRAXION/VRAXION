# Phase D17 Loss-Landscape Compass

Date: 2026-05-02

## Summary

D17 added a bounded heatmap-compatible landscape scan around the D16b `top_07` context candidate. The scan keeps the original EQ-bar / smooth output-loss signal visible, but scores it together with context-margin controls so output-only improvements cannot be mistaken for usable context.

Verdict from the first 64-cell compass:

```text
D17_OUTPUT_ONLY_TRAP
```

## Run Shape

- start: `output/phase_d16b_context_climb_main_20260502/candidates/top_07.ckpt`
- safety reference: `output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt`
- mode: `loss-landscape-compass`
- mutation scope: `threshold,edge`
- radii: `1,2,4,8`
- samples per radius: `8`
- eval_len: `512`
- eval seeds: `974201..974204`
- output root: `output/phase_d17_loss_landscape_compass_20260502/main`

## Result

The scan produced the requested heatmap-ready artifacts:

- `landscape_samples.csv`
- `landscape_cells.csv`
- `landscape_heatmap_matrix.csv`
- `D17_LOSS_LANDSCAPE_COMPASS_REPORT.md`

Main results:

- ready cells: `0`
- usable compass cells: `0`
- output-only trap cells: `7`
- best safe-navigation cell: `edge / radius 4 / ray 0`
- best cell still failed artifact/context margin criteria

Interpretation: some mutations improve or preserve the raw output-loss surface, but the fake-context controls move with the real context signal. This means the local top_07 neighborhood is not a safe climb direction for release-candidate context.

## Next Decision

Do not continue blind local mutation around top_07. The next useful branch is one of:

- redesign the context objective so search directly optimizes real-vs-fake margin;
- inspect older solo/context checkpoints for stronger context-carry circuits;
- redesign readout/projection if fake context remains tied to real context.

High-H brute force remains blocked until context/eval quality improves.
