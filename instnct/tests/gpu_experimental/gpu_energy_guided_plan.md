# GPU Energy-Guided Mutation Plan

## Current GPU finding

- Existing-edge scoring A/B is implemented in `gpu_energy_edge_score_ab.py`.
- Canonical reference log lives outside the repo in the original experiment log directory.
- `V64_N192`, `energy_pct=0.60`, `16k`, seeds `42,77,123`:
  - `product(src,dst)`: `76.0%`, `64.95 aps`
  - `max(src,dst)`: `77.6%`, `83.34 aps`

## What the result means

- The scoring formula itself is **not** the main cost.
- The expensive part is the weighted selection path:
  - `torch.nonzero(mask != 0)`
  - edge-weight vector materialization
  - `torch.multinomial(...)`
  - `.item()` / Python-side indexing
- On the current GPU reference path, `max` is the better existing-edge score than `product`.

## GPU-specific next question

Can a thresholded **hot-list shortlist** preserve the energy-guided quality signal while cutting the weighted-selection overhead?

The GPU-oriented hypothesis:

- `weighted_max` may be better than `product`, but still wastes time building weights every pick.
- `hot_threshold` should be cheaper if:
  - build `alive`
  - filter to edges where both ends exceed an energy threshold
  - do plain `random.choice` over the hot shortlist
- This keeps exploit/explore mixed by using `energy_pct` only as a gate:
  - with probability `energy_pct`, use the guided shortlist
  - otherwise use a plain random alive-edge pick

## Planned GPU experiment

Script:

- `gpu_energy_hotlist_ab.py`

Modes:

- `random`
- `weighted_max`
- `hot_threshold`

Config ladder:

1. `V64_N192`, `10 seeds`, `16k attempts`
2. `V128_N384`, `3 seeds`, `16k attempts`

Shared rules:

- `energy = sum_t sum_batch abs(act_t[:, n])`
- `add` stays fixed:
  - source = energy-guided
  - destination = random
- Only existing-edge ops vary by mode:
  - `flip`
  - `remove`
  - `rewire`

## Metrics to log

- `best_acc`
- `best_score`
- `attempts_per_sec`
- `mean_effective_changes_per_attempt`
- `selection_ms_per_attempt`
- `guided_pick_rate`
- `hot_hit_rate`
- `hot_fallback_rate`
- `mean_alive_edges_seen`
- `mean_hot_edges_seen`

## Acceptance rule

`hot_threshold` only advances if:

1. `mean(best_acc)` is not worse than `weighted_max` by more than `0.5pp`
2. `mean(best_score)` is not worse by more than `0.002`
3. `attempts_per_sec` is at least `1.15x`

If quality fails, keep `weighted_max`.
If quality passes but speed win is tiny, keep `weighted_max` and stop.

## Open design choices

- Threshold shape:
  - absolute threshold (`energy > tau`)
  - or percentile threshold (`top p%`)
- Refresh cadence:
  - every existing-edge pick
  - every accept
  - every checkpoint
- Ranked-list variant:
  - `flip/rewire` from the top slice of an energy-ranked alive-edge list
  - `remove` from the bottom slice
  - only worth trying after the simpler `hot_threshold` reference, because this
    adds a sort step and a more opinionated exploit/sculpt split

Start with the simplest reference:

- absolute threshold
- rebuild every guided pick

If that shows promise, only then optimize the cache cadence.
