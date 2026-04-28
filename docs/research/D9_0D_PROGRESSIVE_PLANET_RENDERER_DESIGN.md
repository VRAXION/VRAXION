# D9.0d Progressive Planet Renderer Design

## Summary

D9.0d turns the D9.0b static genome-sphere visualization into a progressive
planet renderer. The goal is a live mission-control UI for genome-space
sampling: an initially empty tessellated planet fills in as sampling jobs write
tile statistics. The renderer is a shadow atlas, not exact high-D geometry.
Tile statistics are exact for the samples assigned to each tile.

The key D9.0b lesson is mandatory: mutation type must be first-class. The
current H=256 medium diagnostic produced a TYPE_SPLIT result:

| type | low-radius cliff rate | classification |
|---|---:|---|
| edge | 0.13 | rugged/searchable |
| threshold | 0.30 | rugged/searchable |
| channel | 0.42 | cliffy |
| polarity | 0.82 | cliffy |

Aggregating these into one map hides the signal. D9.0d must support per-type
viewing and per-type queues.

## UI Requirements

The HTML renderer must be self-contained and use Canvas without external
dependencies.

Top-bar toggles:

- Metric: `Fitness mean`, `Best`, `STD`, `Cliff`, `Confidence`, `Scan-prio`,
  `Split-prio`.
- Type: `All`, `edge`, `threshold`, `channel`, `polarity`.
- Eval layer: `scout`, `confirmed`, `combined`.
- Resolution: `16x32`, `32x64`, `64x128`.

Tile visual states:

| state | meaning |
|---|---|
| `UNKNOWN` | no sample yet |
| `SCOUT` | low-confidence cheap samples exist |
| `DESERT` | stable bad / low variance |
| `CLIFFY` | high cliff rate |
| `NOISY` | high variance / mixed tile |
| `PROMISING` | positive mean or high best with manageable risk |
| `CONFIRMED_GOOD` | expensive samples confirm signal |
| `RETIRED` | no longer worth sampling |
| `SPLIT_CANDIDATE` | should subdivide into finer local tiles |

Hover panel per tile:

- `tile_id`, active type, lat/lon bin.
- `n_scout`, `n_confirmed`, `target_n`.
- `mean_delta`, `best_delta`, `std`, `cliff_rate`, `positive_rate`.
- `confidence`.
- current state and recommended action.
- per-type breakdown when type mode is `All`.

HUD:

- Coverage progress.
- Confident-tile progress.
- Promising-tile count.
- Retired-tile count.
- Split queue count.
- Top-20 acquisition queue with type tags.
- Stop criteria status.
- Acquisition-weight sliders.

Every screenshot must include the caveat:

> PCA shadow projection. Geometry approximate. Tile statistics exact. Do not
> read absolute screen positions as true high-D distances.

## Progressive Sampling State

The sampler writes state incrementally:

- `progressive_atlas_state.json`: current tile table, queue, aggregate progress.
- `progressive_atlas_history.jsonl`: append-only per-sample history.
- `d9_0d_progressive_planet.html`: polling renderer.

The HTML polls `progressive_atlas_state.json` every 5 seconds. If the sampler is
stopped and restarted, it resumes from the saved state instead of starting from
zero.

## Default Scan Setup

Initial scout:

- H: `256`
- base checkpoint:
  `output/phase_d7_operator_bandit_20260427/H_256/D7_BASELINE/seed_42/final.ckpt`
- resolution: `32x64`
- scout `eval_len`: `100`
- confirmed `eval_len`: `1000`
- scout samples per tile/type: `2`
- confirmed target per tile/type: `10`
- first pass expensive limit: top `5-10%` tiles.

This is intentionally a renderer-style loop:

1. Start with unknown planet.
2. Cheap scout pass fills broad coverage.
3. Classify tiles.
4. Queue promising / uncertain / noisy tiles.
5. Confirm queued tiles with expensive eval.
6. Retire stable bad tiles.
7. Split mixed tiles.

## Acquisition Score

Default acquisition score:

```text
 0.35 * z(best_delta_score)
+ 0.25 * z(uncertainty)
+ 0.20 * z(std_delta_score)
+ 0.10 * z(positive_rate)
- 0.20 * z(cliff_rate)
- 0.10 * z(confidence)
```

The default weights are engineering heuristics. The UI may expose sliders, and a
later calibration mode can fit weights from observed tile outcomes. D9.0d should
not claim the default weights are theoretically optimal.

## State Transitions

Initial deterministic transition rules:

- `UNKNOWN -> SCOUT` after at least one sample.
- `SCOUT -> DESERT` if `mean_delta <= -0.005`, `std < 0.005`, and `n >= 3`.
- `SCOUT -> CLIFFY` if `cliff_rate > 0.65` and `n >= 3`.
- `SCOUT -> NOISY` if `std > 0.02` and `n >= 5`.
- `SCOUT -> PROMISING` if `mean_delta >= 0`, `positive_rate > 0.10`, and
  `n >= 3`.
- `SCOUT -> UNCERTAIN` otherwise.
- `PROMISING -> CONFIRMED_GOOD` after expensive confirmation keeps `mean_delta >= 0`.
- `NOISY -> SPLIT_CANDIDATE` if high variance persists after `n >= 5`.
- `CLIFFY` and `DESERT` can become `RETIRED` after confidence is high.

## Stop Criteria

The renderer can mark a pass as finished when:

- coverage is at least `80%`,
- confident-tile fraction is at least `60%`,
- and fewer than `1%` of tiles changed state in the last 5 minutes.

Manual stop is always allowed. Stopping must not lose progress because state is
persisted.

## Implementation Order

1. Add the polling HTML shell with type toggle and progress HUD.
2. Add a Python state builder that converts existing D9.0b samples into
   `progressive_atlas_state.json`.
3. Add resumable scout sampler.
4. Add expensive confirm queue.
5. Add recursive split only after scout/confirm state is stable.

Do not implement recursive subdivision before type filtering and live state
persistence work.
