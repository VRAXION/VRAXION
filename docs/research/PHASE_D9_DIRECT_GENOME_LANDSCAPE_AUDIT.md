# Phase D9.0b Direct Genome Landscape Audit

Verdict: **D9_DIRECT_LANDSCAPE_TYPE_SPLIT**

## Summary

- Rows analyzed: `600`
- Base checkpoints: `1`
- Radii: `[1, 4, 16]`
- Probe mode: `medium`
- Eval length: `100`
- Samples per type/radius/base: `50`
- Overall median delta: `-0.010000`
- Overall cliff rate: `0.533`
- Overall positive rate: `0.037`
- Psi available: `False`

## Per-Type Classification

| mutation_type | classification | n | low_radius_cliff_rate | low_radius_positive_rate | best_of_9_low_radius | rho_radius_behavior |
| --- | --- | --- | --- | --- | --- | --- |
| channel | cliffy | 150 | 0.42 | 0.04 | 0.000833333 | 0.618697 |
| edge | rugged | 150 | 0.13 | 0.04 | 0.00416667 | 0.606878 |
| polarity | cliffy | 150 | 0.82 | 0.02 | -0.00166667 | 0.746352 |
| threshold | rugged | 150 | 0.3 | 0.03 | 0.00166667 | 0.603022 |

## Visual Artifacts

- `output\phase_d9_direct_genome_landscape_20260428\analysis\direct_landscape_atlas.html`
- `output\phase_d9_direct_genome_landscape_20260428\analysis\sphere_landscape.html`
- `output\phase_d9_direct_genome_landscape_20260428\analysis\sphere_tiles.csv`
- `output\phase_d9_direct_genome_landscape_20260428\analysis\local_zone_heatmap.png`
- `output\phase_d9_direct_genome_landscape_20260428\analysis\radius_score_delta_heatmap.png`
- `output\phase_d9_direct_genome_landscape_20260428\analysis\cliff_rate_by_radius.png`
- `output\phase_d9_direct_genome_landscape_20260428\analysis\per_type_radius_profiles.png`

## Caveats

- D9.0b freezes projection and mutates only the persisted core genome.
- A type-split verdict should not be collapsed into a global failure.
- Short eval-length smoke/medium runs are diagnostic; use the full command for final evidence.
- Random archive-pair baseline is marked unavailable unless supplied by a later extended run.
