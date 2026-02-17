# VRAXION

[![RESEARCH](docs/assets/badges/v2/research.svg)](https://github.com/VRAXION/VRAXION/wiki/Hypotheses) [![NONCOMMERCIAL](docs/assets/badges/v2/noncommercial.svg)](LICENSE) [![DOI 10.5281/zenodo.18332532](docs/assets/badges/v2/doi_10_5281_zenodo_18332532.svg)](https://doi.org/10.5281/zenodo.18332532) [![WIKI](docs/assets/badges/v2/wiki.svg)](https://github.com/VRAXION/VRAXION/wiki) [![CANONICAL](docs/assets/badges/v2/canonical.svg)](https://github.com/VRAXION/VRAXION/wiki/Governance)

VRAXION is a research codebase centered on **repeatable internal mechanisms** (loops/recurrence) and **instrumented evaluation** (so performance claims come with artifacts, not vibes).

## Diamond Code

The active codebase lives in `Diamond Code/`. Key files:

| File | Purpose |
|------|---------|
| `swarm_model.py` | SwarmByteRingModel (424M params, hash LCX, C19 activation, bottleneck projection) |
| `test_swarm_config.py` | Training loop (Goldilocks Ant v4 config, progressive schedule, dreaming phase) |
| `run_goldilocks.bat` | Launch script |
| `influx_writer.py` | InfluxDB telemetry (Grafana dashboards) |
| `live_controls.py` | Live-tunable training controls via controls.json |
| `byte_data.py` | Data loading and metrics |
| `traindat_loader.py` | Dataset interface (Gray code encoding) |

## Status

Research preview (v3.2.001). Current status lives in the [Roadmap](https://github.com/VRAXION/VRAXION/wiki/Chapter-11---Roadmap) and [Releases](https://github.com/VRAXION/VRAXION/releases).

## Where to look

- **Pages (landing):** https://vraxion.github.io/VRAXION/
- **Wiki (deep dives):** https://github.com/VRAXION/VRAXION/wiki
- **Roadmap (public):** https://github.com/orgs/VRAXION/projects/4
- **Releases (public proof):** https://github.com/VRAXION/VRAXION/releases

## Quickstart

Compile check (safe, no GPU):

```powershell
python -m compileall "Diamond Code"
```

Launch training (requires RTX 4070 Ti SUPER or equivalent):

```powershell
cd "Diamond Code"
run_goldilocks.bat
```

## Versioning (MAJOR.MINOR.BUILD)

VRAXION uses a simple cadence tracker stored in `VERSION.json`:

- `BUILD` increments on every merged "ticket completion" PR (fast/beta cadence).
- `MINOR` increments only for curated public updates (BUILD unchanged).
- `MAJOR` increments only for lifetime milestones (MINOR resets to 0; BUILD unchanged).

## Naming conventions

- Runtime env vars use the `VRX_` prefix.
- Legacy naming (`prime_c19`, `tournament_phase6`, `TP6_*`) is intentionally removed from the active code surface.
