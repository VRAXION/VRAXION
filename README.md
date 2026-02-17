# VRAXION

[![RESEARCH](docs/assets/badges/v2/research.svg)](https://github.com/VRAXION/VRAXION/wiki/Hypotheses) [![NONCOMMERCIAL](docs/assets/badges/v2/noncommercial.svg)](LICENSE) [![DOI 10.5281/zenodo.18332532](docs/assets/badges/v2/doi_10_5281_zenodo_18332532.svg)](https://doi.org/10.5281/zenodo.18332532) [![WIKI](docs/assets/badges/v2/wiki.svg)](https://github.com/VRAXION/VRAXION/wiki) [![CANONICAL](docs/assets/badges/v2/canonical.svg)](https://github.com/VRAXION/VRAXION/wiki/Governance)

VRAXION is a research program pursuing **repeatable internal mechanisms** for machine reasoning. The core scaling idea is **resolution over reshuffling**: grow capacity locally without breaking what coordinates mean. Progress is accepted only with explicit objectives, hard fail gates, and reproducible artifacts.

## What makes this different

Most systems optimize for producing the right output by absorbing patterns. VRAXION bets on learning the **mechanism that generates logic** — looped refinement rather than one-pass generation. "Better" means the loop converges more reliably under pressure (noise, longer contexts, resets), and we can show it via artifacts.

Key principles:
- **Mechanism over memorization** — internal paths matter, not just outputs
- **Instrumented evaluation** — every run has an objective, fail gates, and an artifact bundle
- **Resolution over reshuffling** — grow capacity via local refinement at deterministic boundaries

## Current state

**v3.2.001** (theory stage). Architecture complete — all 10 bottleneck levers locked. Training paused for architecture validation.

- **Model:** SwarmByteRingModel (424M params, D=6180, depth=12, hash LCX memory, C19 activation, bottleneck projection)
- **Observability:** Grafana + InfluxDB (66 panels), live-tunable controls
- **Platform:** NVIDIA RTX 4070 Ti SUPER (16 GB VRAM), Windows 11

Current status lives in the [Roadmap](https://github.com/VRAXION/VRAXION/wiki/Chapter-11---Roadmap) and [Releases](https://github.com/VRAXION/VRAXION/releases). See [CHANGELOG.md](CHANGELOG.md) for version history.

## Diamond Code

The active codebase lives in `Diamond Code/`. Key files:

| File | Purpose |
|------|---------|
| `swarm_model.py` | SwarmByteRingModel (424M params, hash LCX, C19 activation, bottleneck projection) |
| `test_swarm_config.py` | Training loop (Goldilocks Ant v4 config, progressive schedule, dreaming phase) |
| `run_goldilocks.bat` | Launch script (D=6180 full model) |
| `run_d618.bat` | Launch script (D=618 edge model) |
| `influx_writer.py` | InfluxDB telemetry (Grafana dashboards) |
| `live_controls.py` | Live-tunable training controls via controls.json |
| `byte_data.py` | Data loading and metrics |
| `traindat_loader.py` | Dataset interface (Gray code encoding) |

## Documentation

| Resource | What you'll find |
|----------|-----------------|
| [Wiki](https://github.com/VRAXION/VRAXION/wiki) | Full documentation — architecture, engineering, governance, evidence |
| [Engineering](https://github.com/VRAXION/VRAXION/wiki/Engineering) | Capacity guardrails, evaluation doctrine, failure taxonomy, telemetry |
| [Architecture](https://github.com/VRAXION/VRAXION/wiki/Diamond-Code-v3-Architecture) | Diamond Code v3 system specification |
| [Governance](https://github.com/VRAXION/VRAXION/wiki/Governance) | Wiki policy, versioning, provenance, contribution workflow |
| [Roadmap](https://github.com/orgs/VRAXION/projects/4) | Public project board |
| [Releases](https://github.com/VRAXION/VRAXION/releases) | Version history with detailed notes |
| [Pages](https://vraxion.github.io/VRAXION/) | Landing page |
| [Discussions](https://github.com/VRAXION/VRAXION/discussions) | Questions and community |

## Quickstart

**Code review and compilation** (no GPU required):

```powershell
python -m compileall "Diamond Code"
```

**Launch training** (requires NVIDIA GPU with 16+ GB VRAM):

```powershell
cd "Diamond Code"
run_goldilocks.bat
```

## Versioning (MAJOR.MINOR.BUILD)

VRAXION uses a cadence tracker stored in `VERSION.json`:

- `BUILD` increments on every merged "ticket completion" PR (fast/beta cadence).
- `MINOR` increments only for curated public updates (BUILD unchanged).
- `MAJOR` increments only for lifetime milestones (MINOR resets to 0; BUILD unchanged).

## License

- **Noncommercial use:** [PolyForm Noncommercial 1.0.0](LICENSE)
- **Commercial licensing:** [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) (contact: kenessy.dani@gmail.com)
- **Citation:** [CITATION.cff](CITATION.cff)

## Naming conventions

- Runtime env vars use the `VRX_` prefix.
- Legacy naming (`prime_c19`, `tournament_phase6`, `TP6_*`) is intentionally removed from the active code surface.
