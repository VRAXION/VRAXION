# Diamond Code v3 — SwarmByteRingModel

**Research implementation** of a swarm-based neural memory architecture with hash-bucketed long-term consensus memory (LCX/LTCM).

## Architecture

**SwarmByteRingModel** — 424M parameters (D=6180, depth=12)

- **Ring Memory**: Circular buffer with soft pointer addressing and content-based jump routing
- **LCX (Long-term Consensus Matrix)**: 2000 hash-bucketed memory slots with SimHash retrieval
  - Single HD level, expandable via `resize_lcx()` during training
  - ~512 slots searched per query regardless of total size
- **Swarm**: Multi-being ensemble (currently 1 being, expandable)
- **Progressive Training**: 6 effort tiers from Alpha (pure brain) to Zeta (deep contemplation)
- **Golden Ratio Fractal Stack**: D=6180, key_dim=618, top_k=6, ring=62

### Progressive Training Tiers

| Tier | Think Ticks | LCX | Batch | Name |
|------|-------------|-----|-------|------|
| Alpha | 0 | OFF | 10 | Reflex |
| Beta | 1 | ON | 10 | Recall |
| Gamma | 2 | ON | 5 | Reason |
| Delta | 4 | ON | 3 | Depth |
| Epsilon | 8 | ON | 2 | Emergence |
| Zeta | 16 | ON | 1 | Zenith |

## Key Files

| File | Description |
|------|-------------|
| `swarm_model.py` | Core model (3000+ lines) |
| `test_swarm_config.py` | Training loop & evaluation |
| `influx_writer.py` | InfluxDB telemetry writer |
| `live_controls.py` | Live Grafana control interface |
| `run_goldilocks.bat` | Training launcher |
| `tools/control_panel.py` | HTTP control panel for Grafana |
| `tools/checkpoint_autopsy.py` | Checkpoint forensics |
| `tools/lcx_forensics.py` | LCX memory analysis |

## Observability Stack

- **Grafana** (`:3000`): Primary dashboard with LTCM visualization, routing metrics, training controls
- **InfluxDB** (`:8086`): Time-series metrics storage (bucket: `diamond`)
- **Control Panel** (`:7777`): HTTP bridge for bidirectional Grafana controls
- **Streamlit** (`:8501`): Legacy dashboard

### Dashboard Sections

1. **Top Bar** — Step counter, loss, accuracy, eval metrics
2. **Main Training** — Training overview chart + live controls form
3. **LTCM** — Memory utilization, write heat, slot dynamics, Score Margin routing quality
4. **Beings & Swarm** — Ensemble metrics (for multi-being experiments)
5. **Ant/Bit Intelligence** — Per-bit accuracy, jump gates, IQ metrics
6. **Controls** — Data mix, effort timeseries, control change log
7. **Data Tables** — Input/output/memory state tables (collapsed)
8. **Channel Analysis / Debug** — RGB channel norms, scratchpad (collapsed)

## Quick Start

```bash
# Generate training data
python generate_traindat_suite.py

# Launch training
run_goldilocks.bat

# Launch Grafana (if not running as service)
grafana/start_grafana.bat

# Dashboard at http://localhost:3000
```

## Environment

- GPU: NVIDIA RTX 4070 Ti SUPER (16 GB VRAM)
- Python 3.11, PyTorch with CUDA, bfloat16 mixed precision
- InfluxDB 2.x, Grafana 11.x

## Documentation

- **Architecture**: [Diamond Code v3 Architecture](https://github.com/VRAXION/VRAXION/wiki/Diamond-Code-v3-Architecture) (wiki)
- **Session Logs**: [Feb 14-16 Sprint](https://github.com/VRAXION/VRAXION/wiki/Session-Log-Feb-14-16-2026) (wiki)
- **Theory**: [Theory of Thought](https://github.com/VRAXION/VRAXION/wiki/Theory-of-Thought) (wiki)

---

Version: v3.0.001 (2026-02-16)
