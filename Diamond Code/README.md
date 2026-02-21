# Diamond Code v3 — SwarmByteRingModel

**Dual-model neural memory architecture** with hash-bucketed long-term consensus memory (LCX/LTCM), C19 periodic activation, and golden ratio fractal dimensioning.

## Two-Model Strategy

| | **Goldilocks Ant** | **Goldilocks Nano** |
|---|---|---|
| Role | GPU research model | CPU edge model |
| Params | 424M | 31.2M |
| D | 6180 | 618 |
| Depth | 12 | 62 |
| seq_len | 62 | 6 |
| num_bits | 8 (expandable to 618) | 6184 (binary-bits) |
| Context | 496 bytes/pass | 4,638 bytes/pass |
| LCX | 2000 slots, key_dim=618 | 2000 slots, key_dim=61 |
| Device | RTX 4070 Ti SUPER (bf16) | CPU only (fp64) |
| Step time | ~5.3s (tt=1) | ~1.8s (tt=1) |
| Deploy | ~800 MB fp16 | 59.5 MB fp16, 0.7 GB RAM |
| Launcher | `run_goldilocks.bat` | `run_nano_golden.bat` |

Both models share the same `SwarmByteRingModel` architecture with golden ratio fractal dimensioning (phi = 0.61803):
- Scale 1: phi x 10 = 6 (top_k, nano seq_len)
- Scale 2: phi x 100 = 62 (nano depth, ant seq_len/ring)
- Scale 3: phi x 1000 = 618 (nano D, ant key_dim)
- Scale 4: phi x 10000 = 6180 (ant D, nano num_bits)

## Architecture

**LCX Read Bottleneck** (all 10 levers locked, v3.2.000):
```
lcx_read [D] -> Lin(D->D/10) -> C19 -> Lin(D/10->D/10) -> C19 -> Lin(D/10->D) x zoom_gate -> + hidden
```
- C19 periodic activation at all 6 sites (replaces GELU)
- Orthogonal init, no LayerNorm, no residual skip
- zoom_gate_init = 0.0 (balanced, model learns to open)
- Bottleneck at both input-time and think-tick integration points

**Key Components:**
- **Ring Memory**: Circular buffer with soft pointer addressing and content-based jump routing
- **LCX**: Hash-bucketed memory slots with SimHash retrieval (~512 slots searched per query)
- **Progressive Training**: 6 effort tiers (Alpha through Zeta), batch/tt tradeoff
- **Binary-Bits Encoding**: 25x more parameter-efficient than byte-token (PF-013)

## Key Files

| File | Description |
|------|-------------|
| `swarm_model.py` | Core model (3000+ lines) |
| `test_swarm_config.py` | Training loop and evaluation |
| `traindat_loader.py` | Data loading (vectorized, 3837x speedup) |
| `byte_data.py` | Metrics and byte-level utilities |
| `influx_writer.py` | InfluxDB telemetry writer |
| `live_controls.py` | Live Grafana control interface |
| `frame_renderer.py` | Visualization and frame rendering |
| `golden_disc.py` | Disc format helper |
| `generate_traindat_suite.py` | Training data generation |
| `launch_nano_golden.py` | Nano Python launcher (avoids shell escaping) |
| `run_goldilocks.bat` | Ant GPU training launcher |
| `run_nano_golden.bat` | Nano CPU training launcher |
| `tools/control_panel.py` | HTTP control panel for Grafana |
| `tools/checkpoint_autopsy.py` | Checkpoint forensics |
| `tools/lcx_forensics.py` | LCX memory analysis |
| `tools/validate_traindat.py` | Training data validation |

## Observability Stack

- **Grafana** (`:3000`): Primary dashboard — 66 panels, 8 sections
- **InfluxDB** (`:8086`): Time-series metrics (bucket: `diamond`)
- **Control Panel** (`:7777`): HTTP bridge for bidirectional Grafana controls
- **Streamlit** (`:8501`): Legacy dashboard

## Quick Start

```bash
# Generate training data
python generate_traindat_suite.py

# Launch Ant (GPU) training
run_goldilocks.bat

# Launch Nano (CPU) training
python launch_nano_golden.py

# Grafana dashboard at http://localhost:3000
```

## Proven Findings

| ID | Finding | Evidence |
|----|---------|----------|
| PF-001 | sqrt(N) scaling law: K = N^(1/2) optimal connectivity | E4, Diamond Code swarm |
| PF-002 | Jump gate prevents dead beings in swarm | E3, N=64 experiment |
| PF-003 | Ring memory + soft pointers > hard discrete jumps | E3, ablation |
| PF-004 | One HD level + grow > multi-level LCX | E4, SimHash bucketing |
| PF-005 | AGC unnecessary for single-being training | E3, TOT-H007 |
| PF-006 | Score Margin tracks routing quality (4.5% -> 53%) | E3, 250 Beta steps |
| PF-007 | Address-before-content: model learns WHERE first | E3, Phase A/B pattern |
| PF-008 | tt=1 optimal from random init (higher tt amplifies noise) | E3, mini model probe |
| PF-009 | C19 > GELU (+4.2% BitAcc, mini model) | E3, activation probe |
| PF-010 | 2x618 bottleneck: sweet spot (95.5% vs 94.6% 1x) | E3, architecture probe |
| PF-011 | No residual in bottleneck: proj(x) beats x+proj(x) by 9.4% | E3, lever 9 probe |
| PF-012 | Autoregressive ceiling: only 2/11 tasks solvable at seq_len<=16 | E4, difficulty ladder |
| PF-013 | Binary-bits 25x more efficient than byte-token | E4, encoding probe |
| PF-014 | Golden ratio fractal Nano: 31.2M params, 1.64s/step CPU | E4, bench probe |

Full details: [Proven Findings](https://github.com/VRAXION/VRAXION/wiki/Proven-Findings) (wiki)

## Environment

- GPU: NVIDIA RTX 4070 Ti SUPER (16 GB VRAM)
- CPU: Training supported (Nano model, ~1.8s/step, 0.7 GB RAM)
- Python 3.11, PyTorch with CUDA, bfloat16 mixed precision
- InfluxDB 2.x, Grafana 11.x

## Documentation

- [Diamond Code v3 Architecture](https://github.com/VRAXION/VRAXION/wiki/Diamond-Code-v3-Architecture)
- [Proven Findings](https://github.com/VRAXION/VRAXION/wiki/Proven-Findings)
- [Probe Archive v3](https://github.com/VRAXION/VRAXION/wiki/Probe-Archive-v3) — all experiment results
- [Session Log Feb 21](https://github.com/VRAXION/VRAXION/wiki/Session-Log-Feb-21-2026) — Zoom gate unfreeze + sleep cycles
- [Session Log Feb 18](https://github.com/VRAXION/VRAXION/wiki/Session-Log-Feb-18-2026) — Nano launch + binary-bits
- [Session Log Feb 17](https://github.com/VRAXION/VRAXION/wiki/Session-Log-Feb-17-2026) — 10 levers locked
- [Theory of Thought](https://github.com/VRAXION/VRAXION/wiki/Theory-of-Thought)

---

Version: v3.4.000 (2026-02-21) — Zoom gate gradient unfreeze (AGC exempt). Sleep cycle validation (6 cycles, 3 promotions). Beta language training stable (73% eval, 85% peak). Dashboard consolidated (3→1).
