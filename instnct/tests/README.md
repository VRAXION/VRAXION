# INSTNCT Test Suite

## Core Validation (run on every change)

| File | Description |
|------|-------------|
| `test_model.py` | **22 adversarial probes** — stress tests for graph.py: NaN injection, empty/full networks, save/load fidelity, mutation determinism, batch consistency, alive cache coherence, etc. |
| `resonator_toy_test.py` | Deterministic resonator chamber test: P1 separation, P2 selectivity, P3 energy decay, P4 determinism, P5 reciprocal ablation. No training, no randomness. |

## Resonator Theory Analysis

These scripts produced the findings documented in `instnct/RESONATOR_THEORY.md`.
They are informational/analytical — run them to reproduce results, not as pass/fail gates.

| File | Description | Requires |
|------|-------------|----------|
| `resonator_scale_test.py` | Multi-scale sweep (32-512 neurons): is optimal inhib/reciprocal ratio universal? | numpy only |
| `resonator_weight_resolution_test.py` | Weight resolution sweep (binary → float32): how many bits do edges need? Tests fly-brain config with graded weights and varying I→E strength. | numpy only |
| `flywire_analysis.py` | FlyWire Drosophila connectome (139K neurons, 16.8M connections): SCC, reciprocal enrichment, clustering, NT breakdown. | pandas, pyarrow, FlyWire data in `data/flywire/` |

## A/B Experiments (historical — document results)

These were run during development to validate specific hypotheses.
Results are recorded in commit messages and `RESONATOR_THEORY.md`.

| File | Result |
|------|--------|
| `spike_readout_ab.py` | Charge readout slightly better than spike readout (-1 to -2.8%) |
| `spike_readout_ab_v2.py` | Theta×readout interaction: charge+theta=2 identical to charge+theta=0 |
| `learnable_rho_ab.py` | Learnable rho +1.6% accuracy; rho≥0.25 lethal before theta clip fix |
| `loop_mutation_ab.py` | Loop macro-mutations hurt on permutation task |
| `loop_sequential_ab.py` | Loop mutations mixed results on sequential task |
| `test_binary_vs_ternary.py` | Binary {0,1} masks viable; negative edges not critical |
| `test_c19_vs_relu_nonlinearity.py` | C19 wave vs ReLU on XOR/parity tasks |
| `test_edge_magnitude_ab.py` | Edge magnitude learning A/B |
| `test_knob_conditioned.py` | Knob-conditioned training A/B |

## Running Tests

```bash
# Core validation (fast, no dependencies beyond numpy)
python -m pytest instnct/tests/test_model.py -v
python instnct/tests/resonator_toy_test.py

# Full resonator analysis suite (~2 min)
python instnct/tests/resonator_scale_test.py
python instnct/tests/resonator_weight_resolution_test.py

# FlyWire connectome analysis (requires data download, ~30s)
python instnct/tests/flywire_analysis.py

# Syntax check all test files
python -m py_compile instnct/tests/*.py
```
