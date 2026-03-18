# VRAXION Self-Wiring Graph

This repository is the slimmed `main` line for the self-wiring graph work.
Everything outside the active self-wiring path has been moved out of the mainline scope.

## Scope

`main` is now intentionally focused on the `v4.2` self-wiring graph stack:

- `v4.2/model/graph.py`: reference NumPy self-wiring graph
- `v4.2/model/graph_v3.c`: C backend and scaling path under active evaluation
- `v4.2/lib/utils.py`: scoring and cyclic training helpers
- `v4.2/tests/`: sweeps, smoke tests, scaling probes, and benchmark scripts

The half-ready `surprise` learning path is not part of this cleaned mainline. It needs redesign before it belongs on the default branch again.

## What The Model Is

The current self-wiring graph is a gradient-free graph learner with:

- flat directed graph topology
- ternary signed edges baked into the mask
- persistent charge and state dynamics
- mutation plus selection as the training loop
- cyclic `rewire -> crystallize` training support

The stable reference implementation is [`v4.2/model/graph.py`](v4.2/model/graph.py).

## Repo Layout

- [`v4.2/README.md`](v4.2/README.md): module-level map
- [`v4.2/model/graph.py`](v4.2/model/graph.py): reference Python model
- [`v4.2/model/graph_v3.c`](v4.2/model/graph_v3.c): C implementation path
- [`v4.2/lib/utils.py`](v4.2/lib/utils.py): training helpers
- [`v4.2/tests/`](v4.2/tests/): experiment scripts and results
- [`v4.2/CREDIT_GUIDED_REWIRING.md`](v4.2/CREDIT_GUIDED_REWIRING.md): design note for credit-guided rewiring
- [`v4.2/tests/benchmark_ab.py`](v4.2/tests/benchmark_ab.py): SWG vs MLP vs random-search benchmark
- [`v4.2/tests/benchmark_expressiveness.py`](v4.2/tests/benchmark_expressiveness.py): expressiveness benchmark across task families
- [`v4.2/tests/ab_rng_knee.py`](v4.2/tests/ab_rng_knee.py): random-source knee benchmark
- [`v4.2/viz/swg_demo.html`](v4.2/viz/swg_demo.html): interactive self-wiring graph viewer

## Quickstart

Create a Python environment and install the minimal dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run the reference stress test:

```bash
python v4.2/tests/test_model.py
```

Run a quick RNG benchmark:

```bash
python v4.2/tests/rng_tier_benchmark.py --seeds 2 --vocab 32 --attempts 2000
```

## Notes On Branch Policy

- `main` should stay small and self-wiring focused.
- Historical non-self-wiring lines should live on archive branches, not in the default branch tree.
- Experimental branches are fine, but they should not redefine what `main` is about until they beat the current self-wiring baseline.

## License

- Noncommercial: [LICENSE](LICENSE)
- Commercial terms: [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)
- Citation: [CITATION.cff](CITATION.cff)
