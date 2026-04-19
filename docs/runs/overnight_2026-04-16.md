# VRAXION overnight findings — 2026-04-16 to 2026-04-17

Migrated into the repo from a local session artifact and cross-checked against reruns on the 30 MB FineWeb corpus.

## TL;DR

- `α` did not survive an iso-parameter test. Its earlier win was a parameter-count illusion.
- `δ` is statistically tied with `B0` at the same parameter budget, with a more bounded, self-regularizing output profile.
- The `α+δ` combination is worse than either alone at iso-param scale.

## Corpus and method

- Corpus: `<local FineWeb-EDU sample, not included in repo>`
- Train setup: 300 epochs, 2000 samples/epoch
- Seeds: `42`, `1337`, `9999`
- Comparison target:
  - `B0`: Beukers baseline, `nf=128`, 2 projections, `32,411` params
  - `α-iso`: modulated Beukers, `nf=88`, 3 projections, `32,235` params
  - `δ`: group-norm Beukers, `nf=128`, 2 projections, `32,411` params
  - `α+δ combo`: combined per-filter damping plus group norm, `nf=88`, 3 projections, `32,235` params

## Iso-param multi-seed results

| Variant | Params | Best test mean +- std | Seeds (42 / 1337 / 9999) | `co_max_mean` |
|---|---:|---:|---|---:|
| `B0` Beukers | 32,411 | `74.03 +- 1.05%` | `75.10 / 74.40 / 72.60` | `0.943` |
| `δ` Group-norm | 32,411 | `73.47 +- 0.86%` | `73.50 / 74.50 / 72.40` | `0.702` |
| `α-iso` Modulated | 32,235 | `72.10 +- 0.22%` | `72.40 / 71.90 / 72.00` | `0.923` |

### Statistical read

- `B0` vs `α-iso`: diff `1.93pp`, combined SE `0.62`, t-stat `3.11`
  - Interpretation: significant, `α-iso` is genuinely worse at iso-param budget.
- `B0` vs `δ`: diff `0.56pp`, combined SE `0.79`, t-stat `0.71`
  - Interpretation: not significant, effectively a tie on this 30 MB setup.
- `δ` vs `α-iso`: diff `1.37pp`, combined SE `0.52`, t-stat `2.63`
  - Interpretation: significant, `δ` beats `α-iso`.

## What this means

The earlier single-seed `α` win came from giving it an extra projection and about 45% more parameters. Once the parameter budget is matched, `B0` comes out ahead.

The likely mechanism is that `α` spends a full convolutional projection on a damping signal that behaves more like a per-filter scalar control than a rich feature extractor. At small or iso-param width, that is an expensive use of capacity.

`δ` does not beat `B0`, but it does have a distinct character:

- much lower `co_max` (`0.70` vs `0.94`)
- self-regularizing group behavior
- better-bounded outputs

That makes `δ` a credible stability-oriented alternative even without an accuracy win on this corpus size.

## Alpha-plus-delta combo

| Variant | Params | Best test mean +- std | Seeds (42 / 1337 / 9999) |
|---|---:|---:|---|
| `B0` Beukers | 32,411 | `74.03 +- 1.05%` | `75.10 / 74.40 / 72.60` |
| `α+δ combo` | 32,235 | `71.60 +- 0.08%` | `71.60 / 71.50 / 71.70` |

### Read

- `α+δ combo` is `1.87pp` worse than `B0`
- `α+δ combo` is `0.50pp` worse than `α-iso`
- combo `gn_mean` was about `0.31`, below the earlier `δ`-alone reading of about `0.39`

Interpretation: the two damping mechanisms compose in the wrong direction here. `α` already suppresses per-filter output, so the extra group norm mostly reduces signal further instead of adding useful control.

The very low variance (`+- 0.08`) suggests a stable but weak attractor rather than a noisy failure.

## Reproducibility

Examples added in the working tree:

- `instnct-core/examples/diag_isoparam_multiseed.rs`
- `instnct-core/examples/diag_alpha_delta_combo.rs`

Representative commands:

```powershell
cargo run -p instnct-core --release --example diag_isoparam_multiseed -- "<local FineWeb-EDU sample, not included in repo>"
cargo run -p instnct-core --release --example diag_alpha_delta_combo -- "<local FineWeb-EDU sample, not included in repo>"
```

## Bottom line

`B0` remains the strongest current direction on FineWeb 30 MB at matched parameter budget.

- `B0`: best current default
- `δ`: valid alternative with a more bounded stability profile
- `α`: not justified as an architecture win at iso-param scale
- `α+δ`: negative finding, not worth promoting
