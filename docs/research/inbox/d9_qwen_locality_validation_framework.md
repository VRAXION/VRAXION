# D9 Inbox: Qwen Locality Validation Framework

Source: `C:/Users/kenes/Downloads/Validating Locality in Latent Genome Compilers_ A Falsifiable Framework for Progressive Behavior Space Exploration.pdf`

Date ingested: 2026-04-28

Status: source digest, not yet accepted architecture.

## Executive Read

Qwen's report reinforces the consensus and adds one especially useful structure:

```text
D9.0 should have two sequential killer gates:

Gate 1: locality + validity
Gate 2: progressive behavior-basin scan
```

This is the cleanest framing so far. It prevents a common failure:

```text
structural locality passes,
but behavior search is still useless
```

Qwen also cleanly answers where toy basins should live:

```text
primary basins: behavior space phi
diagnostic basins: genome space and z-space
```

## Accepted Claims For D9.0

### Three Spaces Must Stay Separate

Qwen distinguishes:

```text
z-space:
  where we sample and scan

genome space:
  where structural locality is measured

behavior space phi:
  where success/failure is ultimately evaluated
```

Accepted:

D9.0 must not collapse these into one descriptor. If toy behavior is identical to genome descriptors, the identity decoder can fake success.

### Behavior-Space Basins Are Primary

Qwen states that the main benchmark basins should be defined primarily in behavior space `phi`.

Accepted:

The real VRAXION goal is behavior/function, not just graph geometry. Therefore:

```text
smooth, deceptive, multi-basin, needle basins should be phi-grounded
```

But include diagnostics:

```text
genome-grounded basin:
  tests structural locality only

z-grounded basin:
  scanner sanity check only
```

### Two Sequential Gates

Qwen proposes:

```text
Gate 1: Locality and Validity
  - valid genome generation
  - deterministic output
  - z -> genome locality
  - z -> behavior locality at least above controls
  - negative controls fail

Gate 2: Progressive Scan Efficiency
  - find behavior basins faster than random/hash controls
  - tile enrichment
  - basin clustering quality
```

Accepted:

Gate 2 should never run as a success claim if Gate 1 fails.

### Negative Controls

Qwen includes:

- random hash control
- identity decoder control
- nonlocal control

Accepted with refinement:

The identity decoder definition in Qwen's PDF is framed as "valid only if z corresponds to a prerecorded good point." Our synthesis should use a broader identity/raw-byte pitfall:

```text
D_IDENTITY_RAW:
  maps z bytes/features directly to genome bytes/features
  may pass structural locality trivially
  must fail semantic/behavior/progressive gates
```

Both forms are useful:

```text
memorized-good-point identity control
raw-byte identity control
```

### Hidden Shortcut Risk

Qwen explicitly warns that a compiler can manipulate simple genome features like depth or parameter count and appear to pass without real functional structure.

Accepted:

D9.0 should include shortcut diagnostics:

- compare behavior basins against genome-only basins
- include behavior metrics that are not direct genome descriptors
- test whether target-cell hits remain after controlling for size/depth/edge count

## Metrics To Keep

Qwen's metric categories:

### Validity And Determinism

```text
valid_network_rate
exact_roundtrip_rate for compiler-native genomes
approximate_inverse_reconstruction_quality for old networks
```

D9.0 narrowing:

- native exact roundtrip means log `z`, compiler version, seed, and graph hash.
- approximate inverse is deferred.

### Structural Locality

```text
z-distance vs genome edit distance
```

Use random `z` perturbation and pairwise samples.

### Behavioral Searchability

```text
z-distance vs behavior distance
target_cell_hit_rate / tile_enrichment
basin_clustering_quality
```

These are the real gates. Structural locality alone is insufficient.

## Toy Benchmarks To Keep

Qwen proposes:

```text
smooth basin          behavior space
deceptive basin       behavior space
multiple basins       behavior space
needle basin          behavior space
genome-grounded basin genome space diagnostic
z-grounded basin      z-space sanity check
```

Accepted.

Important final interpretation:

```text
z-grounded basin pass proves the scanner can work on an easy latent geometry.
genome-grounded basin pass proves structural locality.
behavior-grounded basin pass is the only meaningful D9 utility signal.
```

## Roadmap Alignment

Qwen's roadmap:

```text
D9.0 toy latent genome decoder audit
D9.1 VRAXION-lite deterministic genome grammar
D9.2 latent scan + archive atlas
D9.3 inverse search for old networks
D9.4 optional learned proposal/decoder augmentation
```

Accepted with one constraint:

D9.1 is blocked until canonical genome/provenance is defined for the real VRAXION side. The toy can proceed without it.

## Risk Claims To Preserve

Qwen lists several high-value risks:

### Behavior Locality May Fail

The compiler may show:

```text
z -> genome locality: pass
z -> behavior locality: fail
```

This means the compiler is geometrically smooth but functionally useless.

### Diversity May Collapse

A valid grammar can produce too narrow a family of networks.

Required D9.0 check:

```text
behavior coverage / unique basin discovery / descriptor entropy
```

### Benchmark Overfitting

Toy pass may not transfer to VRAXION.

Mitigation:

D9.1 VRAXION-lite should emulate real D8/D4 failure modes, not generic toy tasks only.

### Hidden Shortcut / Semantic Misinterpretation

The compiler may learn shallow structural proxies.

Mitigation:

Behavior basins should be independent of simple graph descriptors where possible.

## Comparison Against Gemini/GPT/Grok/Claude Wave-1

Agreement:

- deterministic compiler first
- learned decoders deferred
- exact inverse only for generated/logged genomes
- behavior-cell direct inversion rejected
- negative controls required
- progressive scan efficiency required

Qwen-specific useful addition:

- clean two-gate structure
- primary basins in behavior space, diagnostics in genome/z
- shortcut-risk framing
- z-grounded sanity basin vs genome-grounded diagnostic basin distinction

Claude wave-1-compatible additions:

- identity decoder pitfall is covered, but should be broadened
- behavior metric coverage bias should be tested
- regular vs irregular behavior basin should be explicit

## Bottom Line

Qwen provides the best gate structure so far:

```text
Gate 1:
  Is the compiler valid, deterministic, and non-hashlike?

Gate 2:
  Does its geometry actually help find behavior basins?
```

D9.0 should pass both before any VRAXION-lite or Rust integration.
