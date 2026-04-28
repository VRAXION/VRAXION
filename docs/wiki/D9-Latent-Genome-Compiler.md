# D9 Latent Genome Compiler

> Status: research synthesis and next experimental plan. No live runtime change has been made by D9 yet.

## Short Version

D8 built a behavior atlas: existing network states were evaluated, fingerprinted, and placed into behavior cells.

D9 asks a different question:

```text
Can we define a deterministic coordinate space where:

  z coordinate -> genome compiler -> valid network -> evaluation -> behavior fingerprint

and where nearby z values statistically tend to produce related networks?
```

The first goal is not to prove that this works. The first goal is to falsify it cheaply.

D9.0 will therefore be an offline Python toy audit with strict negative controls. It will not modify SAF, Rust runtime search, K(H), operator schedules, archive parent selection, or acceptance rules.

## Why D9 Exists

D8 showed that archive memory and behavior-cell structure are useful for organizing observed states. The D8 atlas can answer:

```text
Given a network state we already observed, where does it land in behavior space?
```

But it cannot directly answer:

```text
Given a point in a searchable genome space, what network does it generate?
```

That is the missing layer D9 tests.

```text
D8: observed-state map

  network g -> evaluate -> behavior fingerprint phi(g) -> atlas cell

D9: generative genome map

  coordinate z -> compiler D(z) -> genome g -> evaluate -> phi(g), score, psi
```

The distinction matters. A behavior atlas can label animals after observing them. A genome compiler is a factory that produces animals from a deterministic recipe.

## Core Hypothesis

D9 tests whether VRAXION can use a structured, deterministic genome compiler instead of an opaque random hash.

Desired properties:

- Validity: most z values compile into executable network genomes.
- Statistical locality: nearby z values usually compile into related genomes.
- Behavior relevance: related genomes should show more related behavior than random-hash controls.
- Exact provenance: every generated genome stores the z that produced it.
- Falsifiability: if the compiler is hash-like, D9.0 should fail quickly.

Non-goals for D9.0:

- No learned decoder.
- No live Rust integration.
- No archive steering.
- No mutation operator changes.
- No claim that this is a universal coordinate system for intelligence.

## Primary Design: PIC

The first D9.0 decoder is **PIC: Parametric InitConfig**.

PIC treats the existing VRAXION initialization/configuration surface as a set of continuous dials. Instead of a random string deciding a network, each z field controls a concrete structural property.

Examples of PIC-style dials:

- edge density,
- inhibitory fraction,
- chain density,
- threshold regime,
- channel bias,
- recurrence bias,
- mirror symmetry,
- modularity,
- irregularity amplitude,
- irregularity phase.

The key design rule:

```text
z controls structural quantities.
root_seed only breaks ties.
```

The seed must not decide how many edges exist, which rule runs, or what regime the network enters. If the seed dominates, the compiler degenerates into a random hash.

## Why Not Start With A Neural Decoder?

Grammar-constrained learned decoders can be valid and coherent. Grammar VAE is a useful example: a learned model constrained by a grammar can generate valid parse trees and a more coherent latent space than an unconstrained baseline.

D9.0 still starts with a deterministic compiler because it is easier to audit:

- no hidden training distribution,
- no opaque learned failure mode,
- reproducible output for every z,
- simpler negative controls,
- easier graph-distance and provenance checks.

Learned decoders are deferred to a later phase only if the deterministic baseline is insufficient.

## The Red-Team Finding

The most important D9 research result came from adversarial review.

An earlier proposed "killer test" used byte-level distance to check locality. The red team constructed an adversarial decoder, called `D_advr`, that could pass that naive test while carrying no meaningful graph structure.

That means the naive test was already broken before implementation.

The corrected first gate is now:

```text
IDENTITY_AUGMENTED_KILLER
```

It is an internal red-team protocol, not a published benchmark.

It requires:

- graph-edit distance, not raw byte Hamming distance,
- multi-component non-triviality checks,
- explicit separation from `D_advr`,
- random-hash and non-local decoder controls,
- calibrated thresholds against the run's own controls.

If `D_advr` also passes, the test itself is invalid.

## D9.0 Test Plan

D9.0 is an offline Python audit.

Expected files:

```text
tools/analyze_phase_d9_latent_genome_toy.py
docs/research/PHASE_D9_0_LATENT_DECODER_TOY_AUDIT.md
output/phase_d9_latent_genome_toy_<YYYYMMDD>/
```

The first test is fail-fast:

```text
1. Sample random z values.
2. Decode with PIC, random-hash control, and D_advr.
3. Compare z-distance to graph-edit distance.
4. Verify non-trivial graph structure.
5. Verify real decoder beats controls by calibrated margins.
6. Stop immediately if any DNP gate fires.
```

If the first gate passes, D9.0 then runs toy landscapes:

- smooth basin,
- deceptive basin,
- multi-basin landscape,
- needle/high-variance landscape,
- random-control landscape.

These are not intended to prove live VRAXION improvement. They test whether the compiler is worth connecting to the real system later.

## Verdict Gates

D9 uses "Do Not Proceed" gates. These are engineering audit thresholds, not laws of nature.

Key gates:

- `DNP_VALIDITY_COLLAPSE`: too many z values fail to produce valid networks.
- `DNP_LOCALITY_COLLAPSE`: z-distance does not predict genome distance better than controls.
- `DNP_BEHAVIOR_HASHLIKE`: z-distance does not predict behavior distance better than controls.
- `DNP_SCAN_NO_GAIN`: progressive scanning does not beat random scanning.
- `DNP_CONTROL_PARITY`: negative controls pass the same gates as the real compiler.
- `DNP_TOO_HEAVY`: runtime or dependency budget is violated.

Candidate verdict names:

```text
D9_LATENT_DECODER_TOY_PASS
D9_DECODER_VALIDITY_FAIL
D9_DECODER_NO_LOCALITY
D9_DECODER_HASHLIKE_BEHAVIOR
D9_TILE_SCAN_NO_SIGNAL
D9_CONTROL_PARITY_FAIL
```

## What D9 Does Not Claim

D9 does not claim:

- that a useful latent genome space already exists,
- that the literature proves this will work,
- that PIC has a mathematical worst-case locality guarantee,
- that behavior cells can be inverted into genomes,
- that this is a coordinate system for consciousness or intelligence,
- that learned decoders are bad.

D9 claims only this:

```text
The idea is plausible enough to test,
dangerous enough to test adversarially,
and cheap enough to falsify offline before touching live search.
```

## Literature Position

### Grammatical Evolution Locality

Rothlauf and Oetzel show that deterministic genotype-to-phenotype mappings can still have poor locality. This is the strongest cautionary precedent for D9.

Source: [On the Locality of Grammatical Evolution](https://openreview.net/forum?id=JMtLlNTIJf)

### Indirect Encodings And Regularity

Clune, Stanley, Pennock, and Ofria show that indirect encodings such as HyperNEAT exploit regularity, but can struggle with irregular exceptions. Therefore D9.0 must include irregular/deceptive toy tests.

Source: [On the Performance of Indirect Encoding Across the Continuum of Regularity](https://cse.msu.edu/~mckinley/Pubs/files/Clune.TEC.2011.pdf)

### Grammar-Constrained Latent Models

Grammar VAE supports the importance of validity constraints in discrete latent generation. It does not imply D9.0 should start with a learned decoder.

Source: [Grammar Variational Autoencoder](https://proceedings.mlr.press/v70/kusner17a.html)

### MAP-Elites And Behavior Archives

MAP-Elites supports the archive/behavior-space framing used in D8. In the canonical MAP-Elites loop, new candidates are generated from stored elites, not from cell coordinates themselves. D9's coordinate-as-generator idea is therefore a new design choice.

Source: [Illuminating search spaces by mapping elites](https://arxiv.org/abs/1504.04909)

### Latent Space Illumination

Latent-space quality-diversity methods exist, especially around learned generators such as GANs. D9.0 is deliberately more primitive and auditable: a hand-designed deterministic compiler first, learned components later.

Sources:

- [Illuminating Mario Scenes in the Latent Space of a GAN](https://arxiv.org/abs/2007.05674)
- [Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space](https://arxiv.org/abs/1912.02400)

### FDC And Mantel Cautions

Fitness-distance correlation is not safe as a standalone gate. Mantel-style distance tests are useful for D9's distance-hypothesis, but need careful controls and calibration.

Sources:

- [Fitness Distance Correlation Analysis: An Instructive Counterexample](https://dynamics.org/~altenber/PAPERS/FDCAAIC/)
- [Benchmarking the Mantel test and derived methods](https://agp.unige.ch/en/outreach/benchmarking-the-mantel-test-and-derived-methods-for-testing-association-between-distance-matrices)

## Internal Source Map

Primary internal research artifacts:

- [D9 open threads citation audit](https://github.com/VRAXION/VRAXION/blob/main/docs/research/PHASE_D9_OPEN_THREADS_CITATION_AUDIT.md)
- [Claude swarm final synthesis](https://github.com/VRAXION/VRAXION/blob/main/docs/research/inbox/d9_claude_final_synthesis.md)
- [Gemini genome compiler digest](https://github.com/VRAXION/VRAXION/blob/main/docs/research/inbox/d9_gemini_genome_compiler_design.md)
- [GPT latent genome compiler digest](https://github.com/VRAXION/VRAXION/blob/main/docs/research/inbox/d9_gpt_deep_research_latent_genome_compiler.md)
- [Grok latent genome decoder digest](https://github.com/VRAXION/VRAXION/blob/main/docs/research/inbox/d9_grok_latent_genome_decoder.md)
- [Qwen locality validation framework](https://github.com/VRAXION/VRAXION/blob/main/docs/research/inbox/d9_qwen_locality_validation_framework.md)

## Roadmap

```text
D9.0
  Offline toy compiler audit.
  Python only. No Rust runtime changes.

D9.1
  VRAXION-lite deterministic genome grammar after D9.0 passes.
  Requires canonical genome serialization.

D9.2
  Latent scan plus behavior atlas integration.

D9.3
  Approximate inverse search for legacy networks.
  Exact inverse only exists for D9-generated genomes with stored z.

D9.4
  Optional grammar-constrained learned proposal/decoder.
  Only after deterministic baseline is understood.
```

## Practical Next Step

Implement D9.0 exactly as an offline falsification tool.

Do not add:

- live search,
- archive steering,
- neural decoders,
- mutation operators,
- unrelated cleanup,
- Rust runtime changes.

The next meaningful result is not "D9 works." The next meaningful result is one of:

```text
D9_LATENT_DECODER_TOY_PASS
or
an explicit D9 failure verdict that tells us which assumption died.
```

