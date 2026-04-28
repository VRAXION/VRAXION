# Phase D9 Open Threads Citation Audit

Date: 2026-04-28

Scope: targeted citation and adversarial wording pass for the upcoming public D9 latent genome compiler report. This is not a new D9 design document. It checks whether the strongest D9 claims are externally supportable, should be softened, or should remain internal-only engineering assumptions.

## Verdict

The D9 public report is ready to draft after wording hardening.

No new literature found that invalidates the D9.0 direction. The main correction is framing: D9.0 should be presented as a falsifiable deterministic genome-compiler audit, not as an established method. Several thresholds and red-team controls are internal engineering gates, not literature-derived facts.

## Publication-Safe Claims

### 1. Deterministic grammar/compiler mappings can fail locality

Status: supported.

Rothlauf and Oetzel's Grammatical Evolution locality paper is the strongest cautionary precedent. Its abstract states that GE has locality problems because many neighboring genotypes do not correspond to neighboring phenotypes, and mutation-based search performs worse than high-locality alternatives.

Public wording:

> Prior work on Grammatical Evolution shows that deterministic genotype-to-phenotype mappings can still have poor locality, so D9.0 treats locality as something to measure, not assume.

Source:
- https://openreview.net/forum?id=JMtLlNTIJf

### 2. Indirect encodings help on regular structure but struggle with irregular exceptions

Status: supported.

Clune, Stanley, Pennock, and Ofria show HyperNEAT benefits from problem regularity, while performance decreases on irregular problems. Their HybrID result supports the idea that an indirect regularity-producing encoding can benefit from a refinement mechanism for irregular exceptions.

Public wording:

> Indirect encodings are useful when the problem contains exploitable regularity, but they are not magic. D9.0 must include irregular/deceptive toy landscapes because regular-only tests would be too favorable.

Source:
- https://cse.msu.edu/~mckinley/Pubs/files/Clune.TEC.2011.pdf

### 3. Grammar constraints can improve validity and latent coherence

Status: supported, but must not be overread.

Grammar VAE demonstrates that grammar-constrained learned decoders can produce valid parse-tree outputs and more coherent latent spaces. This supports the importance of structure/grammar, not the claim that learned decoders should be used in D9.0.

Public wording:

> The load-bearing idea is not "deterministic beats learned." The load-bearing idea is that the decoder must impose valid structure. D9.0 chooses a deterministic compiler first because it is easier to audit and falsify.

Source:
- https://proceedings.mlr.press/v70/kusner17a.html

### 4. MAP-Elites supports archive/behavior-space thinking, but D9 is not standard MAP-Elites

Status: supported.

MAP-Elites stores elites in behavior-feature cells and generates new candidates by selecting existing elites and mutating them. This supports D8's archive/atlas direction. It does not directly validate D9's proposed use of a latent coordinate as a generative seed.

Public wording:

> D8 follows the MAP-Elites intuition of mapping high-performing diverse states. D9 goes one step earlier: it asks whether we can define a generative coordinate system that produces valid candidate genomes before they are evaluated.

Sources:
- https://arxiv.org/abs/1504.04909
- https://arxiv.org/pdf/1504.04909

### 5. Latent-space QD exists, but it usually searches a learned generator's latent input, not a hand-audited genome compiler

Status: supported.

Fontaine et al. use QD algorithms to illuminate GAN latent space for Mario scene generation. CMA-ME combines CMA-ES adaptation with MAP-Elites style archiving. This is relevant but not identical to D9: D9's compiler is deterministic and manually auditable rather than a trained GAN.

Public wording:

> Latent-space illumination is a real family of methods, but D9.0 is deliberately more primitive: a hand-audited compiler with negative controls before any learned generator is allowed.

Sources:
- https://arxiv.org/abs/2007.05674
- https://arxiv.org/abs/1912.02400

### 6. FDC alone is unsafe as a gate

Status: supported.

Altenberg gives a counterexample where Hamming-distance-based FDC fails to predict GA difficulty. D9 should keep FDC diagnostic-only, not as a standalone pass/fail criterion.

Public wording:

> We do not use fitness-distance correlation as a standalone verdict because known counterexamples show it can mislead when the distance measure is not aligned with the actual search operator.

Source:
- https://dynamics.org/~altenber/PAPERS/FDCAAIC/

### 7. Mantel tests require careful framing

Status: supported with nuance.

Quilodran et al. benchmark Mantel-derived tests and discuss power/type-I-error concerns under spatial autocorrelation and variable-type transformations. The correct D9 takeaway is not "Mantel is bad"; it is "use distance-variable hypotheses, random/non-autocorrelated sampling where possible, controls, and calibration."

Public wording:

> Mantel-style distance correlation is useful for the specific question "do nearby z values decode to nearby genomes/behaviors?", but the test must be paired with randomization controls and calibrated negative baselines.

Source:
- https://agp.unige.ch/en/outreach/benchmarking-the-mantel-test-and-derived-methods-for-testing-association-between-distance-matrices
- DOI: 10.1111/1755-0998.13898

## Claims To Soften Before Public Release

### A. "PIC has a Lipschitz guarantee"

Do not publish as a guarantee. The top-K selector can break worst-case locality at near-tie score clusters. Publish as:

> PIC is designed for statistical locality, and D9.0 explicitly tests whether that design survives top-K crossing failures.

### B. "0.20 / 0.10 Mantel gaps are literature-backed thresholds"

Do not say this. These are internal engineering gates. Publish as:

> We use calibrated negative controls and require a substantial margin over the run's own hash baseline.

If numbers are included, label them as audit thresholds, not scientific constants.

### C. "All QD systems use archive occupants, never cell coordinates"

This is broadly true for MAP-Elites-style pseudocode and the cited LSI descriptions, but it is too universal for a public claim. Publish as:

> In the canonical MAP-Elites loop, new candidates are generated from stored elites, not from cell coordinates themselves. D9's coordinate-as-generator idea should therefore be treated as a new design choice, not something established by MAP-Elites.

### D. "Grammar VAE contradicts deterministic decoders"

Too strong. Publish as:

> Grammar VAE shows that learned decoders can also benefit from grammar constraints. It does not invalidate deterministic compilers; it only warns against claiming deterministic compilers are uniquely correct.

### E. "Mantel random z sampling is mandatory because grid sampling is invalid"

Too strong. Publish as:

> For locality statistics we avoid grid-only sampling because autocorrelation can distort significance tests. Grid/scan sampling remains useful for progressive scanning, but it should not be the only evidence for locality.

## Internal-Only / Do Not Present As External Literature

### 1. D_advr

`D_advr` is our internal red-team construction, not a published decoder. It is valuable and should be included in the report, but framed as an adversarial control we invented.

Safe wording:

> Our red team constructed an adversarial decoder that would pass a naive byte-distance test. That forced us to replace the naive test with a graph-edit-distance and non-triviality gate.

### 2. IDENTITY_AUGMENTED_KILLER

This is an internal protocol, not an established benchmark. Present it as a fail-fast engineering gate.

### 3. D9 verdict names

`D9_LATENT_DECODER_TOY_PASS`, `DNP_LOCALITY_COLLAPSE`, etc. are internal audit labels. They are fine in the GitHub report, but Reddit should explain them in plain English.

### 4. PIC / MMD / GPF names

These are project-specific designs. Explain them as:

- PIC: continuous knobs over the existing initialization/config space.
- MMD: motif-mixture generator.
- GPF: geometric field decoder.

Do not imply these names are known literature terms.

## Remaining Open Threads

### 1. No direct precedent for "Spearman z-distance vs graph-edit distance" in neuroevolution

The metric is reasonable, but appears to be a D9-specific composition of locality ideas rather than a standard neuroevolution benchmark. This is acceptable if framed as an audit metric.

Action:
- Keep the metric.
- Present it as an operational test, not as a literature-standard definition.

### 2. Entropy non-triviality gate needs implementation care

The Claude synthesis proposes an entropy gate to catch identity/byte-echo decoders. The idea is strong, but the exact formula should be treated as provisional. It may be better to combine:

- graph structural entropy,
- degree-distribution entropy,
- edge-pattern entropy,
- rule_trace diversity,
- and D_advr separation.

Action:
- In D9.0 implementation, do not rely on one entropy scalar alone.

### 3. Runtime claims for Mantel permutations must be measured

The 30-minute budget and 90-second killer test are plausible but not externally validated. The tool must report wall time per stage and fail/skip if too heavy.

Action:
- Keep `DNP_TOO_HEAVY`.
- Vectorize distance/permutation calculations.

### 4. D9.0 output path should follow repo convention

Claude's embedded prompt says `outputs/d9_toy_run_<timestamp>/...`. Existing project convention mostly uses `output/...`.

Action:
- Use `output/phase_d9_latent_genome_toy_YYYYMMDD/...` unless there is a strong reason otherwise.

## Recommended Framing For Reddit

Use this framing:

> We built a behavior atlas of observed network states. That told us where past states landed, but not how to generate new states directly. D9 asks whether we can define a deterministic "genome compiler": a coordinate z goes in, a valid network genome comes out. The first goal is not to prove it works. The first goal is to cheaply falsify it with negative controls, including a red-team decoder that breaks naive locality tests.

Avoid:

> We found the coordinate system for consciousness / intelligence / all networks.

Avoid:

> The literature proves this will work.

Use:

> The literature says this is plausible enough to test and dangerous enough to test adversarially.

## Source List For Public Report

- Rothlauf, F. and Oetzel, M. "On the Locality of Grammatical Evolution." EuroGP 2006. https://openreview.net/forum?id=JMtLlNTIJf
- Clune, J., Stanley, K. O., Pennock, R. T., and Ofria, C. "On the Performance of Indirect Encoding Across the Continuum of Regularity." IEEE TEC 2011. https://cse.msu.edu/~mckinley/Pubs/files/Clune.TEC.2011.pdf
- Kusner, M. J., Paige, B., and Hernandez-Lobato, J. M. "Grammar Variational Autoencoder." ICML 2017. https://proceedings.mlr.press/v70/kusner17a.html
- Mouret, J.-B. and Clune, J. "Illuminating search spaces by mapping elites." arXiv:1504.04909. https://arxiv.org/abs/1504.04909
- Pugh, J. K., Soros, L. B., and Stanley, K. O. "Quality Diversity: A New Frontier for Evolutionary Computation." Frontiers in Robotics and AI 2016. https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full
- Fontaine, M. C. et al. "Illuminating Mario Scenes in the Latent Space of a Generative Adversarial Network." AAAI 2021. https://arxiv.org/abs/2007.05674
- Fontaine, M. C. et al. "Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space." GECCO 2020. https://arxiv.org/abs/1912.02400
- Altenberg, L. "Fitness Distance Correlation Analysis: An Instructive Counterexample." ICGA 1997. https://dynamics.org/~altenber/PAPERS/FDCAAIC/
- Quilodran, C. S., Currat, M., and Montoya-Burgos, J. I. "Benchmarking the Mantel test and derived methods for testing association between distance matrices." Molecular Ecology Resources. DOI: 10.1111/1755-0998.13898. https://agp.unige.ch/en/outreach/benchmarking-the-mantel-test-and-derived-methods-for-testing-association-between-distance-matrices

