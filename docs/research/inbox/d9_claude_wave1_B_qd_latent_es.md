# D9 Wave 1 — Agent B: QD / Novelty / Latent ES Literature

## Compared to Gemini Digest

The Gemini digest proposes a deterministic z → genome compiler with locality as a measurable property, not a guarantee. The literature review **largely supports** this framing but adds three critical corrections. First, the closest historical precedent (Grammatical Evolution) is a **cautionary failure**, not a success: its deterministic genotype→phenotype compiler produced empirically poor locality, with Rothlauf and Oetzel (EuroGP 2006) finding that the large majority of neighboring genotypes do not map to neighboring phenotypes, directly hurting mutation-based search. Second, the Gemini digest underweights the distinction between *cell coordinate as generative seed* versus *stored occupant genotype as parent*: MAP-Elites uses the latter, never the former, and the two are conceptually different design choices. D9's proposal to use z as a generative seed (not a stored occupant) is a departure from MAP-Elites orthodoxy and has no direct empirical support in the QD literature. Third, the over-exploration failure of latent space optimization (LSO) is well-documented and operationally relevant: LSO in VAE spaces "tends to over-explore the latent space, resulting in unrealistic or invalid molecules" (LES, OpenReview 2023), precisely the condition D9's DNP_DECODER_HASHLIKE_BEHAVIOR gate is meant to catch.

---

## Direct Empirical Findings

**MAP-Elites cell usage — storage bin, not generative coordinate.** [direct empirical] Mouret and Clune (arXiv:1504.04909) describe MAP-Elites parent selection as: "a cell in the map is randomly chosen and the genome in that cell produces an offspring via mutation and/or crossover." The cell location itself carries zero generative weight; it is the *stored elite genotype* that seeds new search. Fontaine et al.'s Latent Space Illumination paper (arXiv:2007.05674, AAAI 2021) confirms this with GAN latent vectors: MAP-Elites, MAP-Elites (line), and CMA-ME all select archive *occupants* as parents, not cell coordinates. D9.0 implication: D9's design where z is used directly as a generative seed (not retrieved from an archive) is architecturally distinct from all tested QD systems. This is neither validated nor refuted empirically; it is a novel untested commitment.

**CMA-ME improvement emitters — archive-driven, not coordinate-driven.** [direct empirical] Fontaine et al. (arXiv:1912.02400, GECCO 2020) describe improvement emitters as modified CMA-ES instances that use *archive feedback* (whether a new solution improved an occupied cell) to update their covariance matrix and step-size. Starting point is an archive sample (occupant genotype), not a cell coordinate. CMA-ME more than doubles MAP-Elites performance on standard QD benchmarks. D9.0 implication: CMA-ME's improvement emitters are not a precedent for coordinate-seeded generation, but their "start from best known occupant and adapt covariance" logic is adaptable as a scan strategy around a given z region.

**Grammatical Evolution locality — documented failure.** [direct empirical] Rothlauf and Oetzel (EuroGP 2006) measured GE's genotype→phenotype mapping and found that the large majority of neighboring genotypes do not map to neighboring phenotypes. Performance comparison of a (1+1)-EA using GE representation versus standard GP with high locality showed GE representation leads to measurably lower search performance. This is the most direct precedent for D9: GE is a deterministic rule-based genotype→phenotype compiler, structurally analogous to D9's proposed compiler, and it failed locality tests. D9.0 implication: DNP_LOCALITY_COLLAPSE and the random hash control are not paranoid — GE shows that deterministic compilers can and do produce hash-like locality. The negative control is mandatory, not optional.

**Latent space illumination (LSI) in GAN space — partial success with documented failure modes.** [direct empirical] Fontaine et al. (arXiv:2007.05674) ran MAP-Elites, MAP-Elites(line), and CMA-ME in a 32-dimensional DCGAN latent space for Mario level generation. CMA-ME outperformed random search and MAP-Elites. Key failure: on 8-dimensional binary behavior metrics, all algorithms discovered less than 10% of possible mechanic combinations. Second failure: MAP-Elites(line) underperformed because its directional operator assumes "different elites in the archive have similar search space parameters" — an assumption violated in practice. Third failure: scenes maximizing sky tiles were not actually traversed by the evaluation agent, creating a mismatch between the optimized latent coordinate and the measured behavior. D9.0 implication: behavioral characterization mismatch (optimizing z for one proxy while φ measures something else) is a real failure mode. The DNP_BEHAVIOR_HASHLIKE gate must test *actual behavior correlation*, not just structural genome distance.

**Over-exploration in latent space optimization.** [direct empirical] The LES paper (OpenReview 2023, evaluated across 5 LSO benchmark tasks and 22 VAE models) documents that Bayesian Optimization in a VAE's continuous latent space "tends to over-explore the latent space, resulting in unrealistic or invalid molecules." The failure occurs when optimization ventures into regions deviating from the training distribution. D9.0 implication: a D9 random-scan over z without a valid-network-rate gate will reproduce this failure. DNP_VALIDITY_COLLAPSE (valid_network_rate < 0.99) is correctly positioned as the first gate.

**Novelty search beats objective-based search in deceptive tasks.** [direct empirical] Lehman and Stanley (Evolutionary Computation 19(2), 2011) demonstrated empirically in maze navigation and biped walking that novelty search significantly outperforms objective-based search when fitness is deceptive. However, Pugh, Soros, and Stanley (Frontiers in Robotics and AI 2016) extend this: QD algorithms "following the compass of an unaligned BC are incrementally less able to find QD as the level of deception increases." On hard maze tasks, DirectionBC-driven QD treatments "perform significantly worse than fitness" alone. D9.0 implication: D9's synthetic landscape tests (smooth, deceptive, multi-basin) are not ornamental — the deceptive basin test is the one most likely to distinguish a real compiler from a random hash decoder.

---

## Theoretical / Formal Results

**Multiple genotypes map to same behavior cell — inverse is underdetermined.** [theoretical] The QD literature accumulates an archive precisely because many distinct genotypes produce the same behavior descriptor. Pugh et al. (2016) frame this as the "unaligned BC" problem: the behavior characterization throws away structural information by construction. No paper in the QD literature attempts to recover genotype from BC; all proceed forward only. D9.0 implication: the Gemini digest's warning against "inverse mapping as D9.0 primary strategy" is correct and theoretically grounded. Exact inverse is only available when z and the rule trace were logged at compile time. For external genomes, only approximate search is possible and should be flagged as such.

**BOP-Elites: Bayesian GP surrogate is most sample-efficient scan for expensive QD.** [theoretical / direct empirical] Kent and Branke (arXiv:2307.09326) propose BOP-Elites, which models both fitness and behavioral descriptors with Gaussian Process surrogate models. The paper reports BOP-Elites "significantly outperforms other state-of-the-art algorithms without the need for problem-specific parameter tuning" when evaluations are expensive. D9.0 implication: for cheap toy compilers, simple lattice scan or random scan is sufficient. BOP-Elites (surrogate + acquisition function) is the right escalation path for D9.1+ when the compiled network requires real VRAXION evaluation.

---

## Analogies and Speculative Ideas

**Volz 2018 GAN-CMA-ES is the closest successful precedent.** [analogy] Volz et al. (arXiv:1805.00728, GECCO 2018) showed that CMA-ES operating in a 32-dimensional DCGAN latent space successfully generated Mario levels with target properties in most cases (ground-tile fraction optimization succeeded in all but one condition). The latent space was low-dimensional (32) and the decoder (DCGAN) was trained to produce valid structures. D9.0 analogy: if D9's z is low-dimensional (8-16 dimensions as hinted in the plan), and the compiler is rule-based (always producing valid graphs), the Volz result suggests search in this space should work. The critical difference: GAN decoders are learned and may have better locality than a hand-written rule compiler.

**Lattice scan vs. novelty-driven random — no direct QD comparison.** [speculative] Basin-hopping in the chemistry/physics literature (Wales and Doye, 1997) iterates perturbation + local minimization and outperforms uniform random in molecular optimization, but no paper directly compares basin-hopping to MAP-Elites or to novelty-driven random in a QD setting. This is not found after search.

---

## Open Questions / Contradictions

One contradiction: the LSI paper (Fontaine 2021) reports CMA-ME outperforms random search in 32-dimensional GAN latent space, while the LES paper (OpenReview 2023) reports Bayesian Optimization in VAE latent space suffers from over-exploration. Both are empirical. The resolution is likely task-dependent: GAN latent space with low-dim structured z and CMA-ME's covariance adaptation avoids over-exploration via directed search; Bayesian Optimization in high-dimensional or poorly structured VAE latent space over-explores because the surrogate uncertainty is too large. D9.0 should start with the CMA-ME / structured low-dim precedent, not the BO precedent.

Open question not resolved: whether CMA-ME improvement emitters can be re-seeded from z structural fields rather than archive occupants. No paper tests this.

---

## Failures of Similar Systems (Priority Section)

**Grammatical Evolution locality failure.** [direct empirical] GE is the closest conceptual precedent to D9: integer-coded genome → deterministic rule application → phenotype. Rothlauf and Oetzel (EuroGP 2006) found that GE's mapping has severely poor locality, with the large majority of genotypic neighbors failing to produce neighboring phenotypes. Search performance was measurably worse than high-locality alternatives. Root cause: the codon-to-rule mapping with modular wrapping breaks the smooth correspondence between genome neighborhood and phenotype neighborhood. D9.0 must not replicate the wrapping mechanism or any scheme where small z perturbations produce discontinuous rule selections.

**LSI on 8-dimensional binary metrics.** [direct empirical] Fontaine et al. (AAAI 2021) found all QD algorithms discovered less than 10% of possible mechanic combinations when behavior was measured in binary (present/absent mechanics). Root cause: the A* evaluation agent ignored mechanics irrelevant to speed, so behavior cells were filled by a biased subset of level types. D9.0 implication: if D9's behavior fingerprint φ is measured by a biased evaluator, DNP_BEHAVIOR_HASHLIKE will give a false pass. The evaluator must be checked for coverage bias before trusting scan efficiency results.

**CMA-ME unrealistic outputs at extreme behavior targets.** [direct empirical] In the LSI paper, CMA-ME generated images with high boldness values "that do not look realistic." The decoder (GAN) was being pushed out of its training distribution. D9.0 analogy: a rule-based compiler has no training distribution constraint, but it has structural validity constraints. Extreme z values may hit edge cases in rule application that produce structurally degenerate graphs. DNP_VALIDITY_COLLAPSE is the correct gate.

**Latent space over-exploration in Bayesian Optimization.** [direct empirical] LES (OpenReview 2023) documented across 22 VAE models that naive BO in latent space produces out-of-distribution solutions. This is the default failure mode of unconstrained latent search. D9.0 does not use BO or a VAE, but the same failure mode can appear as random z sampling at extreme values. The valid_network_rate gate directly addresses this.

**Volz CMA-ES numerical instability in GAN latent space.** [direct empirical] The Volz 2018 implementation notes numerical instability that can prevent convergence when searching GAN latent spaces with bounded input vectors. D9.0 implication: any scan strategy operating on a bounded z domain must handle boundary conditions explicitly; z clamping can introduce discontinuities that destabilize gradient-free optimizers.

---

## Background Only

The following items have no implementable D9.0 consequence and are recorded for completeness only.

MAP-Elites with Sliding Boundaries (MESB, arXiv:1904.10656) adapts cell boundaries based on population distribution. Useful for dynamic behavior characterizations but requires running MAP-Elites, which D9.0 does not yet do.

CMA-MAE (arXiv:2205.10752) addresses CMA-ME's three failure modes (premature abandonment of objective, flat-objective struggling, poor low-resolution archive performance). D9.1+ architecture planning material.

pyribs RIBS framework (Tjanaka et al. 2023) separates QD algorithms into archive, emitter, and scheduler components. Directly usable as a library for D9.1+ when VRAXION-lite evaluation is integrated. Not needed for D9.0 toy benchmark.

Novelty search with unsupervised behavior descriptors (Hierarchical Behavioral Repertoires, arXiv:1804.07127) learns descriptors from data. Out of scope for D9.0 (requires learned components).

Cellular Encoding (Gruau 1994) uses graph grammar with cell division operations. Related family to D9's proposed grammar, but no locality measurements published in the original work.

## Sources

- [MAP-Elites arXiv 1504.04909](https://arxiv.org/abs/1504.04909)
- [CMA-ME arXiv 1912.02400](https://arxiv.org/abs/1912.02400)
- [Illuminating Mario Scenes in GAN Latent Space arXiv 2007.05674](https://arxiv.org/abs/2007.05674)
- [Pugh Soros Stanley 2016 QD Frontiers](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)
- [On the Locality of Grammatical Evolution Springer](https://link.springer.com/chapter/10.1007/11729976_29)
- [Rothlauf Oetzel EuroGP 2006 bib entry](https://gpbib.cs.ucl.ac.uk/gp-html/eurogp06_RothlaufOetzel.html)
- [Volz 2018 Mario GAN-CMA-ES arXiv 1805.00728](https://arxiv.org/abs/1805.00728)
- [BOP-Elites arXiv 2307.09326](https://arxiv.org/abs/2307.09326)
- [LES over-exploration OpenReview](https://openreview.net/forum?id=0jm6gkAuYH)
- [MAP-Elites Sliding Boundaries arXiv 1904.10656](https://arxiv.org/abs/1904.10656)
- [CMA-MAE arXiv 2205.10752](https://arxiv.org/abs/2205.10752)
- [pyribs documentation](https://docs.pyribs.org/en/latest/index.html)
- [Salimans ES arXiv 1703.03864](https://arxiv.org/abs/1703.03864)
- [Lehman Stanley 2011 Abandoning Objectives](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf)
- [Hierarchical Behavioral Repertoires arXiv 1804.07127](https://arxiv.org/abs/1804.07127)
