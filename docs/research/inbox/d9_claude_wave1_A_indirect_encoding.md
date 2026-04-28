# D9 Wave 1 — Agent A: Indirect Encoding Literature

## Compared to Gemini digest

**Claim: "D9 should start with a deterministic rule/grammar compiler, not a learned graph decoder."**
PARTIALLY SUPPORTED but with a critical caveat the digest does not state. The Clune, Stanley, Pennock, Ofria 2011 paper (IEEE TEC) provides the strongest empirical evidence for deterministic/structured encodings — but the finding is conditional: indirect encodings with regularity-exploiting structure win on regular problems and fail on irregular ones. The digest presents the grammar-first recommendation as unconditional. Literature says the regime matters. [direct empirical]

**Claim: "z should be a structured recipe, not an opaque random hash."**
SUPPORTED. All indirect encoding literature (NEAT 2002, CPPN 2007, HyperNEAT, L-systems, cellular encoding) agrees on structure over random hash, but this is largely theoretical or based on task performance, not direct locality measurement. [theoretical + analogy]

**Claim: "Locality must be measured directly."**
STRONGLY SUPPORTED and under-examined in the primary literature. No EC indirect encoding paper found directly measures latent-space Spearman rank correlation between genome distance and behavior distance the way D9 proposes. The digest's framing is more rigorous than the literature it is implicitly citing. [direct empirical — gap in literature, see Open Questions]

**Claim that advanced graph grammar formalisms (nc-eNCE, subgraph isomorphism) are "too heavy for D9.0."**
SUPPORTED. The primary Gruau 1994 cellular encoding paper treats cellular encoding as a parallel graph grammar, but published locality tests are absent. The formalism has not yielded D9.0-implementable locality guarantees. [theoretical]

**"Rule-Based Grammar Over Neural Decoder" section of digest: regime dependency not stated.**
This is the primary weakness the literature exposes. The digest recommends starting deterministic without acknowledging that Kusner et al. 2017 (Grammar VAE, ICML) showed a learned decoder constrained by grammar achieves better latent-space coherence and higher validity than either a pure deterministic decoder or a pure neural decoder. The grammar VAE result partially contradicts the false dichotomy between "deterministic rule compiler" and "learned decoder." [direct empirical — contradicts Gemini framing]

---

## Direct empirical findings

**Finding 1: Indirect encoding wins on regular problems, fails on irregular ones. [direct empirical]**
Clune, Stanley, Pennock, Ofria. "On the Performance of Indirect Encoding Across the Continuum of Regularity." IEEE TEC, 2011. In three problem domains (target weights, bit mirroring, quadruped locomotion), HyperNEAT achieved near-optimal performance on fully regular problems. As irregularity increased, HyperNEAT degraded significantly. On the most irregular bit-mirroring task, median HyperNEAT fitness (302.98) was matched by HybrID (304.97) only after switching to a direct encoding layer. On target weights, HyperNEAT error was 0.2115 versus HybrID 0.0229. The paper explicitly states: "As the regularity of the problem decreases, the performance of the generative representation degrades to, and then underperforms, the direct encoding." This is a direct empirical refutation of unconditional grammar-first strategies.

D9.0 implication: The z-to-genome compiler must not be assumed to preserve locality simply because it is rule-based. If the target behavior space contains irregular basins, a pure grammar compiler will collapse locality. The D9.0 toy must include an irregular basin as a negative fitness landscape, not just smooth ones.

**Finding 2: Geometric arrangement matters for CPPN locality — but the effect is sensitive to coordinate layout. [direct empirical]**
Clune, Ofria, Pennock. "The Sensitivity of HyperNEAT to Different Geometric Representations of a Problem." 2009 (Michigan State University). Human-engineered geometric configurations outperformed randomized ones by 10–20% on quadruped gait evolution (p < 0.05). HyperNEAT still outperformed direct FT-NEAT even with random geometric placement (p < 0.001). This means the locality benefit of the CPPN is real but fragile — the encoding geometry must align with the problem geometry, or locality degrades without collapsing entirely.

D9.0 implication: The D9 z vector should encode structural fields that have a geometric interpretation. Fields like recurrent-loop-bias or inhibition-ratio should map monotonically to graph properties, not be packed into an opaque seed.

**Finding 3: Grammar-constrained learned decoder achieves high latent-space coherence and validity. [direct empirical]**
Kusner, Paige, Hernandez-Lobato. "Grammar Variational Autoencoder." ICML 2017. By constraining a VAE decoder to produce only valid parse trees from a context-free grammar, the paper showed that nearby z values decoded to similar discrete outputs and that validity rates exceeded those of vanilla VAE. Bayesian optimization on molecular synthesis improved over unstructured VAE baseline. This directly contradicts the Gemini digest's implicit framing that grammar compiler and learned decoder are mutually exclusive. A learned decoder that incorporates grammar structure achieved both high validity and latent coherence simultaneously.

D9.0 implication: Grammar VAE is explicitly deferred from D9.0 by the project rule (no learned decoder). However, this finding weakens the Gemini claim that grammar-first is inherently superior. The correct claim is that grammar structure is necessary for locality, whether implemented as a static compiler or a constrained learned decoder. D9.0 should not interpret its success as evidence that static grammars win in general.

**Finding 4: HyperNEAT on irregular tasks showed locality collapse even with 500 generations of optimization. [direct empirical]**
Helms, Clune et al. "Improving HybrID: How to Best Combine Indirect and Direct Encoding in Evolutionary Algorithms." PLoS ONE, 2017. The experiments showed that on irregular problems, "indirect encodings have difficulty generating phenotypes with irregular elements, which negatively impacts their performance on irregular problems." Genome compressibility (LZW proxy) was used as a regularity proxy instead of a direct genotype-phenotype locality metric. This confirms the locality collapse is real but also reveals a gap: no paper in this line of work directly measures latent-space Spearman correlation as a locality metric.

D9.0 implication: The D9 DNP_LOCALITY_COLLAPSE gate is correctly motivated. The HybrID empirical result shows this gate would correctly fire on irregular basins with a grammar-only compiler.

---

## Theoretical / formal results

**NEAT innovation numbers as deterministic alignment mechanism. [theoretical]**
Stanley and Miikkulainen. "Evolving Neural Networks Through Augmenting Topologies." Evolutionary Computation, 2002. Innovation numbers solve the competing conventions problem by assigning a globally unique historical marker to each structural gene. During crossover, genes with matching numbers align deterministically regardless of network size. This guarantees that genotype-level structural similarity is preserved through the alignment function.

D9.0 implication: The D9 genome provenance field (root seed, rule sequence, motif origin) is the static-compiler analog of NEAT innovation numbers. The log of rule applications is the alignment index. If two z values share a common prefix in rule sequence, their genomes are structurally related by construction. This is implementable as a required field in the toy genome JSON: `rule_trace: [list of (rule_id, application_site, subseed)]`. Without this trace, the D9 genome is NEAT without innovation numbers — alignment becomes guesswork.

**Cellular encoding as parallel graph grammar. [theoretical]**
Gruau. "Neural Network Synthesis Using Cellular Encoding and the Genetic Algorithm." PhD thesis, ENS Lyon, 1994. Cellular encoding was formally defined as a parallel graph grammar. Genomes are labeled trees; production rules rewrite nodes to subgraphs. This formalism gives deterministic, reproducible z-to-graph mapping.

D9.0 implication: The direct consequence is that D9.0's hand-written production rules can be formally justified as a subset of Gruau's cellular grammar. However, Gruau never published a locality measurement, so the formal structure does not guarantee locality. This is background unless combined with Finding 1 above: grammar structure is necessary but not sufficient.

**CPPN without local interaction still produces structured phenotypes. [theoretical]**
Stanley. "Compositional Pattern Producing Networks: A Novel Abstraction of Development." Genetic Programming and Evolvable Machines, 2007. CPPNs map coordinates to connection weights without local interaction: each weight is set independently as a function of endpoint coordinates. The paper argues this suffices to reproduce structural motifs usually attributed to local developmental rules.

D9.0 implication: In D9, the z vector plays the role of CPPN parameters, and the compiler plays the role of the CPPN evaluation. The finding implies that global-coordinate-based rules can achieve local-looking structure in the graph — but only if the z fields have geometric meaning. This supports Gemini's structured recipe recommendation, but the empirical locality claim from the paper is based on visual inspection of evolved images rather than a Spearman correlation measurement.

---

## Analogies and speculative ideas

[analogy] L-system bodies evolved by Hornby and Pollack (2001, GECCO) produced creatures with hundreds of parts through reuse of production rules, compared to prior methods capped at 50 parts. This suggests production-rule reuse amplifies coverage of the phenotype space from a compact genotype — analogous to what D9 wants from a grammar compiler. However, the Hornby/Pollack papers do not measure genome locality.

[analogy] The Junction Tree VAE (Jin, Barzilay, Jaakkola, ICML 2018) showed that a two-phase learned decoder — first generating a tree scaffold, then filling subgraph details — achieved smooth latent space and 100% chemical validity on molecule generation. The hierarchical structure is analogous to D9's proposal of module-seed then edge-seed. This supports the Gemini hierarchical seed hierarchy but in a learned, not rule-based, context.

[speculative] Novelty search (Lehman and Stanley, Evolutionary Computation, 2011) showed that abandoning the fitness objective and searching for behavioral novelty alone can outperform direct objective optimization in deceptive maze tasks. This suggests that strong locality to the fitness landscape may actually impede exploration. D9's own digest acknowledges this caveat ("strong locality is not always automatically good"), but the literature does not provide a quantitative tradeoff between locality strength and exploration efficiency for grammar-based decoders.

---

## Open questions / contradictions in the literature

**No EC indirect encoding paper directly measures Spearman(latent distance, phenotype distance).** The Clune 2011 paper uses task performance as the proxy. Helms 2017 uses LZW compressibility. The Gemini digest's DNP_LOCALITY_COLLAPSE gate requires Spearman z-distance vs genome-distance. This metric has no direct precedent in the indirect encoding literature reviewed. It is borrowed from molecular latent space literature (Grammar VAE, JT-VAE) rather than from neuroevolution. Whether the same metric transfers to arbitrary graph-structured genomes is not established.

**Contradiction: Grammar VAE (Kusner 2017) suggests learned decoder with grammar constraint achieves better locality than static deterministic compiler alone.** The Gemini digest treats these as a choice. The literature treats them as a spectrum. The correct empirical finding is that grammar structure is the load-bearing component, not whether the decoder is static or trained. This contradicts Gemini's framing but does not contradict the D9.0 rule prohibiting learned decoders — it simply means D9.0's static compiler is a corner case of a larger design space, not the uniquely correct choice.

**Regime dependency unresolved for D9:** whether the VRAXION behavior space resembles a regular or irregular problem is unknown. If the D8 cell atlas contains behavioral basins with low regularity (which the D8 scan delta results suggest is possible given the 73% ceiling finding later overturned), then a grammar-based D9 compiler may fail the DNP_LOCALITY_COLLAPSE gate on real data even if it passes on smooth toy basins.

---

## Background only (formal claims without D9.0 consequence)

Gruau's cellular encoding graph grammar formalism includes proof that the production system is a type-2 parallel graph grammar capable of generating any finite graph. This is theoretically interesting but has no D9.0 implementable consequence beyond justifying the choice of production rules as a grammar.

Graph grammar formalisms nc-eNCE and edNCE (referenced in Gemini digest) provide confluence and termination guarantees for certain rule classes. These guarantees are not needed for D9.0's toy compiler where all graphs are generated from scratch and there is no need for inverse parsing.

NEAT speciation and historical markings together are shown through ablation to be mutually necessary in the 2002 paper. Removing either degrades performance. This is a systems result about search efficiency, not a locality theorem. D9 is not evolving topologies, so this does not transfer directly.

## Sources

- [On the Performance of Indirect Encoding Across the Continuum of Regularity | IEEE](https://ieeexplore.ieee.org/document/5910671/)
- [Improving HybrID: How to best combine indirect and direct encoding | PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5363933/)
- [The Sensitivity of HyperNEAT to Different Geometric Representations | Academia.edu](https://www.academia.edu/327833)
- [Clune 2011 full PDF](http://jeffclune.com/publications/2011-CluneEtAl-IndirectEncodingAcrossRegularityContinuum-IEEE-TEC.pdf)
- [Grammar Variational Autoencoder | ICML 2017](https://proceedings.mlr.press/v70/kusner17a.html)
- [Grammar VAE arXiv](https://arxiv.org/abs/1703.01925)
- [ES-HyperNEAT Enhanced Hypercube Encoding | MIT Press](https://direct.mit.edu/artl/article/18/4/331/2720/An-Enhanced-Hypercube-Based-Encoding-for-Evolving)
- [HybrID: Hybridization of Indirect and Direct Encodings | Springer](https://link.springer.com/chapter/10.1007/978-3-642-21314-4_17)
- [NEAT 2002 Stanley Miikkulainen PDF](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [Gruau 1994 cellular encoding thesis](http://gpbib.cs.ucl.ac.uk/gp-html/Gruau_1994_thesis.html)
- [CPPN 2007 Stanley | Springer](https://link.springer.com/article/10.1007/s10710-007-9028-8)
- [Abandoning Objectives: Novelty Search | Lehman Stanley 2011](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf)
- [Junction Tree VAE | ICML 2018](https://proceedings.mlr.press/v80/jin18a/jin18a.pdf)
- [Hornby Pollack 2001 L-systems GECCO](http://www.demo.cs.brandeis.edu/papers/hornby_gecco01.pdf)
