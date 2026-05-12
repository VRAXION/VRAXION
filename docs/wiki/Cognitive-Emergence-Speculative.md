# Speculative Extension — Cognitive Emergence

> **⚠️ This page is a speculative extension of the [Local Constructability Framework](Local-Constructability-Framework), not a paper-supported claim.** The framework's empirical claims live in the three sub-documents (Interference Dynamics, Mutation-Selection Dynamics, Constructed Computation). This page records research-direction notes — what the framework *might* say about cognition and consciousness if certain experimental gaps are closed. Nothing here has been validated experimentally on the current setup.

---

## What is being speculated

If the two mechanisms of the framework — destructive interference at the signal level and mutation-selection construction at the structure level — generalise beyond the byte-pair prediction setting, one might ask whether they could underlie *cognition* (in a stronger sense than 397-class classification accuracy) or even *consciousness* (in any of the existing technical senses).

The framework as currently evidenced supports neither claim. Below, we list which claims would need to be made and what evidence would be required for each.

---

## Cognition (potentially testable)

The shorter-distance speculation. Cognition here means "behaviour that generalises across tasks, supports compositional structure, and is not collapsible to memorisation".

What the framework *would* claim, if extended: cognition arises in a network when (i) interference dynamics are tuned to preserve task-distinguishing signal across the substrate, (ii) the mutation schedule has built a topology that supports it, and (iii) the same substrate can be re-tuned for a related task without re-construction from scratch.

What evidence is missing:

- **Multi-task validation.** Phase A and Phase B are single-task (397-class byte-pair). At minimum, two more tasks (e.g. a classification task on a different alphabet, a sequence-completion task) under the same architecture before any "cognition" claim.
- **Transfer.** Take a network trained on task A. Re-tune on task B with a budget smaller than rebuild-from-scratch. If C_K on task B starts higher than from random init, the substrate has internal structure that transfers. This is the operational test for "cognition".
- **Compositional generalisation.** Train on simple inputs; test on compositions of those inputs not in the training distribution. Performance retention (above-chance) indicates compositional structure.

The framework's mechanisms permit each of these tests; the experiments have not been run.

---

## Grounded self-controller toy result (bounded)

The [Grounded Modular Self-Controller Finding](Grounded-Modular-Self-Controller)
is relevant to this page because it tests a minimal self-anchor/controller
mechanism:

```text
semantic event
-> inferred grounding mode
-> hard committed self-state
-> learned controller over frozen modules
```

The result is positive inside a controlled toy setup: the integrated controller
solves hard-counterfactual action choices, no-commit/static baselines fail, and
frozen primitive modules remain intact while a shared end-to-end baseline shows
primitive drift.

This does **not** change the consciousness boundary. The finding is evidence for
a small mechanism shape: grounded events can update a committed internal state
that later controls protected action/skill routing. It is not evidence for
phenomenal consciousness, biological validity, natural-language understanding,
production readiness, or full VRAXION behavior.

---

## Consciousness (currently untestable on this setup)

The longer-distance speculation. The framework explicitly does *not* claim consciousness. The word appears here only to draw a clear boundary with the discussion of established consciousness theories, which can be loosely related to the framework but not derived from it.

Loose alignments with existing theories:

| Theory | Alignment | Strength of alignment |
|---|---|---|
| **Integrated Information (Tononi, IIT)** | The framework's "anti-collapse index" rewards partition-resistant computation, similar in spirit to Φ. | Verbal only; no Φ has been computed on our networks (intractable at H=128–400). |
| **Global Workspace (Baars, Dehaene)** | The framework's mutual-inhibition seed creates winner-take-all output dynamics, surface-similar to GW broadcast. | Surface-similar; we lack reentrant top-down dynamics that GW theories require. |
| **Predictive Processing (Friston, Clark)** | The framework's bigram cosine fitness rewards prediction-of-next-token. | Conceptual analogy only; no hierarchical generative model is present. |
| **Recurrent Cortical Reentry (Edelman)** | Recurrent topology + selection over time. | Closest topology match; selectional dynamics are different (Phase B mutation-selection vs Edelman's neural Darwinism on synaptic populations). |

What evidence would be required: any operational test of any of these consciousness theories at our scale produces noise rather than signal. Current consensus in the field is that these theories cannot be empirically distinguished on networks at our complexity. Until that consensus shifts, the framework does not engage with these claims.

---

## Boundary the framework draws

- **Cognition is, in principle, testable extension** of the framework. The required experiments are listed; they have not been done.
- **Consciousness is not a paper claim** of the framework. References here are taxonomic, not predictive.
- The framework should not be cited as support for any cognitive or conscious claim about systems beyond its immediate experimental scope (gradient-free mutation-selection on sparse spiking nets, byte-pair prediction).

---

## Read next

- [Local Constructability Framework](Local-Constructability-Framework) — umbrella; what is supported.
- [Constructed Computation](Constructed-Computation) — what is empirically measured at the emergence level.
- [Research Process & Archive](Timeline-Archive) — historical claims (some of which were stronger than current evidence supports).
