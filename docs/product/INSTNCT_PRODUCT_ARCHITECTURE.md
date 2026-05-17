# INSTNCT Product Architecture

Status: 056 productization planning artifact.

This document defines the first product boundary for the INSTNCT research stack
after the 049/050 adversarial frozen-eval package and the 052-055 visual stack.
It is architecture planning, not a release announcement.

## What Is Sellable Now

The sellable asset today is not a production autonomous model. It is a bounded
research and evaluation package with enough structure to support paid research
integration, private evaluation, and early enterprise technical discovery.

Sellable now:

- Research artifact package around bounded route-grammar behavior.
- Evaluation and anti-collapse methodology.
- Reproducibility and paper-audit bundle.
- Visual Lab V1 for audit, demo, replay, and debugging.
- Engineering services for private research integration and evaluation.
- Commercial license negotiation for source-available usage rights.

Not sellable as a claim today:

- Production clinical system.
- School high-stakes assessment system.
- General language-grounded assistant.
- Full VRAXION system.
- Autonomous consciousness or biological equivalence.
- Production default training component.

## Product Layers

### INSTNCT Core

INSTNCT Core is the research engine layer. It contains the experimental
route-grammar, training/search, checkpoint, evaluation, rollback, and visual
export mechanisms.

Initial product posture:

- Research-only core.
- Source-available under a noncommercial/public-benefit license.
- Commercial use requires a separate written license.
- No production API stability guarantee until 057/061 gates complete.

### INSTNCT SDK

The SDK is the future integration boundary. It is not released in 056. The SDK
candidate should expose only stable, narrow operations:

- Train bounded fixture or customer-approved datasets.
- Infer through an approved model/checkpoint.
- Evaluate heldout, OOD, regression, and collapse gates.
- Save/load checkpoint.
- Roll back checkpoint.
- Export visual snapshots.
- Produce audit reports.

### INSTNCT Visual Lab

Visual Lab is the audit and demo surface. V1 can replay schema-first visual
bundles and show topology, playback, diff, and metrics views.

Product role:

- Debugging and audit for internal teams.
- Reviewer/reproduction walkthrough.
- Enterprise sales demo with explicit research boundary.
- Not a production monitoring dashboard.

### Domain Wrappers

Domain wrappers are future packaging layers around Core/SDK/Visual Lab:

- Research edition.
- Enterprise edition.
- Hospital edition.
- School edition.

The wrappers must restrict claims and features by domain. Hospital and school
wrappers cannot ship high-stakes functions without separate compliance work.

## Product Boundary

056 defines the product shape but does not ship a product. The first commercial
offer should be framed as:

```text
INSTNCT Research Evaluation and Visual Audit Package
```

The offer can include source access, private evaluation support, visual audit
setup, and commercial licensing discussions. It must not promise production
automation or regulated use readiness.

## Required Before Release Candidate

- SDK/API release candidate with documented stability rules.
- License package reviewed by counsel.
- Security and privacy posture.
- Deployment harness.
- Regression and audit gates.
- Domain-specific acceptable-use gates.
- Pilot contracts that preserve claim boundaries.

## Claim Boundary

056 supports productization planning only. It does not support production
release, public beta promotion, production API readiness, clinical readiness,
school high-stakes readiness, full VRAXION, language grounding, consciousness,
biological/FlyWire equivalence, or physical quantum behavior.

