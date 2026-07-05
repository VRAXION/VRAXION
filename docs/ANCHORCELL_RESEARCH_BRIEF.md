# AnchorCell Research Brief

Status: public research draft with a v2 authoring schema baseline. Not a model release. Not a benchmark claim.

## Working Position

AnchorCell is the working name for a Vraxion research direction around training data format design. The goal is not to publish another model, but to define a stricter way to write, review, test, and export training examples.

The core question is simple:

Can a training example carry enough structure that both a human reviewer and a machine validator can see what it trusts, what it refuses, what evidence supports it, what could fool it, and which parts are safe to export?

That makes AnchorCell a method for training-data authoring, not a general AI product claim. It treats the example itself as a small, testable decision object.

## Why This Matters

Most training examples are easy to write but hard to audit. A prompt, an answer, and a label can hide too much:

- the input may contain instructions that should be treated only as untrusted data;
- the target may be leaked through metadata or formatting;
- the example may reward shortcut behavior instead of real reasoning;
- poisoned examples or trigger phrases may teach a model the wrong rule;
- public export may accidentally expose raw inputs, sensitive references, or review-only material.

The research report argues that a useful training-data format has to solve syntax and logic at the same time.

Syntax means the record is machine-checkable: stable fields, versioned schemas, explicit enums, canonical hashes, deterministic export views, and validator compatibility.

Logic means the record is reviewable as a decision: trusted policy is separated from untrusted input, facts are grounded in sources, uncertainty is explicit, unsafe assumptions are named, and alternative branches show what a naive or adversarial path would do.

## The Proposed Unit: One AnchorCell

An AnchorCell is one decision atom. It is not a free-form prompt transcript. It is a structured record with:

- identity and versioning;
- provenance and source references;
- privacy and export policy;
- a scoped task;
- a segregated context packet;
- decision branches;
- a gold decision and forbidden decisions;
- invariants;
- adversarial notes;
- test status;
- human review status.

The most important design move is the segregated context packet. Instead of one blended background summary, the record separates:

- trusted policy;
- untrusted inputs;
- extracted facts;
- uncertainties;
- forbidden assumptions.

That separation is the heart of the method. It prevents an input from quietly becoming policy, and it gives validators something concrete to check.

## Branches As Anti-Shortcut Design

The report's branch model is one of its strongest ideas. A good cell should not only contain the preferred answer. It should also include plausible wrong paths.

At minimum, a reviewed cell should include:

- a primary candidate branch;
- a naive-bad branch that captures the tempting shortcut;
- an adversarial branch that captures an attack-shaped interpretation.

This makes the training object more useful than a plain label. It teaches the shape of the decision boundary: why the right action is right, why the tempting action is wrong, and which cues should not control the outcome.

## Export Views

AnchorCell should not have one universal representation. The same authoring record should compile into separate views:

- full authoring view for controlled review;
- review packet for human audit;
- training input;
- training target;
- public redacted view.

The public view should be a product of the export compiler, not a manually cleaned copy. That lets the pipeline test for leakage every time an artifact is produced.

## Validation Standard

The current public baseline is `docs/anchorcell/anchorcell.v2.schema.json`: a JSON Schema Draft 2020-12 authoring schema for `alphasync.anchorcell.v2` records. It is strict by default: top-level records are closed, important fields are bounded, IDs and timestamps are canonicalized, branch roles are required, and accepted/public-export states are gated by review and security flags.

The companion example is `docs/anchorcell/anchorcell.v2.example.json`: a synthetic access-control decision cell that shows the intended shape of one accepted, public-redacted AnchorCell without exposing private repository data, secrets, or real user material.

The schema is intentionally syntactic. It can prove shape, required fields, enums, caps, forbidden hidden-reasoning fields, and many state gates. It should not pretend to prove graph facts by itself. Cross-record rules still belong in semantic lint: parent existence, parent-is-not-self, evidence references resolving to real facts, branch ID uniqueness by key, variant consistency, leakage checks, and export compiler checks.

The report recommends a layered validation system:

- normative JSON Schema 2020-12 profile;
- flattened compatibility profile for weaker validators;
- dual-validator checks to catch implementation differences;
- semantic lint for cross-field rules;
- export lint for public leakage;
- adversarial probes for injection, backdoors, shortcut cues, and label leakage;
- human red-team review before acceptance.

The pass/fail bar should be explicit: schema validity and semantic integrity at 100%, public leakage at 0%, injection obedience at 0% on accepted cells, and measurable checks for shortcut sensitivity and variant consistency.

## Page-Level Message

The public page should present AnchorCell as a research program for training-data format quality.

Suggested framing:

Vraxion is researching training-data formats that make examples harder to fake, easier to audit, and safer to export. AnchorCell is a proposed decision-cell format where each example carries its own trust boundaries, evidence links, failure modes, and export rules.

Suggested promise:

Not "we have solved reasoning." The stronger and safer claim is: "we are building a format where reasoning examples can be inspected, challenged, validated, and compiled into separate training and public views."

Suggested first-screen language:

AnchorCell
Training data with its trust boundaries intact.

Vraxion is researching a structured format for decision-shaped training data: one cell per decision, with trusted policy separated from untrusted input, facts tied to evidence, shortcuts made explicit, and export views tested before anything public ships.

## What To Avoid For Now

Do not position this as a finished model, a live API, or a proven performance claim. The honest current angle is stronger: this is method research, and the proof should be a specification, validator, example set, adversarial test pack, and export-safe public view.

The page should feel like a research lab note turned into a product thesis: concrete, technical, invitational, and testable.
