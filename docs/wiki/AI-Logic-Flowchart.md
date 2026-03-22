<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# AI Logic Flowchart

This is a draft side page for the full logical flowchart of the current VRAXION AI line. It stays out of the primary nav until the visual system is stable enough to serve as a public explainer instead of a moving sketchpad.

## What This Page Is

Use this page to collect the diagram system for how the current INSTNCT line works end to end: fixed interface, hidden graph execution, persistent state, mutation-selection training, and the promotion path from experiment to shipped public truth.

The page is intentionally scaffold-first right now. The exported flowchart panels will be added here gradually as they harden.

## Current Scope

- runtime execution flow
- mutation-selection training flow
- evidence and promotion flow
- final stitched full-system logic map

## Planned Diagram Stack

### 1. Runtime Execution Flow

This panel will explain the live path from input signal to output behavior:

- input or world signal
- `input_projection`
- self-wiring hidden graph
- persistent internal state across ticks
- `output_projection`
- output or behavior

### 2. Mutation-Selection Training Flow

This panel will explain how the structure changes:

- propose mutation
- evaluate
- keep or revert
- co-evolved thresholds and decay
- repeat across accepted and rejected edits

### 3. Evidence and Promotion Flow

This panel will explain how results move through the public stack:

- experimental branch
- validated finding
- candidate recipe or implementation target
- current mainline
- public release surface

### 4. Full System Logic Map

The final stitched view should connect the runtime path, the training loop, and the promotion/evidence path in one readable system map without collapsing into a wall of arrows.

## Asset Plan

The working method for this page is:

1. sketch the logic in chat or draw.io
2. export a clean image asset
3. process it into a stable repo asset path
4. place it here with a short lead-in
5. keep only the strongest panels once the full map stabilizes

## Read Next

- [VRAXION Home](Home)
- [INSTNCT Architecture](INSTNCT-Architecture)
- [Validated Findings](Validated-Findings)
- [Engineering Protocol](Engineering)
- [Project Timeline](Release-Notes)
