<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Hypotheses

## What This Page Is

This page tracks the **live open-hypothesis layer** for VRAXION. It is for present-tense research questions that are still unresolved and still relevant to the current INSTNCT direction.

## Epistemic Boundary

Everything here is a hypothesis until it is backed by reproducible artifacts and either promoted into [Validated Findings](Validated-Findings) or shipped in code on `main`.

Do not use this page as architecture doctrine, a status dashboard, or a historical archive.

## Active Hypotheses

| Topic | Prediction | What would count as evidence | Status |
|---|---|---|---|
| Mixed 18-worker swarm | A mixed worker policy beats the current single English recipe candidate under matched budgets and stays reproducible across reruns. | Matched-budget sweep bundle, reruns, and a summary showing better plateau accuracy without breaking determinism. | Active |
| Voltage-aware schedule pressure | A schedule-pressure policy such as voltage medium leak, or a direct successor, beats the current `8`-tick decay-slot recipe on plateau accuracy instead of only producing a better peak. | Plateau comparison across matched seeds and budgets, plus artifact bundles showing the result survives confirmation runs. | Active |
| Compact learnable schedule control | A low-parameter learnable policy such as the 3-angle decision tree can match or beat the best fixed schedule without collapsing into unstable control behavior. | Repeated runs showing equal or better plateau accuracy than fixed winners, with stable learned angles and no overflow failures. | Active |
| Decay resample promotion | Single-neuron decay resample in `[0.01, 0.5]` remains better than local perturbation across reruns and deserves promotion into the current recipe. | Rerun bundle showing the resample policy wins across seeds or budgets and preserves differentiated per-neuron decay values. | Active |
| Low-theta / low-scale generalization | `INJ_SCALE=1.0` with low theta stays better when tested against today’s stronger English recipe stack, not only the older baseline. | A fresh A/B on the current recipe candidate showing the low-scale setting still wins under matched budgets and gates. | Active |

## Record Format

Every hypothesis record should be specific enough to fail cleanly.

- `Prediction`: what should improve, remain stable, or generalize
- `Evidence`: the minimum artifact bundle needed to call the result meaningful
- `Status`: active, parked, supported, or retired

If a question cannot be phrased as a falsifiable prediction with evidence requirements, it does not belong here yet.

## Promotion Rule

Promotion path:

- `Hypotheses` -> [Validated Findings](Validated-Findings) when a result is reproducible and useful
- `Validated Findings` -> `Current mainline` only when the shipped code on `main` actually adopts it

Historical theory ledgers and superseded hypothesis catalogs belong in the archive layer, not here.

## Read Next

- [Home](Home)
- [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture)
- [Validated Findings](Validated-Findings)
- [Engineering Protocol](Engineering)
