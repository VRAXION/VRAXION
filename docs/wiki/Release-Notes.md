<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Release Notes

## What This Page Is

This page is the **live secondary public status page** for VRAXION. It tracks milestone-level progress, current public targets, and shipped or publicly promoted bundles.

## How To Read It

- Use this page for current status and public progress.
- Use [Validated Findings](Validated-Findings) for experiment-backed evidence.
- Use [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture) for the active technical line.
- Use shipped code on `main` as the final source of truth for anything described as current mainline.

## Current Public Status

- Current mainline code path: [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py)
- Current English recipe candidate on `main`: [`v4.2/english_1024n_18w.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/english_1024n_18w.py)
- Current strongest schedule result in the public evidence layer: voltage medium leak at `22.11%` peak / `21.46%` plateau
- Current strongest compact learnable schedule result: the 3-angle decision-tree schedule at `20.05%`
- Current next public build target: mixed 18-worker swarm, with schedule and mutation policy still under active evaluation

## Recent Milestones

| Milestone | Outcome | Public status |
|---|---|---|
| Public surface hardening | Repo-tracked docs now drive the wiki mirror, and the public stack is split cleanly into Home, Architecture, Findings, Engineering, and status/history surfaces. | Complete |
| INSTNCT naming and public alignment | The public architecture line is now consistently presented as `INSTNCT`, while legacy names stay historical. | Complete |
| Mainline English recipe candidate refresh | The current English recipe candidate on `main` now reflects `8` ticks and the current schedule/mutation line. | Complete |
| Schedule-policy exploration | Voltage medium leak, decay resample, and decision-tree schedule policies are now in the validated-finding layer, not mislabeled as shipped defaults. | In progress |

## Current Next Targets

- Confirm which schedule policy should graduate from validated finding to the next promoted recipe update.
- Resolve whether low-theta / low-scale still wins against the stronger current English recipe stack.
- Continue mixed 18-worker swarm evaluation as the main active training target.

## Open Questions and Promotion Gates

| Topic | What still must be proven | What promotion would mean | Current status |
|---|---|---|---|
| Mixed 18-worker swarm | Show matched-budget reruns that beat the current single English recipe candidate on plateau accuracy without breaking reproducibility. | Promote the mixed swarm line from open target to validated finding, and make it the next serious recipe-update candidate. | Active |
| Voltage-aware schedule pressure | Show that a voltage-style schedule policy wins on plateau accuracy under confirmation reruns, not only on isolated peaks. | Promote the policy from interesting schedule evidence to a stronger recipe candidate. | Active |
| Compact learnable schedule control | Show that a low-parameter learnable controller, such as the 3-angle tree, can match or beat the best fixed schedules without unstable drift or overflow. | Promote the controller from exploratory mechanism to validated schedule candidate. | Active |
| Decay resample promotion | Show that single-neuron decay resample in `[0.01, 0.5]` keeps winning over local perturbation across reruns and budgets. | Promote the resample mutation policy into the current recipe line. | Active |
| Low-theta / low-scale generalization | Re-run `INJ_SCALE=1.0` with low theta against the stronger current English recipe stack instead of the older baseline only. | Promote the low-scale line from older validated evidence into the current recipe discussion. | Active |

## Published Releases

- GitHub Releases: [VRAXION releases](https://github.com/VRAXION/VRAXION/releases)
- Public update issues: [public-update label](https://github.com/VRAXION/VRAXION/issues?q=label%3Apublic-update)

Older sprint-by-sprint bundles and prerelease-era notes remain part of the historical record, but they are no longer the primary status surface.

## Read Next

- [Home](Home)
- [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture)
- [Validated Findings](Validated-Findings)
- [Engineering Protocol](Engineering)
