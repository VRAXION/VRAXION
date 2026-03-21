<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Release Notes

## What This Page Is

This page is the **live primary public timeline page** for VRAXION. It combines current public status, major architecture turns, curated historical milestones, and the open questions that still matter for promotion.

## How To Read It

- Use this page to understand what changed, when it changed, and what now counts as current public truth.
- Use [Validated Findings](Validated-Findings) for experiment-backed evidence.
- Use [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture) for the active technical line.
- Use shipped code on `main` as the final source of truth for anything described as current mainline.

## Current Public Status

- Current mainline code path: [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py)
- Current English recipe candidate on `main`: [`v4.2/english_1024n_18w.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/english_1024n_18w.py)
- Current strongest schedule result in the public evidence layer: voltage medium leak at `22.11%` peak / `21.46%` plateau
- Current strongest compact learnable schedule result: the 3-angle decision-tree schedule at `20.05%`
- Current next public build target: mixed 18-worker swarm, with schedule and mutation policy still under active evaluation

## What Matters Now

- The active public architecture line is [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture); earlier architecture lines are historical context only.
- The strongest schedule and mutation results live in [Validated Findings](Validated-Findings) until code on `main` actually adopts them.
- The current public docs stack is now repo-tracked and mirrored into the GitHub wiki, which means historical pages can be retired without breaking backlinks.
- The main unresolved public targets are mixed 18-worker swarm evaluation, schedule-policy promotion, and re-checking low-theta / low-scale against today’s stronger recipe stack.

## Timeline of Major Turns

| Date | Turn | Why it mattered | Where to go now |
|---|---|---|---|
| Early 2026 | Diamond Code / LCX era became the main active line | External memory, dreaming, observability, and Goldilocks-era experiments dominated the architecture story. | Historical context only: [Diamond Code v3 Architecture](Diamond-Code-v3-Architecture) |
| 2026-02-17 | Governance and Engineering doctrine were split out into dedicated pages | Public structure stopped depending on scattered theory leaves and lab notes. | [Engineering Protocol](Engineering), [Documentation Governance](Governance) |
| 2026-02-17 | Older architecture leaves and terminology drift started getting cleaned up | Current terminology and architecture reading no longer needed a chain of legacy architecture pages. | [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture), this page |
| 2026-02-17 to 2026-02-22 | Evidence discipline hardened around findings instead of page-local claims | Schedule, depth, and mutation results moved into the evidence layer instead of being implied as defaults. | [Validated Findings](Validated-Findings) |
| 2026-03-21 | Repo-tracked docs became the canonical public source and the GitHub wiki became a mirror | Public truth stopped drifting between README, Pages, findings, and wiki surfaces. | [Home](Home), [Documentation Governance](Governance) |
| 2026-03-21 | Vision, roadmap, theory, and PF-era pages were retired into historical stubs | The public story became readable through a small number of current pages instead of a large theory-led wiki. | [Home](Home), [Validated Findings](Validated-Findings), [Engineering Protocol](Engineering) |
| 2026-03-21 | `Hypotheses`, `Theory of Thought`, and `Roadmap` were folded into the current timeline/status stack | Open questions and current public progress now live in one place instead of being split across multiple legacy trackers. | This page |
| 2026-03-21 | `Diamond Code v3 Architecture` and `Legacy Vault` were retired from the active browsing path | Legacy architecture and archive functions remain backlink-safe, but no longer shape the active public stack. | This page for timeline, stubs for historical links |

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

## Key Terms

**Current mainline**
What is actually shipped in code on `main`. If a public page and the code disagree about current behavior, the code wins.

**Validated finding**
An experiment-backed result that has not yet been promoted into the canonical code path.

**Experimental branch**
An active build target or design direction that is still under evaluation, not a live default.

**INSTNCT**
The current public architecture line for VRAXION: a self-wiring, gradient-free system where the learnable object is the evolving directed graph and its co-evolved neuron parameters.

**Artifacts (run bundle)**
The minimum files required to treat a run as evidence: `run_cmd.txt`, `env.json`, `metrics.json`, and `summary.md`.

**Fail gates**
The hard invalidation conditions for a run, such as OOM, NaN/Inf, step-time explosion, heartbeat stall, or VRAM guard breach.

**Engineering Protocol**
The run and evidence contract for VRAXION experiments: how runs are executed, validated, and promoted into public truth. See [Engineering Protocol](Engineering).

### Legacy Terms

**LCX (Latent Context Exchange)**
An earlier external-memory system from the Diamond Code v3 era, not part of the current INSTNCT public line.

**Zoom gate / bottleneck projection / C19 activation / score margin**
Historical Diamond Code v3 terms that remain useful only as legacy context. Use the retired [Diamond Code v3 Architecture](Diamond-Code-v3-Architecture) page for era-specific references.

## Read Next

- [Home](Home)
- [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture)
- [Validated Findings](Validated-Findings)
- [Engineering Protocol](Engineering)
