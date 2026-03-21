<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Project Timeline

## What This Page Is

This page is the **single primary public timeline and lookup surface** for VRAXION. It combines the current snapshot, the major architecture turns, the retirement map for older public pages, the open questions that still block promotion, and the small set of terms needed to read the rest of the stack.

## How To Read It

- Use `Current Snapshot` when the question is what is live right now.
- Use `Project Timeline` when the question is what changed, when it changed, and why it mattered.
- Use `Retired Surfaces and Replacements` when an older wiki page name appears in links, issue history, or screenshots.
- Use `Open Questions and Promotion Gates` for the unresolved lines that still need proof before promotion.
- Use [Validated Findings](Validated-Findings) for experiment-backed evidence.
- Use [INSTNCT Architecture](SWG-v4.2-Architecture) for the active technical line.
- Use shipped code on `main` as the final source of truth for anything described as current mainline.
- Older retired wiki URLs are no longer maintained as live lookup surfaces; this page replaces that role.

## Current Snapshot

- Current mainline code path: [`v4.2/model/graph.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/model/graph.py)
- Current English recipe candidate on `main`: [`v4.2/english_1024n_18w.py`](https://github.com/VRAXION/VRAXION/blob/main/v4.2/english_1024n_18w.py)
- Current strongest schedule result in the public evidence layer: voltage medium leak at `22.11%` peak / `21.46%` plateau
- Current strongest compact learnable schedule result: the 3-angle decision-tree schedule at `20.05%`
- Current next public build target: mixed 18-worker swarm, with schedule and mutation policy still under active evaluation

## What Matters Now

- The active public architecture line is [INSTNCT Architecture](SWG-v4.2-Architecture); earlier architecture lines are historical context only.
- The strongest schedule and mutation results live in [Validated Findings](Validated-Findings) until code on `main` actually adopts them.
- The current public docs stack is repo-tracked and mirrored into the GitHub wiki, which makes the live public story readable through a small number of active pages.
- The main unresolved public targets are mixed 18-worker swarm evaluation, schedule-policy promotion, and re-checking low-theta / low-scale against today’s stronger recipe stack.
- History, terminology, and page retirements now live here instead of being spread across separate glossary, roadmap, theory, and archive leaves.

## Project Timeline

| Date | Turn | Why it mattered | Where to go now |
|---|---|---|---|
| Early 2026 | Diamond Code / LCX experiments dominated the public architecture story | External memory, dreaming, observability, and Goldilocks-style probes shaped the earlier line before INSTNCT became the clean active center. | Use this page for chronology and [INSTNCT Architecture](SWG-v4.2-Architecture) for the current line |
| 2026-02-17 | Engineering doctrine and contributor-facing docs policy were split into stable homes | Public structure stopped depending on scattered theory leaves and lab notes for reproducibility or provenance rules. | [Engineering Protocol](Engineering), [`CONTRIBUTING.md`](https://github.com/VRAXION/VRAXION/blob/main/CONTRIBUTING.md) |
| 2026-02-17 | Terminology and architecture drift started getting cleaned out of the front door | Readers no longer needed a chain of legacy architecture leaves just to understand the current system. | [VRAXION Home](Home), [INSTNCT Architecture](SWG-v4.2-Architecture) |
| 2026-02-17 to 2026-02-22 | Evidence discipline hardened around findings instead of page-local claims | Schedule, depth, and mutation results moved into an explicit evidence layer instead of being implied as shipped defaults. | [Validated Findings](Validated-Findings) |
| 2026-03-21 | Repo-tracked docs became the canonical public source and the GitHub wiki became a mirror | Public truth stopped drifting between README, Pages, findings, and ad hoc wiki edits. | [VRAXION Home](Home), [`CONTRIBUTING.md`](https://github.com/VRAXION/VRAXION/blob/main/CONTRIBUTING.md) |
| 2026-03-21 | Schedule-control work became the main live research frontier | `8` ticks, decay-aware scheduling, voltage/leak control, and compact learnable policies became the strongest present-tense recipe questions. | [Validated Findings](Validated-Findings), this page |
| 2026-03-21 | Roadmap, hypotheses, theory, archive, glossary, and old architecture leaves were collapsed into one timeline surface | History, terminology, open questions, and retirement lookup now live in one place instead of seven smaller pages. | This page |

## Retired Surfaces and Replacements

| Retired surface | What it was | Why it was retired | Replacement |
|---|---|---|---|
| `Glossary` | Standalone terminology page | The useful live terms were short enough to live inline with the timeline and status surface. | `Key Terms` on this page |
| `Legacy Vault` | Archive index / historical directory | The archive index role was more useful as a curated timeline than as a standalone vault page. | `Project Timeline` and this table |
| `Hypotheses` | Open-question tracker | Active public open questions were already coupled to current status and promotion gates. | `Open Questions and Promotion Gates` on this page |
| `Theory of Thought` | Legacy theory / hypothesis ledger | It no longer described the active public architecture or the live open-question surface. | [VRAXION Home](Home), [INSTNCT Architecture](SWG-v4.2-Architecture), and this page |
| `Chapter 11 - Roadmap` | Public roadmap/status page | The project needed one timeline page, not a separate roadmap surface. | This page |
| `Diamond Code v3 Architecture` | Earlier architecture hub | It no longer described the active VRAXION architecture line. | [INSTNCT Architecture](SWG-v4.2-Architecture) and this page |
| `Proven Findings` | Earlier evidence hub | The active evidence layer now lives in one explicit findings page for the INSTNCT-era stack. | [Validated Findings](Validated-Findings) |

## Open Questions and Promotion Gates

| Topic | What still must be proven | What promotion would mean | Current status |
|---|---|---|---|
| Mixed 18-worker swarm | Show matched-budget reruns that beat the current single English recipe candidate on plateau accuracy without breaking reproducibility. | Promote the mixed swarm line from open target to validated finding, and make it the next serious recipe-update candidate. | Active |
| Voltage-aware schedule pressure | Show that a voltage-style schedule policy wins on plateau accuracy under confirmation reruns, not only on isolated peaks. | Promote the policy from interesting schedule evidence to a stronger recipe candidate. | Active |
| Compact learnable schedule control | Show that a low-parameter learnable controller, such as the 3-angle tree, can match or beat the best fixed schedules without unstable drift or overflow. | Promote the controller from exploratory mechanism to validated schedule candidate. | Active |
| Decay resample promotion | Show that single-neuron decay resample in `[0.01, 0.5]` keeps winning over local perturbation across reruns and budgets. | Promote the resample mutation policy into the current recipe line. | Active |
| Low-theta / low-scale generalization | Re-run `INJ_SCALE=1.0` with low theta against the stronger current English recipe stack instead of the older baseline only. | Promote the low-scale line from older validated evidence into the current recipe discussion. | Active |

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
An earlier external-memory system from the Diamond Code era, not part of the current INSTNCT public line.

**Zoom gate / bottleneck projection / C19 activation / score margin**
Historical Diamond Code-era terms that still appear in older issues, notes, or screenshots. Treat them as legacy context, not as current INSTNCT defaults.

## Published Releases

- GitHub Releases: [VRAXION releases](https://github.com/VRAXION/VRAXION/releases)
- Public update issues: [public-update label](https://github.com/VRAXION/VRAXION/issues?q=label%3Apublic-update)

Older sprint bundles, prerelease notes, and retired page histories are now summarized through the timeline above instead of being maintained as separate public lookup pages.

## Read Next

- [VRAXION Home](Home)
- [INSTNCT Architecture](SWG-v4.2-Architecture)
- [Validated Findings](Validated-Findings)
- [Engineering Protocol](Engineering)
