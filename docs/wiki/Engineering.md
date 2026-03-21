<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Engineering Protocol

This page defines how VRAXION runs experiments, validates outcomes, and decides what counts as evidence. It is a **primary public page**. The wiki is a mirrored secondary surface; repo-tracked docs remain canonical.

Documentation Governance covers canonical docs, provenance, and page maintenance. This page covers the run contract and evidence discipline behind VRAXION results.

## What This Page Is

VRAXION treats engineering as a contract problem, not a vibes problem. A run is only useful as public evidence if it can be described clearly, checked against fail gates, and tied to a reproducible artifact bundle.

## Run Contract

Every meaningful run must declare:

- an objective metric
- one budget mode
- hard fail gates
- a required artifact bundle

Results outside that contract can still be useful probes, but they should not be promoted as canonical evidence.

## Required Evidence

| Artifact | Purpose |
|---|---|
| `run_cmd.txt` | Exact command and flags used for the run |
| `env.json` | Environment snapshot: OS, GPU/runtime, Python, package versions |
| `metrics.json` | Time series and summary metrics for the run |
| `summary.md` | Human verdict, including PASS/FAIL and the reason |

Optional extras such as checkpoints, plots, CSV exports, or live logs are useful, but they do not replace the core evidence bundle.

## Fail Gates

| Gate | Trigger |
|---|---|
| OOM / runtime failure | Any out-of-memory or driver/runtime failure invalidates the run |
| NaN / Inf | Any NaN or Inf in tracked metrics invalidates the run |
| Step-time explosion | `p95(step_time) > 2.5 × median(step_time)` |
| Heartbeat stall | No progress after warmup for `max(60s, 10 × median step time)` |
| VRAM guard breach | Peak reserved VRAM exceeds `0.92 × total VRAM` |

These gates apply to probes, sweeps, and training runs alike.

## Sweep Discipline

- Choose exactly one budget mode per sweep: `iso-VRAM`, `iso-params`, or `iso-FLOPs/step`.
- Run the systems curve first: throughput, stability, step-time tails, and resource limits.
- Only run the quality curve after the systems curve is stable.
- Start with a coarse sweep, then rerun the best cells with multiple seeds under the same contract.
- If a result does not reproduce under the same contract, treat it as unconfirmed.

## How This Connects to Public Truth

- **Current mainline** means the setting is actually shipped in code on `main`.
- **Validated finding** means the result is reproducible, but not yet promoted into the canonical code path.
- **Experimental branch** means the direction is active, but should not be described as a default.

Use those labels consistently across the repo front door, the architecture page, and the findings page. If code and docs disagree about **Current mainline**, the code wins.

## Read Next

- [Home](Home)
- [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture)
- [Validated Findings](Validated-Findings)
- [Documentation Governance](Governance)

If the GitHub wiki render looks incomplete, use [Pages](https://vraxion.github.io/VRAXION/) or the repo [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md).
