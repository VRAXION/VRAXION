<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Glossary

## What This Page Is

This page is a **secondary reference glossary** for the current VRAXION public stack. It defines the most important terms used across the active INSTNCT-era docs without turning terminology into doctrine.

## Current Terms

**Current mainline**
What is actually shipped in code on `main`. If a public page and the code disagree about current behavior, the code wins.

**Validated finding**
An experiment-backed result that has not yet been promoted into the canonical code path.

**Experimental branch**
An active build target or design direction that is still under evaluation, not a live default.

**INSTNCT**
The current public architecture line for VRAXION: a self-wiring, gradient-free system where the learnable object is the evolving directed graph and its co-evolved neuron parameters.

**Artifacts (run bundle)**
The minimum files required to treat a run as evidence: `run_cmd.txt`, `env.json`, `metrics.json`, and `summary.md`. See [Engineering Protocol](Engineering).

**Fail gates**
The hard invalidation conditions for a run, such as OOM, NaN/Inf, step-time explosion, heartbeat stall, or VRAM guard breach. See [Engineering Protocol](Engineering).

**Engineering Protocol**
The run and evidence contract for VRAXION experiments: how runs are executed, validated, and promoted into public truth. See [Engineering Protocol](Engineering).

**Release Notes**
The live secondary public status page for milestone-level progress and current public targets. See [Release Notes](Release-Notes).

## Legacy Terms and Historical Context

These terms belong mainly to earlier architecture lines and remain useful only as historical context.

**LCX (Latent Context Exchange)**
An earlier external memory system built around hash-bucketed memory slots. It belongs to the Diamond Code v3 era, not the current INSTNCT public line. See the retired [Diamond Code v3 Architecture](Diamond-Code-v3-Architecture) page or [Legacy Vault](https://github.com/VRAXION/VRAXION/wiki/Legacy-Vault).

**Zoom gate**
A learned gate used in the earlier LCX memory pathway to control how much memory content was blended into the hidden state. Historical term only. See the retired [Diamond Code v3 Architecture](Diamond-Code-v3-Architecture) page.

**Bottleneck projection**
An earlier compression block used to translate LCX reads into the model’s hidden-state space. Historical term only. See the retired [Diamond Code v3 Architecture](Diamond-Code-v3-Architecture) page.

**C19 activation**
A periodic activation used in the Diamond Code v3 era. It is not part of the current INSTNCT public story and should be treated as legacy context unless explicitly discussed in historical pages. See the retired [Diamond Code v3 Architecture](Diamond-Code-v3-Architecture) page for era context.

**Score Margin**
A routing-quality metric from the Diamond Code v3 observability stack. It is useful historical context for older memory experiments, but it is not a core term in the current INSTNCT public layer. See [Legacy Vault](https://github.com/VRAXION/VRAXION/wiki/Legacy-Vault) for broader historical context.

## Read Next

- [Home](Home)
- [VRAXION Architecture (INSTNCT)](SWG-v4.2-Architecture)
- [Validated Findings](Validated-Findings)
- [Legacy Vault](https://github.com/VRAXION/VRAXION/wiki/Legacy-Vault)
