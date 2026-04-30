# Doc Audit Notes

_Audited: 2026-04-30, against release `v5.0.0-beta.8`_

Scope: inconsistencies, dead links, stale version numbers, and drifting duplicated
information found between `README.md` and the doc set (`BETA.md`, `CHANGELOG.md`,
`VALIDATED_FINDINGS.md`, `Python/README.md`, `Rust/README.md`, `docs/VERSION.json`).

Each finding is labeled: **BLOCKER** (contradicts stated behavior or breaks
5-minute proof), **MINOR** (stale or inconsistent but not immediately confusing),
or **NICE-TO-HAVE** (low-priority drift).

---

## BLOCKER findings

### B-1: Python deploy SDK described as working vs. "Under construction" placeholder

**Files in conflict:** `README.md`, `VALIDATED_FINDINGS.md` (Current State block)
vs. `Python/README.md`

`README.md` describes `Python/` as "Block A + B, pure numpy, no ML framework
dependency" and includes `python -m pytest Python/ -q` in the 5-Minute Proof section
as a working verification step. `VALIDATED_FINDINGS.md` Current State block repeats
the claim: "Block A + B, pure numpy (256/256 + 65536/65536 lossless)."

`Python/README.md` says, on its very first status line: "Under construction. First
port (byte encoder) is the next step. Until then, this folder is a placeholder."

A newcomer running the 5-Minute Proof and then reading `Python/README.md` sees a
direct contradiction: the top-level docs say "working SDK," the SDK's own README says
"placeholder."

**What is actually true (verified 2026-04-30):** `Python/block_a_byte_unit/` and
`Python/block_b_merger/` each contain real code, a `__init__.py`, and a `tests/`
subdirectory. Block C (`Python/block_c_embedder/`) also exists with code. The pytest
suite likely passes. The `Python/README.md` "under construction" language was written
before blocks A and B were ported and has not been updated to reflect current reality.

**Recommended fix:** Update `Python/README.md` status line to reflect that Block A
and B are implemented and tested; flag Block C as implemented but not listed in the
top-level README table; flag Embedder V1 and Nano Brain V1 as scaffolds awaiting
training. (Do not change any of the files noted as read-only for this audit pass.)

---

### B-2: Block C omitted from Python/README.md component table

**Files in conflict:** `Python/README.md` component table vs. `CHANGELOG.md`,
`VALIDATED_FINDINGS.md`, and the actual `Python/` directory tree

`Python/README.md` lists five planned components in a table; none mentions Block C.
`CHANGELOG.md` (beta.3 era, 2026-04-21) documents "Block C byte-pair embedder champion"
as a frozen validated finding with a Python deploy SDK at `Python/block_c_embedder/embedder.py`.
`VALIDATED_FINDINGS.md` lists Block C as "Validated finding / Frozen champion."
The directory `Python/block_c_embedder/` physically exists.

A newcomer reading `Python/README.md` has no way to discover Block C from that file.
The README also does not link to the per-block READMEs (only `README.md` top-level
does via the "Read Next" block).

**Recommended fix:** Add Block C row to `Python/README.md` component table with
status "frozen champion, implemented."

---

## MINOR findings

### M-1: README.md "Read Next" links to per-block READMEs — verify they exist

**Files:** `README.md` line 116

`README.md` links:
- `Python/block_a_byte_unit/README.md` — **verified to exist.**
- `Python/block_b_merger/README.md` — **verified to exist.**

No broken link here, but the link set omits `Python/block_c_embedder/` which is a
frozen shipped champion. An auditor checking this section would not find Block C.

---

### M-2: `python_surface` field in VERSION.json describes the archived research lane, not the deploy SDK

**File:** `docs/VERSION.json`

The `python_surface` field reads:
`"archived at tag archives/python-research-20260420 (frozen 2026-04-20; was instnct/)"`.

This correctly describes the old research lane, but a reader expects `python_surface`
to describe the current Python surface, which is `Python/` (deploy SDK). The deploy
SDK has its own field `python_deploy_sdk` but a newcomer parsing this JSON may
misread `python_surface` as "Python is fully archived / no Python surface exists."

**Recommended fix:** Rename or add a clarifying comment. For example:
`"python_surface": "Python/ (Block A + B + C deploy SDK, active)"` and keep
`"python_research_lane": "archived at tag archives/python-research-20260420"`.

---

### M-3: README mentions `Python/graph.py` as "historical reference"

**File:** `README.md`, Current State section, line 63

`README.md` says: "The Python `graph.py` lane remains in-repo as historical
reference/support, not the public mainline."

Verified: `graph.py` does not appear anywhere in the current `Python/` directory tree;
the Python research lane was fully archived to tag `archives/python-research-20260420`
on 2026-04-20. If `graph.py` was part of that archived lane it is no longer on `main`.

This is stale copy that could confuse a newcomer into looking for a file that no
longer exists on `main`.

---

### M-4: `Rust/README.md` and `Python/README.md` are near-identical templates with no cross-link

**Files:** `Rust/README.md`, `Python/README.md`

Both files are clearly from the same template, contain identical "What goes here" /
"What does NOT go here" / "Status" headings, and reference each other as siblings.
However, neither links to the top-level `README.md` or `BETA.md` for a reader to
understand where these SDKs fit in the overall project. A newcomer landing in `Rust/`
or `Python/` has no path back to the front door without a browser back button.

**Recommended fix:** Add a one-line "Back to top: [README](../README.md)" at the top
of each SDK README.

---

### M-5: VALIDATED_FINDINGS.md "Released public tag" history list in Current State block is truncated

**File:** `VALIDATED_FINDINGS.md`, Current State block, line 11

The parenthetical tag history reads:
`(v5.0.0-beta.7 is the prior Phase D9 smooth+accuracy specialist checkpoint;
v5.0.0-beta.6 is the prior Phase D6/D7/D8 ...; v5.0.0-beta.5 is the prior ...;
v5.0.0-beta.1 remains as the original language-evolution beta, historical reference only)`

Beta.2, beta.3, and beta.4 are omitted from this history line (they exist in
`CHANGELOG.md` and in the longer `BETA.md` cascade). The omission is not a broken
claim but a reader expecting a complete version chain will notice the gap. `BETA.md`
has a fuller cascade that includes all eight betas.

---

## NICE-TO-HAVE findings

### N-1: Two distinct archive tag naming conventions coexist

**Files:** `ARCHIVE.md`, various doc references

Tags named `archives/<name>-<date>` (e.g., `archives/python-research-20260420`) and
tags named `archive/<name>-<date>` (e.g., `archive/main-pre-cleanup-20260428`) coexist.
The `archives/` prefix appears to be used for content snapshots; `archive/` for
branch-head snapshots. This distinction is not documented in `ARCHIVE.md`. A reader
trying to restore a specific file cannot reliably tell from the name alone which
convention applies.

---

### N-2: `docs/wiki/` mirror — relationship to GitHub wiki not explained in top-level README

**File:** `README.md`, line 53

README says: "Retired line names and older local folders belong in Project Timeline
(wiki link), not in the current front-door stack." It also notes the wiki is a
"secondary mirror, not an independent source of truth" elsewhere. But the `docs/wiki/`
directory on `main` is never explained to a newcomer as the canonical source that the
wiki mirrors. A reader who finds `docs/wiki/Home.md` does not know whether to trust
it or the GitHub wiki page at the same URL.

---

_End of audit. Total findings: 2 BLOCKER, 4 MINOR, 2 NICE-TO-HAVE._
