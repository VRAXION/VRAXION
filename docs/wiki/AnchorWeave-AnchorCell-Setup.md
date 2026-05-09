# AnchorWeave / AnchorCell Setup

Status snapshot: 2026-05-09.

This page is the default handoff surface for the AnchorWeave / AnchorCell line. It records what exists, what was tested, what failed, what is currently locked, and what should not be repeated.

## Current Status

- AnchorWeave scaffold exists on `main`.
- Canonical storage is rich append-only AnchorCell JSONL.
- Real or private cells must stay private unless explicitly sanitized.
- Current experimental status: the scaffold is valid, but the AnchorWeave format-effect is not proven.
- AWFT-001 synthetic navigation tests were useful as measurement infrastructure, but they did not justify scaling human AnchorCell collection.
- Later HF/LoRA and forced-choice work has been built and tested as experimental runner work; keep it separate from the stable AnchorCell storage contract until merged and reviewed.

## Claim Boundary

AnchorWeave is not a dataset of consciousness.

AnchorWeave is not proof of grounding at scale.

AnchorWeave is infrastructure for operational symbol-grounding research via relational episodic anchors:

```text
episode -> relational graph -> salience -> action/outcome -> counterfactuals
-> memory hook -> pre-symbol abstraction -> symbol attach/reject
```

The working thesis remains:

```text
Do not ground the symbol directly.
Store relational episodic anchors from which stable abstractions can emerge.
Symbols attach late.
```

## Locked AnchorCell Architecture

The locked architecture is not a flat chain. A cell has a canonical core and a derived probe specification:

```text
AnchorCell
  Core
    Meta
    Situation
    ImplicitJob
    RelationalModel
    SalienceMap
    Actions
    Outcomes
    MemoryHooks
    HumanSourceTrace optional/private
    DistilledPolicy
    CounterfactualChecks
    SymbolAttachReject
    ClaimBoundary
    WorldTruth private/eval-hidden

  ProbeSpec derived
    PromptArms
    CandidateActions
    ChoiceOrderSeeds
    FreeResponseTaxonomy
    PairwiseTrapProbes
    ParaphraseVariants
    ScoringSpec
    PassCriteria
```

Core rules:

- The explicit event, scene, task text, and any note or letter belong in `Situation`. They are not the anchor.
- `ImplicitJob` is the latent job extracted from the explicit situation: what the situation is really asking the agent to do.
- `RelationalModel` and `SalienceMap` preserve the episode structure, constraints, tempting distractors, and what should or should not matter.
- `Actions`, `Outcomes`, and `MemoryHooks` are core fields. They tie grounding to available behavior, expected effects, observed effects, and future retrieval cues.
- `HumanSourceTrace` is raw human think-aloud source material. It is optional, private by default, and not canonical truth.
- `DistilledPolicy` is the cleaned, reusable, leak-safe model-facing policy extracted from the source trace.
- `CounterfactualChecks` test whether the policy rejects shortcuts and survives plausible changes.
- `SymbolAttachReject` stays late: symbols attach after relational, salience, and policy structure, not before.
- `ClaimBoundary` states what the cell does and does not justify.
- `WorldTruth` stores hidden world facts for authoring or evaluation. It must not be model-facing.
- Candidate wording, order seeds, pairwise traps, free-response categories, and scoring belong to `ProbeSpec`, not the canonical cell core.

For new agents, the safest rule is:

```text
Core = what happened, what mattered, what policy was distilled, and what is true.
ProbeSpec = how we measure whether a model learned or follows that policy.
```

High-connectivity decision terrain is not stored in one monologue field. It spans `RelationalModel`, `SalienceMap`, `Actions`, `Outcomes`, `MemoryHooks`, and optional/private `HumanSourceTrace`.

Compatibility note:

```text
AnchorWeave-v1.0 remains the current append-only storage schema.
Core + ProbeSpec is the locked conceptual standard for HGA-DESK/S01 and future v2 work.
S01 is a probe manifest until a v2 schema or deterministic v1 exporter exists.
```

## What Was Built

Stable scaffold on `main`:

- AnchorCell schema and taxonomy.
- Two sanitized example cells.
- Dataset README and dataset card.
- Collection protocol and full dataset spec.
- Validator, append tool, and export skeleton.
- Private-data `.gitignore` rules.
- AWFT-001 deterministic synthetic A/B scaffold for navigation / route-memory disambiguation.

Experimental runner work:

- HF/LoRA same-base training runner.
- HF structured-output inference audit path.
- AWFT-001-FC forced-choice runner that removes model-generated JSON from the measurement path.
- Local run artifacts and scoreboards under ignored `target/`.

## What We Learned

JSON-generation evaluation failed as a measurement path because output format compliance was unstable. The model could run through the pipeline, but invalid or malformed structured output made the grounding result hard to interpret.

Forced-choice NLL scoring works better as a measurement surface:

```text
prompt + candidate completion -> candidate-token loss
```

This removes parser, schema, markdown, and missing-field noise. It does not prove recall. It only measures preference among supplied candidate plans, so it must be paired with free-response and adversarial baselines.

AWFT-001-FC result was negative:

- Rich prose beat AnchorWeave on raw candidate accuracy.
- AnchorWeave improved over plain QA, but did not beat rich prose.
- Prior-corrected scoring collapsed toward shortcut candidates.
- Shuffled AnchorWeave failed as expected.

Conclusion:

```text
The current synthetic AnchorWeave navigation setup did not prove a format-effect.
Do not scale human AnchorCell collection from that result.
```

## Current Default Direction

Do not scale human AnchorCell collection yet.

Do not keep expanding synthetic AWFT navigation as the main line.

Move to human-grounded microtests with gradient value/cost scoring:

- small number of carefully locked cells,
- explicit human search judgment,
- adversarial controls,
- forced-choice plus free-response audit,
- mean value remaining as the primary metric.

The next default probe contract is `HGA-DESK-001 / DeskCache S01`.

## Locked Probe Contract: HGA-DESK-001 / DeskCache S01

Task:

```text
Economical search for a USB drive on a cluttered desk.
```

Forced-choice is the primary low-noise recognition probe. It is not sufficient evidence that the model can recall or generate the intended search policy. S01 only becomes a standard cell after it passes forced-choice, choices-only, free-response, order-robustness, and paraphrase checks.

Hidden truth:

```text
The USB drive is plugged into the keyboard side USB port.
```

Primary metric:

```text
mean_value_remaining
```

Exact accuracy is secondary. The point is not only whether the model chooses the gold plan, but how much search energy it wastes if it chooses a plausible trap.

Four prompt arms:

```text
BASE
STYLE_CONTROL
CORRECT_ANCHOR
CORRUPTED_ANCHOR
```

S01 maps onto the locked architecture as:

```text
Situation:
  desk layout, assistant note, task frame, assistant unreachable

ImplicitJob:
  low-cost search under assistant-intent constraints

Actions:
  first-search plans available in the desk search space

Outcomes:
  expected search energy and value remaining for each first-search plan

MemoryHooks:
  retrieval cues such as assistant intent, storage-vs-use, private boundary,
  clutter avoidance, dirty-area avoidance, and low-cost diagnostic action

HumanSourceTrace:
  the user's Hungarian inner-monologue source layer; private/authoring source

DistilledPolicy:
  leak-safe correct anchor policy; it must not mention keyboard, port, plugged, or monitor

WorldTruth:
  USB is plugged into the keyboard side USB port; hidden from model-facing prompts

ProbeSpec:
  prompt arms, candidate plans, choices-only baseline, free-response audit,
  pairwise trap probes, paraphrase variants, scoring, and pass criteria
```

DeskCache S01 is a Search-to-Decision Anchor:

```text
physical search space -> internal decision terrain -> action/outcome
```

It tests whether a model links possible desk locations to costs, social boundaries, shortcut pressure, assistant intent, and a diagnostic first action.

S01 text placement is locked as:

| Existing text | Locked layer | Treatment |
|---|---|---|
| Desk layout | `Situation` | Keep as the visible scene. |
| Assistant note / letter | `Situation` | Use the cleaned model-facing note, not the raw brainstorm. |
| "Find the USB with least wasted search" | `Situation` and `ImplicitJob` | The explicit task stays in `Situation`; the latent job is low-cost search under intent constraints. |
| User's Hungarian inner monologue | `HumanSourceTrace` | Preserve as private authoring source. Do not present it directly as the anchor. |
| Human anchor policy summary | `DistilledPolicy` | Convert into the leak-safe `CORRECT_ANCHOR`. |
| `STYLE_CONTROL` and `CORRUPTED_ANCHOR` drafts | `ProbeSpec.PromptArms` | Balance them against `CORRECT_ANCHOR` by length, style, and fluency. |
| First-search action set | `Actions` | Keep the possible first actions in the core action landscape. |
| Candidate wording / order | `ProbeSpec.CandidateActions` | Use rendered candidate text for measurement only; it is not canonical cell truth. |
| Costs, values, and trap labels | `ProbeSpec.ScoringSpec` | Eval-facing only. |
| Keyboard-side USB-port truth | `WorldTruth` | Hidden truth; never leak into `BASE` or `CORRECT_ANCHOR`. |
| Free-response categories | `ProbeSpec.FreeResponseTaxonomy` | Include `task_frame_drift`. |

Required probes:

```text
forced-choice NLL across multiple deterministic candidate-order seeds
choices-only baseline
free-response audit
pairwise trap checks as diagnostics only
candidate paraphrase robustness
```

## S01 Lock Rules

- `BASE` must not reveal the keyboard-side port.
- `CORRECT_ANCHOR` must not mention `keyboard`, `port`, `plugged`, or `monitor`.
- `STYLE_CONTROL` must match form and approximate length, but must not include domain-specific search policy.
- `CORRUPTED_ANCHOR` must be fluent and plausible, but semantically wrong.
- Candidate plans must be complete first-search plans, not single labels.
- Candidate order must be randomized by seed and scored by `candidate_id`, never by position.
- Single-order forced-choice results are not trusted; aggregate across multiple candidate-order seeds.
- Pairwise trap probes are diagnostic only and must not replace multiclass forced-choice.
- No assistant-call option. The assistant is unreachable.
- No mounted-drive option in S01. Make that a sibling cell if needed.
- No-valid-option reflective judgment belongs in a later sibling cell, not S01.
- Candidate text must be length/style balanced.
- The gold candidate must not be shorter than the traps.
- The gold candidate must not reuse exact anchor phrases such as `ready-to-use` or `hand-work area`.
- Do not repeat the gold fallback phrase inside every wrong candidate. Use a neutral fallback like `continue with the remaining desk search`.
- Report a choices-only baseline. If the gold candidate wins without the scene or note, the cell is invalid as a grounding probe.
- S01 is a locked probe contract, not a validated standard cell, until it passes the full guard stack.

## Current S01 Cell Contract

Desk layout:

```text
Back row:
- wallet on the left
- monitor in the center
- pen cup on the right

Middle:
- keyboard
- ashtray with cigarette butts and tiny trash
- cigarette pack and lighter beside the ashtray

Front:
- mouse
- microphone
- small electronics pouch
- closed USB holder stand
```

The pen cup is crowded with pens, paper clips, erasers, SD cards, and small unrecognized bits. The USB holder stand is visible, but its front slot looks empty and too shallow for this USB drive to sit there properly.

Assistant note:

```text
Hey boss,

I couldn't really put your USB away properly. The little holder at the front of
the desk was a bad fit, and I didn't want to force it.

I also didn't want it mixed with loose bits, grit, metal pieces, or look-alike
junk where it could get dirty, scratched, or disappear from view. I know there
are important materials on it for tomorrow's work, so I didn't do anything
technical to it; I just tried to leave it readable and easy for you to get back
to work.

And do not worry, I did not touch your private things or anything that seemed
like a personal valuables area.

I left it in a place that should let you get back to work with almost no fuss.
You should not need to dig, clean, open personal things, or sort through junk
first. Think less like putting it away, and more like leaving it immediately
usable.

I'm off today, my phone is away, and you are on your own. Good luck with the
work.
```

Task framing:

```text
You are looking for the USB with as little wasted searching as possible.
Your assistant is not reachable right now, so you cannot ask them directly.
Choose the best first search plan.
```

Candidate plans:

| candidate_id | candidate text pattern | cost | value_remaining | trap_type |
|---|---|---:|---:|---|
| `keyboard_side_port_first` | Check the keyboard's side USB port first; if it is not there, continue with the remaining desk search. | 1 | 9 | `gold` |
| `monitor_ports_first` | Check the monitor's USB ports first; if it is not there, continue with the remaining desk search. | 3 | 7 | `plausible_peripheral` |
| `usb_holder_first` | Check the USB holder stand first; if it is not there, continue with the remaining desk search. | 2 | 8 | `surface_association` |
| `electronics_pouch_first` | Open the small electronics pouch first; if it is not there, continue with the remaining desk search. | 4 | 6 | `storage` |
| `pen_cup_first` | Search the pen cup clutter first; if it is not there, continue with the remaining desk search. | 5 | 5 | `small_object_clutter` |
| `smoking_area_first` | Check the cigarette pack and lighter area first; if it is not there, continue with the remaining desk search. | 4 | 6 | `dirty_storage_like` |
| `ashtray_first` | Search the ashtray first; if it is not there, continue with the remaining desk search. | 6 | 4 | `dirty_violation` |
| `wallet_first` | Open the wallet first; if it is not there, continue with the remaining desk search. | 3 | 7 | `personal_boundary` |

Human anchor policy summary:

```text
Do not start by rummaging through containers.
Model the assistant's intent: where would they leave the USB so it is useful,
clean, easy to find, and not mixed into private, dirty, or slow-to-sort places?
First test the lowest-cost use-state possibility before spending energy opening,
sorting, or cleaning anything.
```

The exact `CORRECT_ANCHOR` text must preserve this policy while avoiding the banned location words listed above.

`STYLE_CONTROL` should use the same shape and first-person inner-voice style, but only generic reasoning content:

```text
I should stay calm, use the note carefully, and avoid wasting effort.
I should compare each possible plan against the details before choosing.
```

`CORRUPTED_ANCHOR` should be fluent but wrong:

```text
For a USB, storage and USB-looking clutter are the safest first assumptions.
If the holder is uncertain, check small containers and similar-looking objects
before thinking about use-state.
```

## Required S01 Metrics

Primary:

```text
mean_value_remaining
```

Secondary:

```text
optimal_action_rate
surface_false_positive_rate
storage_false_positive_rate
dirty_area_false_positive_rate
boundary_violation_rate
gold_margin
choices_only_gold_rate
free_response_goldish_rate
pairwise_gold_win_rate
paraphrase_consistency
```

Free-response categories:

```text
goldish_active_use
near_miss_peripheral
surface_storage
small_object_clutter
dirty_or_smoking_area
personal_boundary
task_frame_drift
other
```

`task_frame_drift` means the answer stops giving a concrete first search place or action and shifts into USB safety, data protection, legal or meeting importance, cleaning, moral analysis, or general advice.

## Next Commands / Workflow

Create the single-cell target-only probe first.

Keep generated outputs under:

```text
target/
```

Do not write private data under:

```text
data/anchorweave/cells/
```

Do not scale to 10-20 DeskCache cells until S01 shows a real anchor movement signal:

```text
choices-only baseline does not reliably select the gold plan
CORRECT_ANCHOR > BASE on mean_value_remaining
CORRECT_ANCHOR > STYLE_CONTROL on mean_value_remaining
CORRUPTED_ANCHOR moves toward storage/clutter traps
free-response moves toward active-use / keyboard-like answers
effect survives multiple candidate-order seeds
effect survives at least one paraphrase variant
```

If S01 fails, revise the cell or the human-anchor layer before collecting more cells.

S02 can test monitor or mounted-drive logic. A later S0X can test no-valid-option refusal or reflective judgment. These must not be mixed into S01.

## Publication Caveat

This page is public once synced to the GitHub Wiki. Do not use internet-connected models that may have seen this page as blind evaluation subjects for S01. The intended blind subjects are local or otherwise controlled models whose training data cannot include this page after publication.
