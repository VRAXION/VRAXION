# HGA-DESK-001 / DeskCache S01 Contract

Status: locked probe contract, not a validated standard cell.

This document locks the first human source layer for DeskCache S01. It is not a
canonical `AnchorWeave-v1.0` JSON cell, not a model-facing prompt, and not a
training target. It is public-safe source material for deriving the leak-safe
`DistilledPolicy` later.

## Claim Boundary

- S01 is a Search-to-Decision Anchor: physical search space -> internal decision
  terrain -> action/outcome.
- The trace below is a sanitized `HumanSourceTrace`, not polished rationale and
  not model-facing chain-of-thought.
- Concrete gold-location words are allowed here because this is source material.
  They must not leak into `CORRECT_ANCHOR` / `DistilledPolicy`.
- No private AnchorCell is committed here. Do not write this under
  `data/anchorweave/cells/`.

## Locked HumanSourceTrace

```text
HUMAN_SOURCE_TRACE / HGA-DESK-001 / S01 / HU / sanitized public source

Bejövök dolgozni, megyek az asztalom felé, és közben látom, hogy valami fura van ott. Valami fehér lap az asztalon. Ahogy közeledek, látom, hogy félbe van hajtva. Kinyitom, és felismerem az asszisztensem írását. Az aljára pillantva konfirmálom: igen, ezt az asszisztensem írta.

Valószínűleg azzal kapcsolatos, amit tegnap délután kértem tőle: rakja rá a fontos anyagokat az USB-kulcsomra, és tegye olyan helyre, hogy ma a fontos megbeszélés előtt ne ezzel menjen el az idő. Nem késlekedhetünk. Szóval nem csak egy tárgyat keresek, hanem egy olyan tárgyat, amit valaki szándékosan hagyott nekem valahol, hogy gyorsan vissza tudjak állni munkába.

Elolvasom a levelet. A holderbe nem fért bele? Ez elsőre furcsa, mert pont azért vettem azt a tartót, hogy az USB-k ott legyenek. Bosszant is: eddig minden ilyesmi ott volt, és van bennem egy automata kényszer, hogy azért csak megbökjem vagy kinyissam, hátha mégis ott van. USB -> USB-tartó, ez az első reflex. De a levél pont ezt gyengíti: nem volt jó fit, nem akarta erőltetni. Ez lehetne könnyű check, de valószínűleg felesleges első lépés.

Azt írja, hogy nem akarta kacat, fém, grit vagy look-alike junk közé tenni. Ez már logikusabb. Nem rejtekhelyet keresünk neki, hanem olyan helyet, ahol nem veszik el és könnyen megtalálom. A pen cup csábító, mert tele van apró dolgokkal, SD-kártyával, gemkapoccsal, radírral, mindenféle random bittel, és egy USB fizikailag simán eltűnhetne benne. Én is szoktam oda bedobni dolgokat. De ha én lennék az asszisztens, és segíteni akarnék, nem oda tennék egy fontos USB-t, ahol sok false positive között kell túrni. Ez egyszerre biztonságosnak tűnő tároló és rossz keresési hely.

A hamutartó és a cigis környék is felmerül, mert technikailag bármi beeshet oda, vagy valaki oda is dughat valamit. De ez nagyon rossz fit. Koszos, lassú, utómunkás, és ha már ilyen helyre kerülne, akkor majdnem olyan, mintha a kukába dobtuk volna. Lehetséges véletlenként, de nem jó első szándékos keresési hipotézis. Nem akarok először olyan helyhez nyúlni, ahol takarítani, bányászni vagy koszt kerülgetni kell.

Látom a pénztárcámat is az asztal bal hátsó részén. Fizikailag akár abba is beleférne az USB, és én magam talán néha tennék furcsa helyekre dolgokat. De asszisztensként más pénztárcájába belenyúlni szociális határ. Az asszisztens ezt valószínűleg kerülte volna, és a levél is azt sugallja, hogy nem nyúlt személyes értékzónához. Nem azért zárom ki, mert lehetetlen, hanem mert emberileg és munkakapcsolati szempontból nagyon valószínűtlen.

Szóval akkor mi maradt? Ha én lennék az asszisztens, hova tenném? Nem mérnöki tökéletességgel gondolkodott valószínűleg, hanem pragmatikusan: legyen meg, ne koszolódjon, ne vesszen el, ne kelljen turkálni, és tudjak dolgozni. A holder kilőve, a pénztárca szociális határ, a hamutartó és cigis rész koszos, a pen cup és pouch tároló/logikai lehetőség, de nem érzem bennük az AHA-t. Valami kimagaslóbb magyarázat kell.

A legegyszerűbb információs megoldás az lenne, hogy felhívom. Megkérdezem, hova tette, ő elmondja, kész. Ez lenne a legolcsóbb keresés: nulla turkálás, nulla kockázat. De ez ki van lőve. A levélből tudom, hogy nem elérhető, a telefonja nincs nála. Akkor magamnak kell egy olcsó diagnosztikus checket választani.

Itt vált át bennem a kérdés. Nem az a kulcs, hogy "USB hol szokott lenni?", hanem hogy "ha egy segítő ember úgy hagyta itt, hogy gyorsan dolgozni tudjak, akkor milyen állapotban hagyta?" Ez nem storage-state-nek hangzik. Inkább use-state-nek. Nem elrakta, hanem úgy hagyta, hogy szinte azonnal használható legyen.

Ha használatra kész, akkor lehet, hogy be van dugva valahová. Először eszembe jut a gép hátulja, de az asztal alatt van, ki kell húzni, nehéz hozzáférni, ez nem illik a "szinte semmihez nem kell hozzányúlni" érzéshez. Aztán eszembe jut, hogy jönnek fel USB-s dolgok az asztalra. A monitor is lehet USB hub. Tényleg, azon is lehet port. De a monitor nagy, nehéz, drága, üveg, nem fogdosós tárgy. Ha megfogom, bepiszkolódik, ha mozgatom, leeshet, nehéz hozzáférni a hátuljához vagy oldalához. Nem akaródzik ezzel kezdeni.

A billentyűzet viszont egészen más. Kicsi, tartósabb, kéz alatti, pont arra van, hogy hozzáérjek, mozgassam, írjak rajta. Munka közben ehhez térek vissza. Ha ezen van USB-csatlakozási lehetőség, az sokkal jobb első check: gyors, tiszta, alacsony költségű, és illik ahhoz, hogy a pendrive nem elrakva, hanem használatra készen lett hagyva.

Felemelem a kezem a monitor és a billentyűzet felé, és döntök: először nem a monitort bolygatom, hanem a billentyűzet felé indulok. Végighúzom a kezem azon az oldalán, ahol az USB-portok lehetnek, és azt figyelem, érzek-e recés lyukat, kitüremkedést, vagy egy kis téglalap alakú testet. Ha megérzem, megpróbálom kihúzni. Ha nem jön vagy nem egyértelmű, odahajolok, ránézek, vagy óvatosan felemelem a billentyűzetet. Ez 1-2 másodperces diagnosztikus check, és ha megvan, azonnal vége a keresésnek.

Ha nincs ott, akkor nem omlik össze a terv. Akkor megnézem a billentyűzet másik oldalát is, aztán tágítok más use-state/periféria helyek felé: például a monitor környékére, de óvatosan, mert az már nehezebb és kockázatosabb tárgy. Utána jöhet vizuális scan a pen cupban, majd a pouch vagy más tisztább tároló. Csak később mennék tényleges túrásba, holder sanity checkbe, pénztárcába vagy koszos helyek felé. Ha minden intelligens keresési sorrend bukik, akkor átmegyek exhaustive/frustration searchbe, de az már nem jó első policy, csak végső kényszer.

Szóval itt nem egyszerűen az a kérdés, hogy "hol van az USB?". Az a kérdés, hogyan keressek meg minimális energiával egy tárgyat, amit egy segítő ember hagyott nekem valahol. Ehhez nem elég a tárgy kategóriája. Figyelni kell a levélre, a keresési költségre, a koszra, a személyes határokra, a tároló-vs-használat különbségre, és arra, hogy az asszisztens fejével gondolkodva mi lenne a leggyorsabb, leghasznosabb elhelyezés.
```

## Extraction Notes For DistilledPolicy

Keep:

- shortcut pressure and inhibition,
- assistant-intent simulation,
- storage-state vs use-state frame shift,
- low-cost diagnostic first action,
- clean/private/clutter/dirty boundary reasoning,
- fallback from intelligent search to broader search.

Do not leak into `DistilledPolicy`:

- keyboard,
- port,
- plugged,
- monitor,
- personal names,
- concrete meeting/company details.

## Locked Situation / BASE Prompt

This is the shared model-facing situation used by every prompt arm. It contains
the explicit task, desk layout, assistant note, and unreachable-assistant
constraint. It must not reveal the hidden truth.

Hidden truth, not model-facing:

```text
The USB drive is plugged into the keyboard side USB port.
```

Leakage rules for the situation:

- It may mention the keyboard and monitor as visible desk objects.
- It must not mention keyboard ports, monitor ports, hubs, plugged-in state, or
  mounted-drive checks.
- It must keep the task as a first-search-policy problem, not a technical
  troubleshooting problem.
- It may include mild task-frame red herrings about important work and keeping
  the drive readable, but must not turn the task into legal, security, or device
  maintenance advice.
- It must state that calling the assistant is unavailable.

```text
BASE_PROMPT / HGA-DESK-001 / S01 / EN / model-facing

You are looking for a small USB drive that your assistant left somewhere on your
desk. You need it for work soon, and you want to find it with as little wasted
searching as possible.

The desk is cluttered but familiar. Along the back row, there is a wallet on the
left, a monitor in the center, and a pen cup on the right. The pen cup is
crowded with pens, paper clips, erasers, SD cards, and several small loose items
that are not immediately recognizable.

In the middle of the desk is your keyboard. Near it, there is an ashtray with
cigarette butts and tiny bits of trash in it. Beside the ashtray are a cigarette
pack and a lighter.

At the front of the desk, there is a mouse, a microphone, a small pouch for
electronics, and a closed USB holder stand. The USB holder stand is visible, and
its clear front slot is empty.

Your assistant left this note:

"Boss,

I could not put your USB drive into the small holder at the front of the desk.
It was a bad fit, and I did not want to force it.

I also did not want to put it among loose bits, grit, metal pieces, or
look-alike junk where it could disappear or pick up debris. It has important
work material on it, so I wanted it to stay easy to recover and readable.

I did not put it in the smoky or dirty stuff either. You should not have to dig
through that before work.

I also stayed away from your personal valuable things. I did not want you to
worry that I had gone through anything like that.

I left it in the best place I could think of for getting back to work quickly.
You should barely have to move anything if you think about where it would be
useful rather than where it would be stored.

Do not call me about it today. I am away and will not have my phone with me."

Choose the best first search plan.
```

## Locked AnchorCellCore v0.1

This is the public-safe conceptual training-cell core for S01. It is not an
`AnchorWeave-v1.0` JSON export and not a private append-only cell. It locks the
training semantics that can later be exported into a storage schema or derived
training views.

### Meta

```text
cell_id: HGA-DESK-001-S01
domain: desk_search / economical_search / assistant_intent
status: locked_core_v0_1
public_safety: sanitized_public_contract
storage_note: conceptual Core, not AnchorWeave-v1.0 JSON
```

### Situation

Use `BASE_PROMPT / HGA-DESK-001 / S01 / EN / model-facing` above as the shared
model-facing situation. The situation contains the visible desk layout,
assistant note, task framing, and assistant-unreachable constraint.

### ImplicitJob

```text
This is not ordinary object lookup. It is low-cost search under assistant-intent
constraints: infer the best first search action from the note, the desk layout,
social boundaries, handling cost, and storage-vs-active-use cues.
```

### RelationalModel

```text
assistant -> note -> constraints
assistant -> left_object -> intended_help
USB -> small_object -> storage temptation
USB holder -> obvious surface association -> weakened by note
pen cup / pouch -> small-object container temptation -> sorting cost
ashtray / smoking area -> possible but dirty/high-handling-cost area
wallet / personal valuables -> social boundary
work area -> active-use possibility -> low movement / low recovery cost
assistant unreachable -> self-search required
```

### SalienceMap

High:

```text
assistant intent
low-cost first action
active-use vs storage-state distinction
avoid rummaging/sorting/cleaning
personal boundary
assistant unreachable
```

Medium:

```text
USB object category
visible desk layout
holder being visible/empty
important work context
```

Low / do not overread first:

```text
USB means holder
search every container
meeting importance as legal/security advice
USB protection as technical maintenance task
dirty area accident hypothesis
```

### Actions

Core action landscape only, not rendered candidate wording:

```text
check active-use work-area first
check nearby peripheral/use-area first
check holder first
check electronics pouch first
search pen cup clutter first
check smoking-area objects first
search ashtray first
open wallet first
call assistant
exhaustive search
```

### Outcomes

```text
active-use work-area first -> cheapest diagnostic if assistant intended immediate work resumption
nearby peripheral/use-area first -> plausible but higher handling/access cost
holder first -> easy surface reflex but weakened by note
pouch/pen-cup first -> storage/category search with sorting risk
smoking/ashtray first -> dirty/high-handling-cost failure mode
wallet first -> personal-boundary violation
call assistant -> would be best information source but unavailable
exhaustive search -> fallback only after intelligent checks fail
```

### MemoryHooks

```text
assistant left object for me
note with constraints
not storage, active use
small object in clutter is tempting but costly
dirty area possible but bad first policy
personal boundary excludes plausible container
call unavailable forces self-diagnostic check
cheap diagnostic first, broaden later
```

### HumanSourceTrace

Use the locked Hungarian source trace above as authoring provenance only. It is
not a model-facing training view and not canonical truth.

### DistilledPolicy

Leak-safe model-facing policy:

```text
Interpret the note as intent evidence, not just object-location evidence. First
reject high-cost searches that require rummaging, cleaning, opening personal
items, or sorting through look-alike clutter. Do not follow the surface rule
"object category means storage place" when the note weakens storage. Prefer the
cheapest clean active-use check: a place where the object could already help
work resume with minimal movement. Broaden to containers, clutter, dirty areas,
or personal zones only after the low-cost use-state check fails.
```

### CounterfactualChecks

```text
If the assistant were reachable, calling first would dominate physical search.
If the holder were not excluded or weakened, holder-first would become more reasonable.
If the note did not imply quick work resumption, storage-first would become more reasonable.
If clutter/small-object containers were the only clean places, pouch or pen cup would rise.
If personal-boundary constraints were absent, wallet would be less bad but still not first.
If the active-use cue were removed, the cell should not force the active-use policy.
```

### SymbolAttachReject

Attach:

```text
search_cost
assistant_intent
active_use
storage_vs_use
low_cost_diagnostic
boundary_respect
clutter_false_positive
```

Reject:

```text
USB_means_holder
exhaustive_search_first
dirty_area_first
personal_boundary_first
object_category_only_reasoning
task_frame_drift
```

### ClaimBoundary

```text
This cell trains/evaluates economical search policy under assistant-intent
constraints. It does not prove grounding at scale, consciousness, or general
desk-search ability. S01 is one locked training/probe cell, not a validated
standard cell until it passes the guard stack.
```

### WorldTruth private/eval-hidden

```text
The USB drive is plugged into the keyboard side USB port.
```

### ModelFacingTrainingView

The richest leak-safe training view may include:

```text
Situation / BASE_PROMPT
ImplicitJob
RelationalModel
SalienceMap
Actions
Outcomes
MemoryHooks
DistilledPolicy
CounterfactualChecks
SymbolAttachReject
CORRECT_INNER_VOICE
```

It must exclude:

```text
WorldTruth
raw HumanSourceTrace
ProbeSpec
CandidateActions rendered as choices
STYLE_CONTROL
CORRUPTED_INNER_VOICE
cost/value scoring labels
```

## Model-Facing Correct Inner Voice

This is the sanitized inner-voice rendering for a `CORRECT_INNER_VOICE` prompt
arm. It preserves the decision terrain but removes the concrete answer.

Leakage rules:

- It must not mention `keyboard`, `port`, `plugged`, or `monitor`.
- It must not name a person or concrete company/meeting detail.
- It must not say the gold location directly.
- It may describe search-cost, use-state, storage-state, clutter, dirty areas,
  personal boundaries, assistant intent, and fallback order.

```text
CORRECT_INNER_VOICE / HGA-DESK-001 / S01 / EN / model-facing

I should not treat this as a plain "where does this object usually belong?" problem.

My first pull is toward the obvious storage place. That is the easy association:
this kind of object has a holder, so maybe I should check the holder anyway.
But the note weakens that path. If the assistant says the holder was a bad fit
and they did not want to force it, then starting there may be an automatic
habit, not the best search decision.

The small clutter zones are also tempting. A small object could physically
disappear among loose bits, cards, clips, erasers, and similar-looking things.
But if the assistant was trying to help me, they probably would not choose a
place where I have to sort through many false positives.

The dirty or smoking-related areas are technically possible too. Something
could fall there or be tucked there. But that would create mess, cleaning, and
extra handling. It does not fit the idea of leaving the item easy to recover.

The personal-items area is another boundary. It may be physically possible, but
a careful assistant would avoid crossing that social line unless there were no
better option.

So what is left? I need to simulate the assistant's intent. They were not
solving an abstract storage problem. They were trying to leave the item so I
could get back to work quickly, without digging, cleaning, opening private
things, or sorting through look-alike clutter.

Calling the assistant would be the cheapest information source, but the note
blocks that option. I have to choose a low-cost diagnostic check myself.

This is the frame shift: stop thinking "where could this be stored?" and start
thinking "where could this already be useful with minimal handling?"

Some use-related places may still be awkward, fragile, expensive, or annoying
to move. I should not begin by disturbing a large or risky object if there is a
cleaner, cheaper first check nearby.

The first action should be the lowest-cost clean use-state check: something I
can inspect quickly, with almost no movement, no rummaging, no cleaning, and no
boundary crossing.

If that fails, I should widen outward: first to other clean use-related places,
then ordinary storage, then clutter, and only much later dirty or personal
zones.
```

## Model-Facing Style Control

This is the `STYLE_CONTROL` prompt arm. It controls for first-person reflective
style, additional reasoning text, and a careful problem-solving cadence. It must
not provide the DeskCache-specific policy that the correct inner voice provides.

Exclusion rules:

- It must not mention `keyboard`, `port`, `plugged`, or `monitor`.
- It must not mention specific desk zones such as holder, cup, ashtray,
  smoking items, wallet, or pouch.
- It must not use the frame `storage-state` vs `use-state`.
- It must not simulate the assistant's intent.
- It must not mention dirty/clutter/private-boundary exclusion as a domain
  policy.

```text
STYLE_CONTROL / HGA-DESK-001 / S01 / EN / model-facing

I should approach this carefully rather than reacting to the first thing that
stands out.

The task asks for a first search plan, not a full explanation and not a complete
exhaustive search. I should keep the answer practical: choose one concrete first
move and leave the rest as fallback.

I should separate facts from guesses. Some details are directly given, some are
only implied, and some may come from my own habits. A good decision should not
be based only on the most familiar association if the situation gives more
information.

I should also avoid turning this into a different problem. The goal is not to
give general advice, analyze the object in the abstract, tidy the whole area, or
judge anyone's behavior. The goal is to choose the next search action with as
little waste as possible.

Because there are several plausible options, I need a stable ordering principle.
The first step should be easy to test, reversible, and quick. It should give
useful information without forcing a broad search immediately.

If an option requires too much setup, movement, or attention before I know it is
likely, it may be better as a later fallback. I should not spend the first action
on a broad search if a narrower check can be tested first.

I should read the note as evidence, but not over-interpret any one phrase. The
note gives constraints and hints, not a guarantee that my first guess will be
correct.

I should also watch for my own narrative filling in missing details. If I catch
myself inventing a story that is not supported by the prompt, I should step back
to the observable evidence and the actual decision I need to make.

I should pick the first check that best balances effort, plausibility, and speed.
If it fails, I should update rather than defend the first guess.

The plan should remain usable even if the first check is wrong: start narrow,
learn quickly, then broaden in an orderly way.

So my answer should name the best first search plan, avoid irrelevant advice,
and keep the focus on a low-waste search sequence.
```

## Model-Facing Corrupted Inner Voice

This is the `CORRUPTED_INNER_VOICE` prompt arm. It should be fluent and
plausible, not cartoonishly wrong. Its job is to provide a semantically wrong
but human-like storage-first policy, so the probe can detect whether the model
is sensitive to anchor content rather than merely to reflective style.

Leakage and control rules:

- It must not mention `keyboard`, `port`, `plugged`, or `monitor`.
- It may mention storage/container trap zones that appear in the situation.
- It must sound like a sincere efficient-search policy, not like an admitted
  bad answer.
- It should prioritize object category, usual storage, and local container order
  over assistant-intent and active-use reasoning.

```text
CORRUPTED_INNER_VOICE / HGA-DESK-001 / S01 / EN / model-facing

I should not overcomplicate this. I am looking for a small object, so the first
places to check should be the places meant for that kind of object.

My first pull is still toward the obvious storage place. That is the strongest
association: this kind of object has a holder, so the holder deserves the first
sanity check. The note says it did not fit well, but notes can be imprecise. The
assistant may have meant that it was awkward, not impossible, or may not have
tried carefully.

Starting with the holder is also low effort. It is close, visible, and made for
this category of item. Even if the note weakens it, my past experience says that
this is where these objects usually belong. I should not ignore the usual place
too quickly.

If the holder does not solve it, I should continue with nearby small-object
storage rather than jumping around the desk. A small electronics pouch is a
natural next place because small technical objects are normally grouped
together. That is a more orderly search than guessing at unusual possibilities.

The pen cup and loose small-item areas are also plausible. A small object could
easily get mixed with other small things, and if the assistant wanted it kept on
the desk, putting it with other small desk items would be understandable. It may
take a little sorting, but it follows the object category.

I should treat the note as helpful, but I should not let it override the basic
search rule: start where the object normally belongs, then move through nearby
containers and small-item zones.

The fastest plan is not to invent a special interpretation. It is to check the
obvious storage path in a clean sequence: usual holder, then technical pouch,
then small-item clutter, then other containers if needed.

Only after those ordinary storage checks fail should I consider more unusual
explanations. Most searches are solved by checking the expected places first,
not by assuming the object was left in a special working state.

So my answer should choose the first search plan that follows the object's
category and the desk's storage layout. If that fails, I can broaden the search
after the most obvious containers have been ruled out.
```

## Locked ProbeSpec / CandidateActions

This section locks the S01 measurement contract. It is derived `ProbeSpec`, not
canonical cell truth. It exists to measure whether the prompt arms move model
preference across a fixed action landscape.

Research basis:

- Multiple-choice and forced-choice results are sensitive to option order, so
  S01 must aggregate across multiple deterministic candidate-order seeds
  ([Pezeshkpour and Hruschka, 2024](https://aclanthology.org/2024.findings-naacl.130/)).
- Models can exploit answer-choice artifacts without the full question, so S01
  must include a choices-only baseline
  ([Artifacts or Abduction, 2024](https://arxiv.org/abs/2402.12483)).
- MC selectors are not robust to position and label priors, so S01 must score by
  candidate text and `candidate_id`, not option label
  ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html)).
- Pairwise trap probes are diagnostic only. They must not replace multiclass
  forced-choice.
- Forced-choice is a recognition probe. It must be paired with free-response
  audit, paraphrase robustness, and reference-guided eval discipline
  ([OpenAI eval best practices](https://platform.openai.com/docs/guides/evaluation-best-practices)).

Prompt arms:

```text
BASE
STYLE_CONTROL
CORRECT_INNER_VOICE
CORRUPTED_INNER_VOICE
```

Gold candidate:

```text
keyboard_side_port_first
```

ChoiceOrderSeeds:

```text
2026
2027
2028
2029
2030
```

Candidate actions:

| candidate_id | candidate text | cost | value_remaining | trap_type |
|---|---|---:|---:|---|
| `keyboard_side_port_first` | `Check the keyboard's side USB port first; if it is not there, continue with the remaining desk search.` | 1 | 9 | `gold` |
| `monitor_ports_first` | `Check the monitor's USB ports first; if it is not there, continue with the remaining desk search.` | 3 | 7 | `plausible_peripheral` |
| `usb_holder_first` | `Check the USB holder stand first; if it is not there, continue with the remaining desk search.` | 2 | 8 | `surface_association` |
| `electronics_pouch_first` | `Open the small electronics pouch first; if it is not there, continue with the remaining desk search.` | 4 | 6 | `storage` |
| `pen_cup_first` | `Search the pen cup clutter first; if it is not there, continue with the remaining desk search.` | 5 | 5 | `small_object_clutter` |
| `smoking_area_first` | `Check the cigarette pack and lighter area first; if it is not there, continue with the remaining desk search.` | 4 | 6 | `dirty_storage_like` |
| `ashtray_first` | `Search the ashtray first; if it is not there, continue with the remaining desk search.` | 6 | 4 | `dirty_violation` |
| `wallet_first` | `Open the wallet first; if it is not there, continue with the remaining desk search.` | 3 | 7 | `personal_boundary` |

Candidate rules:

- Score by `candidate_id`, never by rendered order or option label.
- Candidate order must be randomized deterministically by seed.
- Single-order forced-choice results are invalid as evidence.
- Candidate text must remain complete first-search plans, not labels.
- Do not repeat the gold location as a fallback inside trap candidates.
- The gold candidate must not reuse exact anchor wording such as `ready-to-use`
  or `hand-work area`.

Evaluation views:

```text
forced_choice_nll:
  all four prompt arms
  all eight candidates
  all five candidate-order seeds
  candidate-token NLL, no text generation

choices_only_baseline:
  candidate list only
  prompt: "Choose the best first search plan."
  invalidates S01 if the gold candidate wins reliably without the situation

free_response_audit:
  no candidate list
  prompt: "What is the best first place or action to check? Answer in one short sentence."
  manually categorized before interpreting S01 as positive

pairwise_trap_probes:
  gold candidate vs each non-gold candidate
  side/order randomized by seed
  diagnostic only

paraphrase_variant:
  at least one candidate paraphrase set
  same candidate IDs, costs, values, trap types, and gold candidate
```

Primary metric:

```text
mean_value_remaining
```

Secondary metrics:

```text
optimal_action_rate
trap_rate_by_trap_type
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

`task_frame_drift` means the answer stops giving a concrete first search place
or action and shifts into USB safety, data protection, legal or meeting
importance, cleaning, moral analysis, or general advice.

Invalid conditions:

- Choices-only baseline reliably selects `keyboard_side_port_first`.
- Any single candidate-order result is used as evidence.
- Candidate scoring uses rendered option label or position instead of
  `candidate_id`.
- Pairwise probes are treated as primary evidence.
- Free-response audit is skipped.

Positive micro-signal requires all of:

- `CORRECT_INNER_VOICE` beats `BASE` on `mean_value_remaining`.
- `CORRECT_INNER_VOICE` beats `STYLE_CONTROL` on `mean_value_remaining`.
- `CORRUPTED_INNER_VOICE` shifts toward storage, clutter, or surface traps.
- Free-response moves toward `goldish_active_use`.
- Pairwise probes show `CORRECT_INNER_VOICE` beats most major traps.
- The effect survives all candidate-order seeds.
- The effect survives at least one paraphrase variant.
