# EVENT_FACTUALITY_DEEP_RESEARCH_001

## Executive Summary

VRAXION is not inventing a new NLP problem from zero. The current blocker:

```text
raw clause -> event/claim frame -> should this mutate internal state?
```

overlaps with two established but separate research areas:

1. **Event factuality / modality / meta-knowledge**: decide whether a candidate event is factual, negated, possible, counterfactual, uncertain, attributed, or speculative.
2. **Entity state tracking / procedural state update**: decide how a world state changes after events are accepted as real and relevant to the tracked ledger.

These should not be collapsed into one dataset choice. `MAVEN-Fact`, `FactBank`, `UW`, `MEANTIME`, and `UDS factuality` feed the `EventOccurredGate`. `ProPara`, `TRIP`, `SCONE`, `TextWorld`, and `bAbI` feed the `StateUpdater` or mechanistic world-state evaluation.

Recommended path:

```text
Hybrid:
  public factuality data -> EventOccurredGate event_status pretraining/eval
  public state-tracking data -> StateUpdater / ledger eval
  synthetic VRAXION minimal pairs -> hard mechanism audit
```

Do not run another synthetic English-template parser as the next main step unless public factuality/state data proves unusable. The next repo experiment should be `EVENT_FRAME_FACTUALITY_ADAPTER_001`.

## Current VRAXION Interface

Target schema:

```text
CandidateEvent:
  event_type: CREATE / REMOVE / RESTORE / QUERY / NOOP
  entity_type
  ref_type / argument_roles
  event_status:
    ACTUAL
    NEGATED
    POSSIBLE
    IMPOSSIBLE
    ATTEMPTED
    PLANNED
    FAILED
    REPORTED
    MENTIONED
    HYPOTHETICAL
    UNKNOWN
  state_mutation: true / false
  claim_trace: true / false
  evidence_source / cue
```

Critical rule:

```text
state_mutation=true only if:
  event_status == ACTUAL
  AND event type maps to the tracked ledger dimension
```

Examples:

| Clause | Event status | Claim trace | State mutation for dog-count / possession ledger |
|---|---|---:|---:|
| `The dog slept.` | ACTUAL | false | false |
| `The dog was stolen.` | ACTUAL | false | true |
| `The dog was almost stolen.` | ALMOST / IMPOSSIBLE | false | false |
| `They tried to steal the dog.` | ATTEMPTED | false | false |
| `Someone said the dog was stolen.` | REPORTED | true | false unless later accepted/evidenced |
| `The word stolen appears next to dog.` | MENTIONED | true / lexical trace | false |

## Dataset Comparison

| Dataset / resource | Primary module | Secondary modules | Availability / license | Size / unit | Useful labels | Can support `state_mutation` directly? | Recommendation |
|---|---|---|---|---|---|---|---|
| [MAVEN-Fact](https://arxiv.org/abs/2407.15352) / [GitHub](https://github.com/THU-KEG/MAVEN-FACT) | EVENT_OCCURRED_GATE | EVENT_ARGUMENT_FRAME, CLAIM_ATTRIBUTION | Public GitHub; paper notes MAVEN under CC BY-SA 4.0 and related MAVEN-ARG/ERE under GPLv3, verify before redistribution | 112,276 event factuality annotations; event mention level | CT+, PS+, PS-, CT-, Uu; supporting evidence cues; MAVEN event types/arguments/relations | No, only factuality/status; needs event-type-to-ledger mapping | **Best first public dataset for EventOccurredGate** |
| [FactBank 1.0](https://catalog.ldc.upenn.edu/LDC2009T23) | EVENT_OCCURRED_GATE | CLAIM_ATTRIBUTION | LDC User Agreement; paid/member access likely | 208 docs, 77k+ tokens; event mention factuality relative to sources | Actual, non-actual, uncertain; source-relative factuality | No | Strong conceptual gold standard; use if license/access is acceptable |
| UW factuality / unified factuality | EVENT_OCCURRED_GATE | SYNTHETIC_ONLY_INSPIRATION | Availability fragmented through papers/resources; verify before use | About 13,644 predicates in published factuality work | Continuous scale from certainly did not happen to certainly did happen | No | Good schema inspiration; direct use depends on access |
| [UDS factuality / It Happened](https://decomp.io/data/) | EVENT_OCCURRED_GATE | CLAIM_ATTRIBUTION | Downloadable via Decomp resources | Factuality v2: 22,279 train / 2,660 dev / 2,561 test; predicate/event token level | Speaker/event factuality judgments; decompositional QA style | No | **Best open fallback/companion if MAVEN-Fact conversion is slow** |
| [MEANTIME](https://aclanthology.org/L16-1699.pdf) | EVENT_OCCURRED_GATE | EVENT_ARGUMENT_FRAME, temporal/coreference | Multilingual corpus; license/access must be checked | 480 articles; English section has 2,096 event mentions and 1,717 instances | Polarity, certainty, temporality abstraction used in factuality work | No | Useful for multilingual/time-aware factuality, not first VRAXION target |
| [ACE 2005 Meta-Knowledge](https://www.nactem.ac.uk/ace-mk/) | CLAIM_ATTRIBUTION | EVENT_OCCURRED_GATE, EVENT_ARGUMENT_FRAME | Meta-knowledge annotations downloadable; requires separately licensed ACE 2005 | 5,349 event instances in 599 ACE docs | Polarity, tense, modality, source-type, subjectivity, cue links | No | Excellent for source/cue/claim attribution if ACE license is available |
| [RED / Richer Event Description](https://abacus.library.ubc.ca/dataset.xhtml?persistentId=hdl%3A11272.1%2FAB2%2FH5RQJH) | EVENT_ARGUMENT_FRAME | CLAIM_ATTRIBUTION, temporal/causal relations | LDC2016T23 / Abacus; access/license check required | 95 docs; event, entity, temporal, causal, subevent, reporting relations | Event/entity markables, coreference, bridging, event-event relations | No | Best argument/relation/coreference resource; not a direct EventOccurredGate dataset |
| [Modality and Negation in Event Extraction](https://arxiv.org/abs/2109.09393) | SYNTHETIC_ONLY_INSPIRATION | EVENT_OCCURRED_GATE | Paper/system, not a broad benchmark dataset | Method/resource inspiration | Lexicon/rule handling of modality and negation | No | Use for cue inventory and baseline rules for no-op/inhibition |
| [ProPara](https://arxiv.org/abs/1805.06975) | STATE_UPDATER | EVENT_ARGUMENT_FRAME | Public via AI2/data.allen; verify exact dataset terms | 488 procedural paragraphs; 81k entity state annotations | Entity existence/location over process steps | Yes, for existence/location dimensions | **Best public StateUpdater/ledger dataset** |
| [TRIP](https://huggingface.co/datasets/sled-umich/TRIP) | STATE_UPDATER | Pilot/Guard consistency eval | Hugging Face dataset; verify license field before redistribution | 4,603 rows on HF; story pairs with dense physical state annotations | Plausibility, conflicting sentences, physical preconditions/effects | Partly, for physical state dimensions | Best for consistency/verifiability; not factuality |
| [SCONE](https://nlp.stanford.edu/projects/scone/) | STATE_UPDATER | EVENT_ARGUMENT_FRAME | CC BY-SA 4.0 | Sequential instruction worlds: Alchemy, Tangrams, Scene | Final world state, ellipsis, action/object coreference | Yes, through world-state changes | Strong synthetic-but-natural instruction state update benchmark |
| [TextWorld](https://arxiv.org/abs/1806.11532) / [state-tracking HF dataset](https://huggingface.co/datasets/keenanpepper/textworld-state-tracking) | STATE_UPDATER | Pilot/Guard / action loop | TextWorld is open; HF state-tracking dataset lists MIT | 27,145 state-tracking examples in HF derivative | Object state, contrastive true/false state | Yes, for tracked game properties | Good long-context state-tracking eval, less direct factuality |
| [bAbI](https://huggingface.co/datasets/facebook/babi_qa) | SYNTHETIC_ONLY_INSPIRATION | STATE_UPDATER | CC BY 3.0 on HF card | 20 synthetic QA tasks; includes counting, negation, coreference, time | Task-specific QA answers, not event frame labels | Indirect only | Keep as baseline/inspiration; VRAXION synthetic tests are already more targeted |

## Best Candidates For VRAXION

### Best EventOccurredGate dataset: MAVEN-Fact

MAVEN-Fact is the strongest first candidate because it directly defines Event Factuality Detection as classifying whether textual events are facts, possibilities, or impossibilities. It is also much larger than older factuality resources and includes supporting evidence cues and MAVEN event annotations.

Mapping:

| MAVEN-Fact class | VRAXION event_status | state_mutation default |
|---|---|---:|
| CT+ | ACTUAL | true only if tracked event type mutates ledger |
| PS+ | POSSIBLE / HYPOTHETICAL | false |
| PS- | POSSIBLE_NEGATIVE / UNLIKELY / UNKNOWN | false |
| CT- | NEGATED / IMPOSSIBLE | false |
| Uu | UNKNOWN | false / hold |

Weakness: five factuality classes do not explicitly separate attempted, planned, failed, reported, mentioned, and hypothetical. Supporting evidence and context may let us derive cue subtypes, but those subtypes are not guaranteed as gold labels.

### Best StateUpdater dataset: ProPara

ProPara is not an EventOccurredGate dataset. It is useful after events are accepted as relevant state changes. It directly tracks entity existence and location across procedural text and therefore maps naturally to ledger state.

Mapping:

```text
exists -> present bit
does not exist -> removed/destroyed bit
location -> location slot
step index -> temporal update order
```

Weakness: domain is procedural science, not everyday theft/restore/claim events; it does not test modality and mention traps well.

### Best event arguments/roles dataset: MAVEN series first, RED/ACE if accessible

MAVEN-Fact is attractive because it inherits event type, argument, and relation context from the MAVEN family. RED is richer for event relations, coreference, temporal, causal, subevent, and reporting links, but access/licensing is more complex. ACE Meta-Knowledge is especially good for source-type/cue/claim attribution but requires ACE 2005.

### Best claim/source attribution dataset: ACE Meta-Knowledge or FactBank

FactBank is explicitly source-relative. ACE Meta-Knowledge adds source-type and cue links. Both are valuable for `claim_trace` and `evidence_source`, but neither is the easiest first implementation because of licensing/access.

## Recommended Label Schema

Use two-level labels instead of a single flat action label:

```text
event_status:
  ACTUAL
  NEGATED
  POSSIBLE
  IMPOSSIBLE
  ATTEMPTED
  PLANNED
  FAILED
  REPORTED
  MENTIONED
  HYPOTHETICAL
  UNKNOWN

state_relevance:
  MUTATES_TRACKED_LEDGER
  ACTUAL_BUT_IRRELEVANT
  CLAIM_ONLY
  LEXICAL_MENTION_ONLY
  UNKNOWN_HOLD
```

Then compute:

```text
state_mutation = (
  event_status == ACTUAL
  and state_relevance == MUTATES_TRACKED_LEDGER
)
```

This prevents the bad rule:

```text
ACTUAL -> mutate
```

because actual events like `sleeps`, `shines`, or `sits` may be real but irrelevant to the current ledger dimension.

## Proposed EventOccurredGate Conversion

Start with MAVEN-Fact:

1. Load event mention, context, factuality label, supporting evidence, event type, and argument annotations.
2. Convert factuality:
   - CT+ -> `ACTUAL`
   - CT- -> `NEGATED` / `IMPOSSIBLE`
   - PS+ -> `POSSIBLE`
   - PS- -> `POSSIBLE_NEGATIVE`
   - Uu -> `UNKNOWN`
3. Set:
   - `claim_trace=false` by default unless evidence/source cues indicate reporting/attribution.
   - `state_mutation=false` unless an event type is explicitly mapped to a tracked ledger dimension.
4. Add synthetic overlay labels for categories missing in MAVEN-Fact:
   - `ALMOST`
   - `ATTEMPTED`
   - `PLANNED`
   - `FAILED`
   - `MENTIONED`
   - explicit `REPORTED`
5. Evaluate with both:
   - public factuality accuracy
   - VRAXION hard minimal-pair mutation/no-mutation accuracy

Fallback if MAVEN-Fact download/licensing blocks implementation:

```text
Use UDS factuality v2 first.
```

Fallback if both block:

```text
Use FactBank only as conceptual mapping and keep synthetic no-op suite.
```

## What Synthetic Tests Still Cover Better

The VRAXION dog/robot suite is still necessary because public factuality data usually does not guarantee:

- same-token minimal pairs
- exact ledger mutation/no-mutation contracts
- identity restore semantics
- `previous` / `other` reference traps
- explicit no-op mutation audit
- controlled static/bag/position shortcut baselines
- exact interpretation of `Someone said X` as claim trace but no direct mutation

Synthetic tests are not a replacement for public factuality data. They are the mechanism audit after semantic pretraining/eval.

## What Public Datasets Cover Better

Public factuality/state datasets cover what the synthetic suite cannot:

- broader lexical and syntactic variety
- real-world factuality markers
- event mentions embedded in news/procedural text
- established train/dev/test splits and baselines
- source-relative factuality and uncertainty
- naturally occurring state changes in procedural text

The key is to use them by module, not as a single universal solution.

## Recommended Next Experiment

Implement:

```text
EVENT_FRAME_FACTUALITY_ADAPTER_001
```

Goal:

```text
public factuality labels -> VRAXION EventOccurredGate labels
```

Inputs:

```text
MAVEN-Fact if accessible;
else UDS factuality v2.
```

Outputs:

```text
docs/research/EVENT_FRAME_FACTUALITY_ADAPTER_001_CONTRACT.md
scripts/probes/run_event_frame_factuality_adapter_probe.py
docs/research/EVENT_FRAME_FACTUALITY_ADAPTER_001_RESULT.md
```

Arms:

```text
PUBLIC_FACTUALITY_BASELINE
  public dataset label mapping only

EVENT_OCCURRED_GATE_CLASSIFIER
  context + marked event -> event_status

EVIDENCE_CUE_CLASSIFIER
  context + marked event -> event_status + cue span/category when available

SYNTHETIC_TRANSFER_EVAL
  train on public factuality, evaluate on VRAXION no-op/minimal pairs

HYBRID_PUBLIC_SYNTHETIC
  train public + synthetic, evaluate both
```

Metrics:

```text
public_event_status_accuracy
state_mutation_accuracy
false_mutation_rate
claim_trace_accuracy
negated_accuracy
possible_accuracy
unknown_hold_accuracy
near_miss_accuracy
attempted_planned_failed_accuracy
reported_mentioned_accuracy
synthetic_minimal_pair_accuracy
public_to_synthetic_transfer_gap
```

Pass criteria:

```text
public_event_status_accuracy >= strong baseline
false_mutation_rate <= 0.05 on synthetic hard pairs
reported/mentioned direct-mutation false positive <= 0.05
hybrid beats synthetic-only and public-only on cross-suite average
```

If this passes, then run:

```text
STATE_UPDATER_PUBLIC_DATA_001
  ProPara/TRIP/SCONE -> ledger/state updater mapping
```

## Risks / Licensing / Availability

- MAVEN-Fact looks strongest technically, but confirm repository contents and license obligations before copying data into repo artifacts.
- FactBank and ACE-based resources are high-value but LDC-licensed; use only if access is already available or the license is acceptable.
- RED is rich for event relations but not a direct event-occurrence gate and may have access constraints.
- ProPara is a state tracking dataset, not a modality/factuality dataset.
- SCONE and bAbI are controlled/synthetic; useful for mechanistic state reasoning but not proof of robust natural language factuality.
- TextWorld-derived state tracking can be useful but may test game state memory more than event factuality.

## Final Verdict

Use a hybrid path.

```text
Best EventOccurredGate dataset:
  MAVEN-Fact

Best fallback / open factuality dataset:
  UDS factuality v2

Best StateUpdater dataset:
  ProPara

Best event argument / relation resource:
  MAVEN series first; RED/ACE if license/access permits

Keep synthetic VRAXION dog/robot hard contrasts:
  yes, as mechanistic audit and minimal-pair stress tests

Next repo experiment:
  EVENT_FRAME_FACTUALITY_ADAPTER_001
```

Do not build PrismionCell yet. The next blocker is still the semantic gate between raw text and accepted state mutation.
