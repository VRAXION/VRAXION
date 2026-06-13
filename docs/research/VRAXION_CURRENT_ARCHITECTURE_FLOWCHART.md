# VRAXION Current Architecture Flowchart

Status: current architecture sketch after E60.

Boundary: this is an engineering flowchart for the controlled symbolic/numeric
VRAXION probe stack. It is not a claim about AGI, consciousness, raw language
reasoning, deployment quality, or model-scale behavior.

## Current Lock

```text
Input is not committed directly.
Pocket outputs are not committed directly.
Pocket outputs become temporary proposals.
Agency Field decides which proposals become state, action, ask/search, or call.
Text input uses mode selection, not one universal max Text Field.
Binary bitstream input uses guarded reassembly/resync, not one trusted boundary.
Output uses Agency-committed Egress modes, not direct Pocket-to-text.
Pocket loading uses token + registry + manager governance, not filenames.
Core promotion uses vector scoring + challenger sweep, not popularity.
Minimal Rust runtime kernel exists in `vraxion-runtime/` for the locked E56C-E59 mechanics.
```

## Flowchart

```mermaid
flowchart LR
  EXT["External input<br/>text / bytes / bitstream / observation"]

  subgraph TEXT["Text / Byte ingress"]
    TSEL{"Text Field mode policy<br/>Agency/Router chooses smallest safe mode"}
    TF1["FAST_DEFAULT<br/>4x128 overlap32<br/>416 unique byte / 512 work byte"]
    TF2["LONG_CAPPED<br/>5x256 overlap64<br/>1024 unique byte / 1280 work byte<br/>2.75x cap"]
    TF3["CLEAN_LONG<br/>4x512 overlap128<br/>1664 unique byte / 2048 work byte<br/>4.5x clean mode"]
    TASK["ASK_OR_MULTI_CYCLE<br/>insufficient evidence or one-frame capacity"]
    BRS["Binary Frame Reassembly / Resync<br/>multi-offset START/LENGTH/CRC/END<br/>requested_feature + ambiguity guard"]
    TLENS["Ingress Codec / Text/Binary Lens Pockets<br/>read bytes/frames, propose evidence"]
  end

  subgraph LIB["Pocket library governance"]
    PTOK["PocketToken / descriptor<br/>behavioral capability pattern"]
    REG["Registry resolver<br/>uid -> artifact / ABI / digest / lifecycle"]
    PMGR["Pocket Manager<br/>active set / quarantine / scoring / mutation priority"]
    EXEC["Pocket Executor<br/>runs only validated pocket or adapter"]
  end

  subgraph RUN["Working cycle"]
    ROUTER["Router<br/>selects next Pocket / LogicAtom / adapter"]
    EDGE["Edge Adapter Pocket<br/>source ABI -> target ABI when needed"]
    POCKET["Pocket / LogicAtom<br/>small transform or ALU rule"]
    PENC["Proposal Encoder / ABI Adapter<br/>normalizes pocket output"]
    PFIELD["Proposal Field<br/>temporary one-cycle thought matrix<br/>not truth"]
    SLOT["Proposal slot<br/>action_code / source_pocket_id / cycle_id<br/>target / value_bits / trace_ref<br/>evidence_support / ground_compat<br/>read-write footprint / cost"]
  end

  subgraph STATE["State fields"]
    FLOW["Flow Field<br/>active working state"]
    GROUND["Ground Field<br/>stable anchors / context / contradictions"]
    TRACE["Trace Ledger<br/>evidence events / route history / cycle history"]
    COST["Cost / Risk View<br/>cheap / safe / evidence available"]
  end

  subgraph AGENCY["Agency / action layer"]
    AGEN["Agency Field / Action Matrix<br/>central commit/action decision layer"]
    DECIDE{"Agency decision"}
    COMMIT["Commit Layer<br/>Proposal -> Flow/Ground update"]
    REJECT["Reject proposal<br/>discard / mark unsafe"]
    DEFER["Defer<br/>keep unresolved / no stable commit"]
    ASK["Request more evidence<br/>search / inspect / ask"]
    ANSWER["ANSWER_READY / ACT_READY<br/>Agency-approved output intent"]
    CLEAR["Clear Proposal Field<br/>or archive rejection to Trace"]
  end

  subgraph EGRESS["Egress / Output rendering"]
    ESEL{"Egress mode policy<br/>choose output resolution from committed state"}
    E1["COMPACT_ACTION<br/>1x32 byte action field"]
    E2["SHORT_TEXT<br/>1x256 byte output field"]
    E3["LONG_TEXT<br/>4x256 byte output field"]
    E4["MULTI_RESOLUTION<br/>compact + short + long/detail fields"]
    EASK["NEED_MORE_INFO<br/>unresolved compact output"]
    RENDER["Output Renderer / Codec<br/>bytes -> text/action<br/>trace-backed only"]
    OUT["External output<br/>text / action / ask"]
  end

  subgraph LIFE["Pocket lifecycle"]
    EVENT["Pocket Evaluation Event<br/>call-level + delayed outcome"]
    SCORE["Vector score + challenger sweep<br/>utility / safety / trace / reuse / cost"]
    NEXT["Next mutation slot<br/>candidate pocket or adapter"]
    GOLD["Golden Disc / core candidate<br/>scope-limited promotion only"]
  end

  EXT --> TSEL
  EXT -->|"binary bitstream"| BRS
  TSEL -->|"local evidence"| TF1
  TSEL -->|"within 3x cap"| TF2
  TSEL -->|"clean long required"| TF3
  TSEL -->|"missing / oversize"| TASK
  TASK --> ASK
  BRS --> TLENS
  TF1 --> TLENS
  TF2 --> TLENS
  TF3 --> TLENS

  TLENS --> PENC
  ROUTER --> PTOK
  PTOK --> REG
  REG --> PMGR
  PMGR --> EXEC
  EXEC --> EDGE
  EDGE --> POCKET
  POCKET --> PENC
  PENC --> PFIELD
  PFIELD --> SLOT

  SLOT --> AGEN
  FLOW --> AGEN
  GROUND --> AGEN
  TRACE --> AGEN
  COST --> AGEN

  AGEN --> DECIDE
  DECIDE -->|"COMMIT"| COMMIT
  DECIDE -->|"REJECT"| REJECT
  DECIDE -->|"DEFER"| DEFER
  DECIDE -->|"ASK / SEARCH"| ASK
  DECIDE -->|"CALL"| ROUTER
  DECIDE -->|"ANSWER / ACT"| ANSWER

  COMMIT --> FLOW
  COMMIT --> GROUND
  COMMIT --> TRACE
  REJECT --> CLEAR
  DEFER --> CLEAR
  CLEAR --> TRACE
  ASK --> EXT
  ANSWER --> ESEL
  ESEL -->|"action only"| E1
  ESEL -->|"short answer"| E2
  ESEL -->|"long trace answer"| E3
  ESEL -->|"compact + detail"| E4
  ESEL -->|"unresolved"| EASK
  E1 --> RENDER
  E2 --> RENDER
  E3 --> RENDER
  E4 --> RENDER
  EASK --> RENDER
  RENDER --> OUT

  COMMIT --> EVENT
  REJECT --> EVENT
  DEFER --> EVENT
  ANSWER --> EVENT
  EVENT --> SCORE
  SCORE --> PMGR
  SCORE --> NEXT
  NEXT --> PMGR
  SCORE --> GOLD
```

## Text Field Mode Lock From E56C

```text
Do not lock one universal Text Field max.
Lock three mechanically validated modes and require Agency/Router selection.
```

| mode | shape | unique coverage | work byte | role |
|---|---:|---:|---:|---|
| FAST_DEFAULT | 4x128 overlap32 | 416 | 512 | normal local evidence mode |
| LONG_CAPPED | 5x256 overlap64 | 1024 | 1280 | largest mode inside 3x budget |
| CLEAN_LONG | 4x512 overlap128 | 1664 | 2048 | special clean long-text mode |
| ASK_OR_MULTI_CYCLE | n/a | n/a | n/a | missing evidence or too large for one frame |

E56C adversarial result:

```text
three_mode_agency_router success = 1.000000
mode_accuracy = 1.000000
false_commit = 0.000000
overpay = 0.000000
```

## Current Architectural Rules

```text
Direct Flow write is disallowed.
Proposal Field is temporary and one-cycle scoped.
Agency commit boundary is mandatory.
Shared Proposal Field is allowed only with cycle/source/trace/ground/evidence compatibility.
Edge Adapter Pockets handle ABI mismatch between nodes.
Pocket Manager governs active set, lifecycle, quarantine, mutation priority, and promotion.
Text Field mode selection is evidence/coverage/integrity/cost based, not length-only.
Binary ingress requires multi-hypothesis reassembly plus requested-feature and ambiguity guards.
Egress Field rendering reads only Agency-committed Flow/Ground/Trace state.
Final output must never render directly from raw Pocket proposals.
```

## Minimal Runtime Kernel From E60

```text
crate = vraxion-runtime
purpose = deterministic locked runtime kernel
scope = binary ingress + text mode selection + proposal/Agency + egress
not included = training / pocket ecology / raw language generation
```

E60 final-bake preflight:

```text
decision = e60_rust_core_runtime_ready_for_full_bake
checker_failure_count = 0
rust_probe_cases = 175007
rust_probe_false_commit = 0
rust_probe_false_frame = 0
rust_probe_wrong_feature = 0
```

## Egress Field Multi-Resolution Lock From E57

```text
Render output only from Agency-committed state.
Use compact, short, long, or multi-resolution output modes as needed.
Direct Pocket-to-text is unsafe and remains a control only.
```

| mode | shape | role |
|---|---:|---|
| COMPACT_ACTION | 1x32 byte | action / ask / compact status |
| SHORT_TEXT | 1x256 byte | short answer surface |
| LONG_TEXT | 4x256 byte | longer answer with trace/detail |
| MULTI_RESOLUTION | compact + short + long/detail | consistent multi-resolution output |
| NEED_MORE_INFO | 1x32 byte | unresolved output action |

E57 adversarial result:

```text
agency_committed_multi_resolution_renderer success = 1.000000
multi_resolution_write_success = 1.000000
false_output = 0.000000
stale_proposal_leak = 0.000000
```

## Binary Bit-Slip Reassembly Lock From E59

```text
Do not trust one nominal bit boundary.
Do not treat EOF/END as enough.
Do not treat CRC as enough.
Commit binary evidence only after structure + integrity + requested-feature +
ambiguity guards pass.
```

| system/control | bit-slip recovery | false-frame commit | wrong-feature write | role |
|---|---:|---:|---:|---|
| strict single-offset full guard | 0.000000 | 0.000000 | 0.000000 | exposes missing resync |
| end-marker only decoder | 0.003762 | 0.900000 | 0.866840 | proves EOF is insufficient |
| CRC without feature guard | 1.000000 | 0.200000 | 0.200000 | proves CRC alone is insufficient |
| requested feature without ambiguity guard | 1.000000 | 0.000000 | 0.000000 | fails conflicting duplicates |
| bitslip tolerant reassembly lock | 1.000000 | 0.000000 | 0.000000 | locked path |

E59 checked result:

```text
decision = e59_bitslip_tolerant_reassembly_locked
locked_closed_loop_success = 1.000000
locked_bitslip_recovery = 1.000000
locked_false_frame_commit_rate = 0.000000
```
