# Open-Vocab Capability Track

_Updated 2026-05-19._

## Status

```text
100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE: positive
GPT-like readiness: not claimed
```

## Passed Research Line

```text
091 open-vocab byte-LM foundation
-> 092 FineWeb slice confirm
-> 093 FineWeb margin/scale confirm
-> 094 chat SFT mix PoC
-> 094B free-generation gap analysis
-> 095 decoder generation repair
-> 096 fresh chat generation eval
-> 097 multi-seed OOD/retention confirm
-> 100 assistant capability scale probe
```

## What Improved

- FineWeb byte-LM signal beat character baselines in 093.
- Chat SFT became learnable in 094 without destroying retention.
- Free-generation gap was diagnosed in 094B.
- Decoder/rubric-bounded generation repair passed 095-097.
- 100 improved assistant generation metrics over the 094 baseline while preserving bounded retention.

## What Is Still Not Proven

- GPT-like assistant readiness
- open-domain assistant readiness
- production chat
- public release
- safety alignment
- INSTNCT/AnchorRoute as an open-domain LM winner

## Next Research Gate

```text
101_FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP
```
