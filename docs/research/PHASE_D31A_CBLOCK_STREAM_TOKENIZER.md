# Phase D31A - C-Block Stream Tokenizer / Embedder Probe

## Verdict

```text
D31A_CBLOCK_TOKENIZER_PASS
```

D31A implements the first C-block probe over the frozen AB/B64 surface:

```text
raw bytes -> sliding 8-byte AB/B64 windows -> C tokenizer -> TokenEvent stream
```

It is a tokenizer/embedder/controller probe, not a language worker.

## Main Result

Run shape:

```text
mode: main
samples: 8,192
boundary_cases: 4,096
total samples: 12,288
control_repeats: 8
artifact: tools/ab_window_codec_v1.json
```

Result:

```text
token_stream_exact_acc:   100%
token_boundary_acc:       100%
token_kind_acc:           100%
normalized_acc:           100%
alu_call_exact_acc:       100%
boundary_case_exact_acc:  100%
```

Controls:

```text
max_control_token_exact:         14.26%
max_control_alu_positive_exact:   8.76%

window_shuffle:
  token exact: ~0.05% to 0.10%

random_b64_projection:
  token exact: 0.00%

label_shuffle:
  token exact: ~13.79% to 14.26%
```

The all-sample `max_control_alu_exact` is higher because many non-ALU samples
correctly have no ALU call. The gate uses positive ALU samples:

```text
max_control_alu_positive_exact: 8.76%
```

## Integration Smoke

```text
25 times 7
  -> NUMBER:25 | OP:OP_MUL | NUMBER:7
  -> 25 OP_MUL 7 -> 175

25*7
  -> NUMBER:25 | OP:OP_MUL | NUMBER:7
  -> 25 OP_MUL 7 -> 175

Give me exactly 25 times 7.
  -> WORD:GIVE | WORD:ME | WORD:EXACTLY | NUMBER:25 | OP:OP_MUL | NUMBER:7 | PUNCT:.
  -> 25 OP_MUL 7 -> 175
```

## TokenEvent V1

Each emitted token has:

```text
kind:
  WORD / NUMBER / OP / PUNCT / NEWLINE / UNKNOWN

payload:
  raw_text
  normalized
  start_byte
  end_byte
  route_hint
  source_window_start
  source_window_end
  c64_embedding
```

Whitespace policy:

```text
spaces/tabs:
  boundary metadata, not normal output tokens

punctuation:
  explicit PUNCT tokens

newline:
  explicit NEWLINE token
```

## C64 Embedding Layout

```text
0..7:
  token kind lanes

8..15:
  route lanes

16..31:
  operator / punctuation / numeric feature lanes

32..63:
  stable sparse signed hash lanes for normalized token text
```

## Interpretation

D31 locked the C-block contract:

```text
B64 stream -> tokens + embeddings + route hints
```

D31A proves that the current B64 surface can support that contract on generated
word, punctuation, number, operator, and boundary-stress examples.

This moves the ABCD stack from short command windows toward stream-capable
input handling:

```text
A = byte codec
B = B64 window bus
C = stream tokenizer / token embedder
D = selected workers
```

## Caveats

This is still a probe/reference implementation:

```text
not a learned language model
not a full semantic tokenizer
not decimal/full-integer arithmetic formatting
```

ALU output is still bytewise/mod256 through D30B. The showcased `25 * 7 -> 175`
is exact because it fits inside one byte.

## Next Step

```text
D31B: C-token stream -> route-selected execution episodes
```

Examples:

```text
Give me exactly 25 times 7.
  -> C tokens
  -> ALU_CALL(MUL, 25, 7)
  -> D30B ALU output 175

Text before and after command spans must remain LANG/TEXT metadata, not worker input.
```

## Artifacts

Generated outputs:

```text
output/phase_d31a_cblock_stream_tokenizer_20260503/smoke/
output/phase_d31a_cblock_stream_tokenizer_20260503/main/
output/phase_d31a_cblock_stream_tokenizer_20260503/integration_smoke/
```

Tracked implementation:

```text
tools/_scratch/d31a_cblock_stream_tokenizer_probe.py
```
