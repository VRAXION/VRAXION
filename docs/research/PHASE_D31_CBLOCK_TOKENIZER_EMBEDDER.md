# Phase D31 - C-Block Tokenizer / Embedder / Controller

## Status

```text
D31_CBLOCK_ARCHITECTURE_LOCK
```

D31 names and locks the next component boundary after the AB codec and the
route-selected worker stack.

This is an architecture/design lock, not an implementation result yet.

## Current Stack

The cleaned component stack is:

```text
raw bytes
  |
  v
A-block
  1 byte <-> 16D byte abstract
  |
  v
B-block
  N x A output -> B latent
  default: 8 x 16D = 128D -> 64D
  |
  v
C-block
  stream tokenizer / span embedder / controller
  |
  v
D-block workers
  ALU / MEM / TRANSFORM / LANG / UNKNOWN policy
```

The currently frozen AB artifact is:

```text
8 bytes <-> A128 <-> B64 <-> A128 <-> 8 bytes
```

So the default C-block input is not raw bytes and not A128:

```text
C-block input: stream of B64 windows
```

## Naming

```text
A-block:
  byte codec
  one byte to/from 16D abstract lanes

B-block:
  window codec / common bus
  N A-block outputs to/from one B latent
  default 8 bytes -> 64D B latent

C-block:
  tokenizer + span embedder + controller
  turns a B64 window stream into token events and route hints

D-block:
  selected workers / executable lanes
  ALU, memory, transform, language, reject/no-op policy
```

D28 should now be understood as:

```text
C0 route-head probe
```

It proved that single B64 windows contain enough information for coarse route
selection. It is not the full C-block.

## Why C Exists

B64 is an 8-byte window surface. Real input is a stream. Important expressions
can cross window boundaries:

```text
Give me apples, i need EXACTLY 25 times 7...
```

Naive 8-byte chunks can split the useful expression:

```text
[EXACTLY]
[ 25 tim]
[es 7...]
```

The C-block must convert overlapping B64 windows into stable spans:

```text
NUMBER(25)
OP_MUL("times")
NUMBER(7)
ROUTE(ALU)
```

This is why C is not just another compression layer. C is the first stream
understanding layer.

## C-Block Responsibilities

The v1 C-block must do four things:

```text
1. Local feature detection
   digit / letter / whitespace / punctuation / operator-word patterns

2. Span state
   track whether the stream is inside a number, word, command, or unknown span

3. Token emission
   emit NUMBER, WORD, OP, COMMAND, ROUTE_HINT, UNKNOWN events

4. Route control
   decide which D worker should receive the completed span or command
```

## Proposed Internal Shape

```text
B64_t
  |
  v
C1 local sparse feature layer
  |
  v
C2 recurrent span state
  |
  v
C3 token emitter
  |
  v
C4 route hint / worker dispatch metadata
```

Default v1 dimensions:

```text
input per tick:       64D B latent
local feature width:  64D or 96D sparse
span state width:     128D sparse recurrent
token embedding:      64D
route labels:         LANG, ALU, MEM, TRANSFORM, UNKNOWN
```

This keeps the system sparse and inspectable. The goal is not a dense
transformer tokenizer. The goal is a small, adversarially checked stream block
that can be searched, pruned, and controlled.

## First Target Behavior

Input:

```text
Give me apples, i need EXACTLY 25 times 7...
```

Sliding windows:

```text
[EXACTLY ]
[ XACTLY 2]
[ACTLY 25]
[CTLY 25 ]
[TLY 25 t]
[LY 25 ti]
[Y 25 tim]
[ 25 time]
[25 times]
[5 times ]
[ times 7]
[times 7.]
```

Expected C output:

```text
TEXT_SPAN("Give me apples, i need EXACTLY")
NUMBER(25)
OP_MUL
NUMBER(7)
ROUTE(ALU)
```

D-block then executes:

```text
ALU_MUL(25, 7)
```

Scope caveat:

```text
D30B currently returns bytewise/mod256 ALU output.
25 * 7 -> 175 is inside byte range and is fine.
27 * 852 still returns 220 until decimal/full-integer output is implemented.
```

## D31 Implementation Probe

The next implementation phase should be:

```text
D31A_STREAM_TOKENIZER_PROBE
```

Suggested tasks:

```text
number extraction:
  "abc 25 def" -> NUMBER(25)

operator word normalization:
  "25 times 7" -> NUMBER(25), OP_MUL, NUMBER(7)

symbol operator normalization:
  "25*7" -> NUMBER(25), OP_MUL, NUMBER(7)

route hint:
  completed arithmetic span -> ROUTE(ALU)

boundary stress:
  expression split across B64 windows must still parse
```

Suggested controls:

```text
window shuffle:
  breaks order and should fail

label shuffle:
  should fail near chance

random B64 projection:
  should fail

chunk-boundary adversary:
  "25 tim" + "es 7" must still become OP_MUL only when the stream order is real

ambiguous text:
  "THE+CAT" should stay UNKNOWN or TEXT, not ALU
```

## Long-Horizon Meaning

D31 is the first layer that can turn the current component stack from
clean-but-local workers into a stream-capable system:

```text
A/B:
  exact byte/window codec

C:
  turns B64 windows into tokens, spans, commands, and route hints

D:
  executes selected worker lanes
```

This is the path toward usable input handling. Without C, the system only works
on short, command-shaped 8-byte windows. With C, it can start handling real
text streams where the useful command is embedded inside ordinary language.

## Boundaries

Not claimed in D31:

```text
general language understanding
full decimal arithmetic formatting
learned semantic reasoning
release-ready AI
```

Claimed in D31:

```text
The C-block name, responsibility, input/output contract, and first test target
are now defined.
```
