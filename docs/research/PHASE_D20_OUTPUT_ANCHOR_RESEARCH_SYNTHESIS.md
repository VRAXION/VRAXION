# Phase D20 Output-Anchor Research Synthesis

Date: 2026-05-02

## Summary

D18/D19 changed the blocker definition. The open question is no longer whether
the H384 graph can carry context at all. It can. The blocker is whether it can
carry context **without ceasing to behave like `top_01`** on ordinary
next-token behavior.

The deep-research conclusion is clear: D20 should preserve function-space
behavior, not parameter-space similarity. In practice, candidate search should
reward real-vs-fake context margin while anchoring ordinary output
distributions to `top_01`.

## Recommended D20 Direction

Use a non-context anchor set `A` and compare candidate output distributions to
`top_01` on the same histories:

```text
output_anchor = mean JS(p_candidate(.|x), p_top01(.|x)) for x in A
```

When full-vector comparison is too expensive, use teacher top-k plus candidate
top-k plus a tail bucket. Do not use top-1, target-only, or margin-only anchors
as the primary behavior-preservation metric.

D20 scoring should combine:

```text
context_margin_lcb
- output_anchor_penalty
- fake_context_penalty
- smooth/accuracy/unigram/echo safety penalties
```

The anchor set must exclude the special context probes where `top_01` is known
to be weak. It should include ordinary next-token histories, high-entropy
positions, low-margin positions, unigram-critical positions, and late rollout
states.

## Why This Helps

D19 found strong context margin, but the candidate paid for it with output
drift:

```text
margin_lower95 ~= +0.004974
smooth_delta   ~= -0.005243
accuracy_delta ~= -0.002281
unigram_delta  ~= -0.018994
```

That is exactly the failure an output-space anchor should reject. The intended
D20 behavior is:

```text
learn real sequential context
AND
remain top_01-like on normal prediction states
```

## Experiment Recommendation

1. Re-rank archived D18/D19 candidates offline with an output-anchor term.
2. Check whether output-anchor divergence separates D18-safe candidates from
   D19-safety-tradeoff candidates.
3. If the anchor separates them, add the anchor to the next context search loop.
4. If it does not separate them, redesign the anchor set before running more
   context search.

Stop rule:

```text
Do not run another unconstrained "maximize context margin" search.
Do not answer D20 failure by scaling H.
If output anchoring cannot predict safety failures, fix anchor coverage first.
```

## Status

Verdict: `D20_OUTPUT_ANCHOR_RESEARCH_READY`

Next implementation target: D20 output-anchored context search or an offline
archive reranker using `top_01` distribution anchors.
