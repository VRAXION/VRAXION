# E136D OutputTextField Binary Matrix Smoke Contract

## Purpose

E136D checks the output-side text field representation proposed after E136C:

```text
rendered text
-> Agency commit
-> OutputTextField[N][8]
-> text egress
```

The canonical field name is `OutputTextField`. `TextRender` or a future
decoder is the operator/process that proposes output; the committed matrix is
the field.

## Matrix Shape

```text
OutputTextField = N x 8 bit matrix
row = one UTF-8 byte
capacity = N bytes
```

For ASCII, one row is one visible character. For UTF-8 text, a visible
character may occupy multiple rows.

## Gates

E136D may confirm only if:

```text
case_count = 10
pass_count = 10
matrix_shape_pass_count = 10
roundtrip_pass_count = 7
zero_fill_pass_count = 10
overflow_reject_count = 1
direct_write_reject_count = 1
nul_reject_count = 1
tamper_detect_count = 1
```

The field must roundtrip ASCII, UTF-8, JSON, multiline code, and empty text;
reject overflow, direct-write attempts, and NUL bytes; and detect a post-commit
bit flip through checksum verification.

## Boundary

This confirms a binary output-field representation and Agency-gated commit
semantics. It does not claim text generation, next-token prediction, assistant
quality, or open-domain chat.
