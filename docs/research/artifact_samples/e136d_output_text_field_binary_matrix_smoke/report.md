# E136D OutputTextField Binary Matrix Smoke

```text
decision = e136d_output_text_field_binary_matrix_confirmed
next     = E136E_OUTPUT_TEXT_FIELD_ROUTE_RENDER_INTEGRATION_CONFIRM
```

## Metrics

```text
case_count = 10
pass_count = 10
fail_count = 0
commit_case_count = 7
reject_case_count = 3
matrix_shape_pass_count = 10
roundtrip_pass_count = 7
zero_fill_pass_count = 10
overflow_reject_count = 1
direct_write_reject_count = 1
nul_reject_count = 1
tamper_detect_count = 1
```

## Interpretation

OutputTextField is the canonical name for the output-side text matrix.
The field is byte-shaped, not tokenizer-shaped: N rows by 8 bit cells.
For ASCII, N bytes equals N characters. UTF-8 text may use more rows than
visible characters.

## Boundary

This confirms binary output-field representation and Agency-gated commit
semantics. It does not claim text generation, next-token prediction,
assistant quality, or open-domain chat.

## Cases

### ascii_greeting

```text
shape = 16 x 8
action = commit
reason = agency_output_text_field_commit
input_byte_len = 5
committed_byte_len = 5
roundtrip_pass = True
zero_fill_after_commit = True
checksum_valid_after_optional_tamper = True
pass_gate = True
```

### age_answer_ascii

```text
shape = 64 x 8
action = commit
reason = agency_output_text_field_commit
input_byte_len = 25
committed_byte_len = 25
roundtrip_pass = True
zero_fill_after_commit = True
checksum_valid_after_optional_tamper = True
pass_gate = True
```

### age_answer_utf8

```text
shape = 64 x 8
action = commit
reason = agency_output_text_field_commit
input_byte_len = 26
committed_byte_len = 26
roundtrip_pass = True
zero_fill_after_commit = True
checksum_valid_after_optional_tamper = True
pass_gate = True
```

### json_status

```text
shape = 96 x 8
action = commit
reason = agency_output_text_field_commit
input_byte_len = 40
committed_byte_len = 40
roundtrip_pass = True
zero_fill_after_commit = True
checksum_valid_after_optional_tamper = True
pass_gate = True
```

### multiline_code

```text
shape = 96 x 8
action = commit
reason = agency_output_text_field_commit
input_byte_len = 48
committed_byte_len = 48
roundtrip_pass = True
zero_fill_after_commit = True
checksum_valid_after_optional_tamper = True
pass_gate = True
```

### empty_output

```text
shape = 8 x 8
action = commit
reason = agency_output_text_field_commit
input_byte_len = 0
committed_byte_len = 0
roundtrip_pass = True
zero_fill_after_commit = True
checksum_valid_after_optional_tamper = True
pass_gate = True
```

### overflow_reject

```text
shape = 8 x 8
action = reject
reason = capacity_overflow
input_byte_len = 10
committed_byte_len = 0
roundtrip_pass = False
zero_fill_after_commit = True
checksum_valid_after_optional_tamper = True
pass_gate = True
```

### direct_write_reject

```text
shape = 16 x 8
action = reject
reason = direct_output_text_field_write_rejected
input_byte_len = 6
committed_byte_len = 0
roundtrip_pass = False
zero_fill_after_commit = True
checksum_valid_after_optional_tamper = True
pass_gate = True
```

### nul_reject

```text
shape = 16 x 8
action = reject
reason = nul_byte_rejected
input_byte_len = 8
committed_byte_len = 0
roundtrip_pass = False
zero_fill_after_commit = True
checksum_valid_after_optional_tamper = True
pass_gate = True
```

### tamper_detect

```text
shape = 16 x 8
action = commit
reason = agency_output_text_field_commit
input_byte_len = 2
committed_byte_len = 2
roundtrip_pass = True
zero_fill_after_commit = True
checksum_valid_after_optional_tamper = False
pass_gate = True
```
