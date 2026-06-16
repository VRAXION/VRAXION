# E136D OutputTextField Binary Matrix Smoke Result

```text
decision = e136d_output_text_field_binary_matrix_confirmed
next     = E136E_OUTPUT_TEXT_FIELD_ROUTE_RENDER_INTEGRATION_CONFIRM
```

E136D confirms the first explicit output-side text field: an
Agency-committed `OutputTextField` represented as an `N x 8` binary matrix.

## Result

```text
field_name = OutputTextField
case_count = 10
pass_count = 10
fail_count = 0
commit_case_count = 7
reject_case_count = 3
matrix_shape_pass_count = 10
roundtrip_pass_count = 7
utf8_valid_count = 7
zero_fill_pass_count = 10
overflow_reject_count = 1
direct_write_reject_count = 1
nul_reject_count = 1
tamper_detect_count = 1
```

## Interpretation

The tested field is byte-shaped, not tokenizer-shaped:

```text
OutputTextField[N][8]
row = UTF-8 byte
cell = bit
```

This means an output capacity of `N` stores `N` bytes. ASCII text maps one
visible character per row; accented UTF-8 text can use more rows than visible
characters.

## Boundary

This is representation and commit evidence only. It does not prove new
pockets, text training, next-token prediction, or open-domain assistant
generation.
