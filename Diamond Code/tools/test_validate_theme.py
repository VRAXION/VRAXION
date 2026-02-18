#!/usr/bin/env python3
"""Deterministic tests for validate_theme.py.

Tests every validation rule with known-good and known-bad inputs.
No randomness, no external dependencies, no GPU.

Usage:
  python tools/test_validate_theme.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent dir so we can import
sys.path.insert(0, str(Path(__file__).parent))
from validate_theme import validate_theme_file, encode_field

PASS = 0
FAIL = 0


def assert_true(condition, test_name):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [ok] {test_name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {test_name}")


def write_jsonl(lines: list) -> Path:
    """Write lines to a temp .jsonl file, return path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="test_theme_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return Path(path)


def make_header(**overrides):
    """Create a valid header dict with optional overrides."""
    h = {
        "_meta": True,
        "theme": "test_theme",
        "version": 1,
        "encoding": "utf8",
        "gist_ratio": 0.10,
    }
    h.update(overrides)
    return h


def make_pair(id="t-001", s="c", d=1, inp="hello", out="world", **extra):
    """Create a valid pair dict."""
    p = {"id": id, "s": s, "d": d, "in": inp, "out": out}
    p.update(extra)
    return p


# ── Test Suite ──────────────────────────────────────────────────────────

def test_valid_minimal():
    """A minimal valid file should pass."""
    print("\n--- test_valid_minimal ---")
    path = write_jsonl([
        make_header(),
        make_pair("t-001", "c", 1, "input1", "output1"),
        make_pair("t-002", "g", 2, "input2", "output2"),
    ])
    try:
        r = validate_theme_file(path)
        assert_true(r["ok"], "valid file passes")
        assert_true(r["n_pairs"] == 2, "pair count = 2")
        assert_true(r["counts"]["c"] == 1, "curriculum count = 1")
        assert_true(r["counts"]["g"] == 1, "gist count = 1")
        assert_true(len(r["errors"]) == 0, "no errors")
    finally:
        os.unlink(path)


def test_empty_file():
    """Empty file should fail."""
    print("\n--- test_empty_file ---")
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        r = validate_theme_file(Path(path))
        assert_true(not r["ok"], "empty file fails")
        assert_true(any("empty" in e for e in r["errors"]), "error mentions empty")
    finally:
        os.unlink(path)


def test_bad_json_header():
    """Non-JSON first line should fail."""
    print("\n--- test_bad_json_header ---")
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        f.write("this is not json\n")
        f.write(json.dumps(make_pair()) + "\n")
    try:
        r = validate_theme_file(Path(path))
        assert_true(not r["ok"], "bad header fails")
        assert_true(any("invalid JSON" in e for e in r["errors"]), "error mentions JSON")
    finally:
        os.unlink(path)


def test_missing_meta_flag():
    """Header without _meta:true should fail."""
    print("\n--- test_missing_meta_flag ---")
    h = make_header()
    del h["_meta"]
    path = write_jsonl([h, make_pair()])
    try:
        r = validate_theme_file(path)
        assert_true(not r["ok"], "missing _meta fails")
    finally:
        os.unlink(path)


def test_missing_header_fields():
    """Header missing required fields should fail."""
    print("\n--- test_missing_header_fields ---")
    path = write_jsonl([
        {"_meta": True, "theme": "test"},  # missing version, encoding, gist_ratio
        make_pair(),
    ])
    try:
        r = validate_theme_file(path)
        assert_true(not r["ok"], "incomplete header fails")
        assert_true(any("encoding" in e for e in r["errors"]), "mentions missing encoding")
        assert_true(any("gist_ratio" in e for e in r["errors"]), "mentions missing gist_ratio")
    finally:
        os.unlink(path)


def test_duplicate_ids():
    """Duplicate pair IDs should fail."""
    print("\n--- test_duplicate_ids ---")
    path = write_jsonl([
        make_header(),
        make_pair("dup-001", "c", 1, "a", "b"),
        make_pair("dup-001", "c", 2, "c", "d"),  # same ID
    ])
    try:
        r = validate_theme_file(path)
        assert_true(not r["ok"], "duplicate IDs fail")
        assert_true(any("duplicate" in e for e in r["errors"]), "error mentions duplicate")
    finally:
        os.unlink(path)


def test_invalid_state():
    """State other than 'c' or 'g' should fail."""
    print("\n--- test_invalid_state ---")
    path = write_jsonl([
        make_header(),
        make_pair("t-001", "x", 1, "a", "b"),
    ])
    try:
        r = validate_theme_file(path)
        assert_true(not r["ok"], "invalid state fails")
        assert_true(any("invalid state" in e for e in r["errors"]), "error mentions state")
    finally:
        os.unlink(path)


def test_invalid_difficulty():
    """Difficulty outside 1-6 should fail."""
    print("\n--- test_invalid_difficulty ---")
    for bad_d in [0, 7, -1, 99, "high"]:
        path = write_jsonl([
            make_header(),
            make_pair("t-001", "c", bad_d, "a", "b"),
        ])
        try:
            r = validate_theme_file(path)
            assert_true(not r["ok"], f"difficulty={bad_d} fails")
        finally:
            os.unlink(path)


def test_missing_pair_fields():
    """Pair missing required fields should fail."""
    print("\n--- test_missing_pair_fields ---")
    path = write_jsonl([
        make_header(),
        {"id": "t-001", "s": "c"},  # missing d, in, out
    ])
    try:
        r = validate_theme_file(path)
        assert_true(not r["ok"], "missing fields fails")
        assert_true(any("missing fields" in e for e in r["errors"]), "error mentions missing")
    finally:
        os.unlink(path)


def test_content_too_large():
    """Content exceeding max_bytes should fail."""
    print("\n--- test_content_too_large ---")
    big_input = "A" * 800  # 800 bytes > 773
    path = write_jsonl([
        make_header(),
        make_pair("t-001", "c", 1, big_input, "small"),
    ])
    try:
        r = validate_theme_file(path, max_bytes=773)
        assert_true(not r["ok"], "oversized input fails")
        assert_true(any("800 bytes" in e for e in r["errors"]), "error shows byte count")
    finally:
        os.unlink(path)


def test_content_exactly_max():
    """Content exactly at max_bytes should pass."""
    print("\n--- test_content_exactly_max ---")
    exact_input = "B" * 773
    path = write_jsonl([
        make_header(),
        make_pair("t-001", "c", 1, exact_input, "ok"),
    ])
    try:
        r = validate_theme_file(path, max_bytes=773)
        assert_true(r["ok"], "exactly 773 bytes passes")
    finally:
        os.unlink(path)


def test_hex_encoding_valid():
    """Valid hex-encoded content should pass."""
    print("\n--- test_hex_encoding_valid ---")
    path = write_jsonl([
        make_header(encoding="hex"),
        make_pair("t-001", "c", 1, "0a3f5b", "ff00ee"),
    ])
    try:
        r = validate_theme_file(path)
        assert_true(r["ok"], "valid hex passes")
    finally:
        os.unlink(path)


def test_hex_encoding_invalid():
    """Invalid hex string should fail when encoding=hex."""
    print("\n--- test_hex_encoding_invalid ---")
    path = write_jsonl([
        make_header(encoding="hex"),
        make_pair("t-001", "c", 1, "not_valid_hex!", "ff00"),
    ])
    try:
        r = validate_theme_file(path)
        assert_true(not r["ok"], "invalid hex fails")
        assert_true(any("cannot encode" in e for e in r["errors"]), "error mentions encoding")
    finally:
        os.unlink(path)


def test_base64_encoding_valid():
    """Valid base64 content should pass."""
    print("\n--- test_base64_encoding_valid ---")
    import base64
    content = base64.b64encode(b"hello world").decode()
    path = write_jsonl([
        make_header(encoding="base64"),
        make_pair("t-001", "c", 1, content, content),
    ])
    try:
        r = validate_theme_file(path)
        assert_true(r["ok"], "valid base64 passes")
    finally:
        os.unlink(path)


def test_set_field():
    """Set field should be tracked."""
    print("\n--- test_set_field ---")
    path = write_jsonl([
        make_header(),
        make_pair("t-001a", "c", 1, "apple is green", "green apple", set="fruit-001"),
        make_pair("t-001b", "c", 1, "apple is green", "apple, green color", set="fruit-001"),
        make_pair("t-002a", "c", 2, "sky is blue", "blue sky", set="sky-001"),
    ])
    try:
        r = validate_theme_file(path)
        assert_true(r["ok"], "set field accepted")
        assert_true(r["n_sets"] == 2, "2 unique sets found")
    finally:
        os.unlink(path)


def test_difficulty_counts():
    """Difficulty distribution should be tracked."""
    print("\n--- test_difficulty_counts ---")
    path = write_jsonl([
        make_header(),
        make_pair("t-001", "c", 1, "a", "b"),
        make_pair("t-002", "c", 1, "c", "d"),
        make_pair("t-003", "c", 3, "e", "f"),
        make_pair("t-004", "c", 6, "g", "h"),
    ])
    try:
        r = validate_theme_file(path)
        assert_true(r["ok"], "valid file passes")
        dc = r["difficulty_counts"]
        assert_true(dc[1] == 2, "d1 count = 2")
        assert_true(dc[3] == 1, "d3 count = 1")
        assert_true(dc[6] == 1, "d6 count = 1")
        assert_true(dc[2] == 0, "d2 count = 0")
    finally:
        os.unlink(path)


def test_no_gist_warning():
    """File with no gist pairs should warn (not fail)."""
    print("\n--- test_no_gist_warning ---")
    path = write_jsonl([
        make_header(),
        make_pair("t-001", "c", 1, "a", "b"),
    ])
    try:
        r = validate_theme_file(path)
        assert_true(r["ok"], "no gist still passes")
        assert_true(any("no gist" in w for w in r["warnings"]), "warns about missing gist")
    finally:
        os.unlink(path)


def test_gist_ratio_mismatch_warning():
    """Gist ratio far from actual should warn."""
    print("\n--- test_gist_ratio_mismatch_warning ---")
    path = write_jsonl([
        make_header(gist_ratio=0.50),  # says 50% gist
        make_pair("t-001", "c", 1, "a", "b"),
        make_pair("t-002", "c", 1, "c", "d"),
        make_pair("t-003", "g", 1, "e", "f"),  # actual = 1/3 = 33%
    ])
    try:
        r = validate_theme_file(path)
        assert_true(r["ok"], "ratio mismatch is warning not error")
        assert_true(any("gist_ratio" in w for w in r["warnings"]),
                    "warns about ratio mismatch")
    finally:
        os.unlink(path)


def test_multibyte_utf8():
    """Multi-byte UTF-8 characters should count bytes correctly."""
    print("\n--- test_multibyte_utf8 ---")
    # Each emoji is 4 bytes in UTF-8
    # 194 emojis * 4 bytes = 776 bytes > 773
    big_emoji = "\U0001f600" * 194  # 776 bytes
    path = write_jsonl([
        make_header(),
        make_pair("t-001", "c", 1, big_emoji, "ok"),
    ])
    try:
        r = validate_theme_file(path, max_bytes=773)
        assert_true(not r["ok"], "776 bytes of emoji exceeds 773 limit")
    finally:
        os.unlink(path)

    # 193 emojis * 4 = 772 bytes <= 773
    ok_emoji = "\U0001f600" * 193
    path = write_jsonl([
        make_header(),
        make_pair("t-001", "c", 1, ok_emoji, "ok"),
    ])
    try:
        r = validate_theme_file(path, max_bytes=773)
        assert_true(r["ok"], "772 bytes of emoji fits in 773 limit")
    finally:
        os.unlink(path)


def test_no_pairs():
    """Header-only file should fail."""
    print("\n--- test_no_pairs ---")
    path = write_jsonl([make_header()])
    try:
        r = validate_theme_file(path)
        assert_true(not r["ok"], "header-only file fails")
        assert_true(any("no data pairs" in e for e in r["errors"]), "error mentions no pairs")
    finally:
        os.unlink(path)


def test_encode_field_utf8():
    """encode_field with utf8 should match .encode('utf-8')."""
    print("\n--- test_encode_field_utf8 ---")
    assert_true(encode_field("hello", "utf8") == b"hello", "ascii encode")
    assert_true(encode_field("caf\u00e9", "utf8") == "café".encode("utf-8"), "accented encode")
    assert_true(len(encode_field("A" * 773, "utf8")) == 773, "773 bytes exactly")


def test_encode_field_hex():
    """encode_field with hex should decode hex strings."""
    print("\n--- test_encode_field_hex ---")
    assert_true(encode_field("ff00ab", "hex") == b"\xff\x00\xab", "hex decode")
    assert_true(len(encode_field("00" * 773, "hex")) == 773, "773 hex bytes")


def test_invalid_encoding_in_header():
    """Unknown encoding should fail."""
    print("\n--- test_invalid_encoding_in_header ---")
    path = write_jsonl([
        make_header(encoding="rot13"),
        make_pair("t-001", "c", 1, "a", "b"),
    ])
    try:
        r = validate_theme_file(path)
        assert_true(not r["ok"], "invalid encoding fails")
        assert_true(any("invalid encoding" in e for e in r["errors"]),
                    "error mentions encoding")
    finally:
        os.unlink(path)


def test_large_valid_file():
    """A file with 1000 valid pairs should pass quickly."""
    print("\n--- test_large_valid_file ---")
    lines = [make_header(gist_ratio=0.10)]
    for i in range(900):
        lines.append(make_pair(f"c-{i:05d}", "c", (i % 6) + 1,
                               f"input {i}", f"output {i}"))
    for i in range(100):
        lines.append(make_pair(f"g-{i:05d}", "g", 1,
                               f"gist input {i}", f"gist output {i}"))
    path = write_jsonl(lines)
    try:
        r = validate_theme_file(path)
        assert_true(r["ok"], "1000-pair file passes")
        assert_true(r["n_pairs"] == 1000, "count = 1000")
        assert_true(r["counts"]["c"] == 900, "900 curriculum")
        assert_true(r["counts"]["g"] == 100, "100 gist")
    finally:
        os.unlink(path)


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VALIDATE_THEME.PY — DETERMINISTIC TEST SUITE")
    print("=" * 60)

    test_valid_minimal()
    test_empty_file()
    test_bad_json_header()
    test_missing_meta_flag()
    test_missing_header_fields()
    test_duplicate_ids()
    test_invalid_state()
    test_invalid_difficulty()
    test_missing_pair_fields()
    test_content_too_large()
    test_content_exactly_max()
    test_hex_encoding_valid()
    test_hex_encoding_invalid()
    test_base64_encoding_valid()
    test_set_field()
    test_difficulty_counts()
    test_no_gist_warning()
    test_gist_ratio_mismatch_warning()
    test_multibyte_utf8()
    test_no_pairs()
    test_encode_field_utf8()
    test_encode_field_hex()
    test_invalid_encoding_in_header()
    test_large_valid_file()

    print()
    print("=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"  ALL CLEAR: {PASS}/{total} tests passed")
    else:
        print(f"  FAILED: {PASS}/{total} passed, {FAIL} failed")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
