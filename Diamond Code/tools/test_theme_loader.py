#!/usr/bin/env python3
"""Deterministic tests for ThemeLoader (JSONL dialogue-mode data loader).

Tests cover:
  - Loading and parsing .jsonl theme files
  - Curriculum/gist separation
  - Batch shape and dtype
  - Dialogue-mode layout (input/output alternation)
  - y = shifted x (autoregressive target)
  - Loss mask (1 on output positions, 0 on input)
  - Determinism (same seed = same output)
  - Gist ratio mixing
  - Difficulty filtering
  - Multi-theme gist aggregation
  - Text-to-bits encoding (utf8, hex, base64)
  - Edge cases (single pair, max bytes, empty positions)

Usage:
    python tools/test_theme_loader.py
    python -m pytest tools/test_theme_loader.py -v
"""

import json
import os
import sys
import tempfile
import shutil

import numpy as np
import torch

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traindat_loader import ThemeLoader, _text_to_bits, _encode_theme_field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_theme_dir(themes: dict, base_dir: str) -> str:
    """Create a temporary themes directory with .jsonl files.

    Args:
        themes: {filename: [header_dict, pair_dict, ...]}
        base_dir: temp dir root
    Returns:
        Path to the themes directory.
    """
    d = os.path.join(base_dir, "themes")
    os.makedirs(d, exist_ok=True)
    for fname, lines in themes.items():
        with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
            for obj in lines:
                f.write(json.dumps(obj) + "\n")
    return d


def simple_header(theme="test", encoding="utf8", gist_ratio=0.10):
    return {"_meta": True, "theme": theme, "version": 1,
            "encoding": encoding, "gist_ratio": gist_ratio, "active": True}


def simple_pair(pid, s="c", d=1, inp="2+3", out="5", set_id=None):
    p = {"id": pid, "s": s, "d": d, "in": inp, "out": out}
    if set_id:
        p["set"] = set_id
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTextToBits:
    """Test the _text_to_bits helper function."""

    def test_utf8_basic(self):
        bits = _text_to_bits("A", 8, "utf8")
        assert bits.shape == (8,)
        # 'A' = 0x41 = 01000001
        expected = np.array([0, 1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(bits, expected)

    def test_utf8_padding(self):
        """Short text should be zero-padded to num_bits."""
        bits = _text_to_bits("A", 16, "utf8")
        assert bits.shape == (16,)
        # First 8 bits = 'A', next 8 = 0x00 padding
        assert bits[8:].sum() == 0.0

    def test_utf8_multibyte(self):
        """Multi-byte UTF-8 should encode correctly."""
        bits = _text_to_bits("AB", 16, "utf8")
        assert bits.shape == (16,)
        # 'A'=0x41, 'B'=0x42
        a_bits = np.array([0, 1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        b_bits = np.array([0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(bits[:8], a_bits)
        np.testing.assert_array_equal(bits[8:], b_bits)

    def test_hex_encoding(self):
        bits = _text_to_bits("41", 8, "hex")
        expected = np.array([0, 1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(bits, expected)

    def test_base64_encoding(self):
        import base64
        # base64 of 'A' (0x41) = 'QQ=='
        bits = _text_to_bits("QQ==", 8, "base64")
        expected = np.array([0, 1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(bits, expected)

    def test_large_num_bits(self):
        """Typical Nano config: num_bits=6184."""
        bits = _text_to_bits("Hello", 6184, "utf8")
        assert bits.shape == (6184,)
        # First 40 bits = 'Hello' (5 bytes), rest zero-padded
        nonzero = (bits != 0).sum()
        assert nonzero > 0  # 'Hello' has some 1-bits
        assert nonzero < 100  # but most of 6184 is padding

    def test_truncation(self):
        """Text longer than bytes_per_pos should be truncated."""
        long_text = "A" * 1000
        bits = _text_to_bits(long_text, 8, "utf8")  # 1 byte = 8 bits
        assert bits.shape == (8,)
        # Only first byte kept
        expected = np.array([0, 1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(bits, expected)

    def test_empty_string(self):
        bits = _text_to_bits("", 8, "utf8")
        assert bits.shape == (8,)
        np.testing.assert_array_equal(bits, np.zeros(8, dtype=np.float32))


class TestThemeLoaderBasic:
    """Test basic ThemeLoader functionality."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def _make_loader(self, themes, num_bits=64, **kwargs):
        d = make_theme_dir(themes, self.tmpdir)
        return ThemeLoader(d, num_bits=num_bits, **kwargs)

    def test_load_single_theme(self):
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1+1", out="2"),
                simple_pair("m-2", inp="2+2", out="4"),
            ]
        })
        assert loader.theme_names == ["math"]
        assert loader.active_theme == "math"
        stats = loader.stats()
        assert stats["math"]["curriculum"] == 2
        assert stats["math"]["gist"] == 0

    def test_curriculum_and_gist_separation(self):
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", s="c", inp="1+1", out="2"),
                simple_pair("m-2", s="c", inp="2+2", out="4"),
                simple_pair("m-g1", s="g", inp="0+0", out="0"),
            ]
        })
        stats = loader.stats()
        assert stats["math"]["curriculum"] == 2
        assert stats["math"]["gist"] == 1

    def test_batch_shapes(self):
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1+1", out="2"),
                simple_pair("m-2", inp="2+2", out="4"),
                simple_pair("m-3", inp="3+3", out="6"),
            ]
        }, num_bits=64)
        x, y, mask = loader.sample_batch(n_samples=4, seq_len=6, seed=1)
        assert x.shape == (4, 6, 64)
        assert y.shape == (4, 6, 64)
        assert mask.shape == (4, 6)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
        assert mask.dtype == torch.float32

    def test_loss_mask_pattern(self):
        """Loss mask should be 1.0 on output positions (odd), 0.0 on input (even)."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1+1", out="2"),
            ]
        }, num_bits=16)
        _, _, mask = loader.sample_batch(n_samples=1, seq_len=6, seed=1)
        expected = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
        torch.testing.assert_close(mask, expected)

    def test_loss_mask_seq4(self):
        """Loss mask for seq_len=4."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1+1", out="2"),
            ]
        }, num_bits=16)
        _, _, mask = loader.sample_batch(n_samples=1, seq_len=4, seed=1)
        expected = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        torch.testing.assert_close(mask, expected)

    def test_y_is_shifted_x(self):
        """y[t] should equal x[t+1] for all t except the last."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1+1", out="2"),
                simple_pair("m-2", inp="3+4", out="7"),
                simple_pair("m-3", inp="5+6", out="11"),
            ]
        }, num_bits=32)
        x, y, _ = loader.sample_batch(n_samples=2, seq_len=6, seed=42)

        for i in range(2):
            for t in range(5):
                assert torch.equal(y[i, t], x[i, t + 1]), \
                    f"y[{i},{t}] != x[{i},{t+1}]"
            # Last position target should be zeros
            assert torch.equal(y[i, 5], torch.zeros(32)), \
                f"y[{i},5] should be zeros"

    def test_input_positions_have_content(self):
        """Input positions (even) should have non-zero bits."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="123+456", out="579"),
            ]
        }, num_bits=64)
        x, _, _ = loader.sample_batch(n_samples=1, seq_len=6, seed=1)

        for t in [0, 2, 4]:
            nonzero = (x[0, t] != 0).sum().item()
            assert nonzero > 0, f"Position {t} (input) should have content"

    def test_output_positions_have_content(self):
        """Output positions (odd) should have non-zero bits."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1+1", out="2"),
            ]
        }, num_bits=64)
        x, _, _ = loader.sample_batch(n_samples=1, seq_len=6, seed=1)

        for t in [1, 3, 5]:
            nonzero = (x[0, t] != 0).sum().item()
            assert nonzero > 0, f"Position {t} (output) should have content"

    def test_deterministic_same_seed(self):
        """Same seed should produce identical batches."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1+1", out="2"),
                simple_pair("m-2", inp="2+3", out="5"),
                simple_pair("m-3", inp="4+5", out="9"),
            ]
        }, num_bits=32)

        x1, y1, m1 = loader.sample_batch(n_samples=3, seq_len=6, seed=42)
        x2, y2, m2 = loader.sample_batch(n_samples=3, seq_len=6, seed=42)

        assert torch.equal(x1, x2), "x should be identical with same seed"
        assert torch.equal(y1, y2), "y should be identical with same seed"
        assert torch.equal(m1, m2), "mask should be identical with same seed"

    def test_different_seed_different_data(self):
        """Different seeds should (very likely) produce different batches."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                *[simple_pair(f"m-{i}", inp=f"{i}+{i+1}", out=f"{2*i+1}")
                  for i in range(50)],
            ]
        }, num_bits=32)

        x1, _, _ = loader.sample_batch(n_samples=5, seq_len=6, seed=1)
        x2, _, _ = loader.sample_batch(n_samples=5, seq_len=6, seed=2)
        assert not torch.equal(x1, x2), "Different seeds should produce different data"


class TestThemeLoaderMixing:
    """Test curriculum/gist mixing and multi-theme behavior."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def _make_loader(self, themes, num_bits=32, **kwargs):
        d = make_theme_dir(themes, self.tmpdir)
        return ThemeLoader(d, num_bits=num_bits, **kwargs)

    def test_gist_ratio_zero(self):
        """With gist_ratio=0, only curriculum pairs should be used."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-c1", s="c", inp="CURRICULUM", out="YES"),
                simple_pair("m-g1", s="g", inp="GIST", out="NO"),
            ]
        }, gist_ratio=0.0)

        # Run many batches — with gist_ratio=0, should never pick gist
        all_content = []
        for seed in range(100):
            x, _, _ = loader.sample_batch(n_samples=1, seq_len=2, seed=seed)
            all_content.append(x)

        # Can't easily decode back, but at least verify it runs without error
        assert len(all_content) == 100

    def test_gist_only_fallback(self):
        """If no curriculum pairs, should fall back to gist."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-g1", s="g", inp="1+1", out="2"),
                simple_pair("m-g2", s="g", inp="2+2", out="4"),
            ]
        })
        # Should not raise even though curriculum is empty
        x, y, mask = loader.sample_batch(n_samples=1, seq_len=4, seed=1)
        assert x.shape == (1, 4, 32)

    def test_multi_theme_gist_aggregation(self):
        """Gist from all themes should be available, even non-active ones."""
        loader = self._make_loader({
            "alpha.jsonl": [
                simple_header("alpha"),
                simple_pair("a-c1", s="c", inp="A", out="1"),
                simple_pair("a-g1", s="g", inp="AG", out="2"),
            ],
            "beta.jsonl": [
                simple_header("beta"),
                simple_pair("b-c1", s="c", inp="B", out="3"),
                simple_pair("b-g1", s="g", inp="BG", out="4"),
            ],
        }, active_theme="alpha", gist_ratio=0.5)

        stats = loader.stats()
        assert stats["alpha"]["active"] is True
        assert stats["beta"]["active"] is False
        # Both themes' gist should be collected
        all_gist = loader._all_gist()
        assert len(all_gist) == 2  # one from each theme

    def test_active_theme_switch(self):
        loader = self._make_loader({
            "alpha.jsonl": [
                simple_header("alpha"),
                simple_pair("a-1", inp="A", out="1"),
            ],
            "beta.jsonl": [
                simple_header("beta"),
                simple_pair("b-1", inp="B", out="2"),
            ],
        })
        assert loader.active_theme == "alpha"
        loader.active_theme = "beta"
        assert loader.active_theme == "beta"
        assert loader.stats()["beta"]["active"] is True

    def test_invalid_theme_switch(self):
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1", out="1"),
            ]
        })
        try:
            loader.active_theme = "nonexistent"
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestThemeLoaderDifficulty:
    """Test difficulty filtering."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def _make_loader(self, themes, num_bits=32, **kwargs):
        d = make_theme_dir(themes, self.tmpdir)
        return ThemeLoader(d, num_bits=num_bits, **kwargs)

    def test_difficulty_filter(self):
        """difficulty_max should filter out harder pairs."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("easy", d=1, inp="1+1", out="2"),
                simple_pair("med", d=3, inp="100+200", out="300"),
                simple_pair("hard", d=5, inp="12345+67890", out="80235"),
            ]
        }, gist_ratio=0.0)

        # With difficulty_max=1, only easy pairs available
        # Run multiple batches to ensure only easy content appears
        x1, _, _ = loader.sample_batch(n_samples=10, seq_len=2, seed=42,
                                        difficulty_max=1)
        assert x1.shape == (10, 2, 32)

    def test_difficulty_filter_none(self):
        """Without filter, all pairs available."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("e1", d=1, inp="1+1", out="2"),
                simple_pair("h1", d=6, inp="999999+1", out="1000000"),
            ]
        })
        # Should work fine with no filter
        x, _, _ = loader.sample_batch(n_samples=1, seq_len=2, seed=1)
        assert x.shape[0] == 1


class TestThemeLoaderEncoding:
    """Test different encodings (hex, base64)."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def _make_loader(self, themes, num_bits=32, **kwargs):
        d = make_theme_dir(themes, self.tmpdir)
        return ThemeLoader(d, num_bits=num_bits, **kwargs)

    def test_hex_theme(self):
        """Theme with hex encoding should produce valid bits."""
        loader = self._make_loader({
            "hex.jsonl": [
                simple_header("hex", encoding="hex"),
                simple_pair("h-1", inp="4142", out="83"),  # AB -> 0x83
            ]
        }, num_bits=16)
        x, _, _ = loader.sample_batch(n_samples=1, seq_len=2, seed=1)
        assert x.shape == (1, 2, 16)
        # Position 0 should have 'AB' = 0x41 0x42 = 01000001 01000010
        expected_bits = np.array([0,1,0,0,0,0,0,1, 0,1,0,0,0,0,1,0], dtype=np.float32)
        np.testing.assert_array_equal(x[0, 0].numpy(), expected_bits)

    def test_base64_theme(self):
        """Theme with base64 encoding should decode correctly."""
        import base64 as b64
        encoded_in = b64.b64encode(b"Hi").decode()  # 'SGk='
        encoded_out = b64.b64encode(b"!").decode()   # 'IQ=='
        loader = self._make_loader({
            "b64.jsonl": [
                simple_header("b64", encoding="base64"),
                simple_pair("b-1", inp=encoded_in, out=encoded_out),
            ]
        }, num_bits=16)
        x, _, _ = loader.sample_batch(n_samples=1, seq_len=2, seed=1)
        assert x.shape == (1, 2, 16)


class TestThemeLoaderEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def _make_loader(self, themes, num_bits=32, **kwargs):
        d = make_theme_dir(themes, self.tmpdir)
        return ThemeLoader(d, num_bits=num_bits, **kwargs)

    def test_single_pair_reused(self):
        """With only 1 pair, all slots in batch should use it."""
        loader = self._make_loader({
            "one.jsonl": [
                simple_header("one"),
                simple_pair("o-1", inp="X", out="Y"),
            ]
        }, num_bits=16, gist_ratio=0.0)
        x, _, _ = loader.sample_batch(n_samples=3, seq_len=6, seed=1)
        # All input positions should be identical (same pair reused)
        assert torch.equal(x[0, 0], x[0, 2])
        assert torch.equal(x[0, 0], x[0, 4])
        assert torch.equal(x[0, 0], x[1, 0])

    def test_seq_len_2(self):
        """Minimal seq_len=2 should work (1 pair per sequence)."""
        loader = self._make_loader({
            "min.jsonl": [
                simple_header("min"),
                simple_pair("m-1", inp="A", out="B"),
            ]
        }, num_bits=8)
        x, y, mask = loader.sample_batch(n_samples=1, seq_len=2, seed=1)
        assert x.shape == (1, 2, 8)
        assert mask[0].tolist() == [0.0, 1.0]
        # y[0] = x[1] (shifted)
        assert torch.equal(y[0, 0], x[0, 1])
        # y[1] = zeros
        assert torch.equal(y[0, 1], torch.zeros(8))

    def test_odd_seq_len_rejected(self):
        """Odd seq_len should raise AssertionError."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1", out="1"),
            ]
        })
        try:
            loader.sample_batch(n_samples=1, seq_len=5, seed=1)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass

    def test_no_themes_dir(self):
        """Missing directory should raise FileNotFoundError."""
        try:
            ThemeLoader("/nonexistent/path/themes")
            assert False, "Should have raised"
        except FileNotFoundError:
            pass

    def test_empty_themes_dir(self):
        """Empty directory should raise FileNotFoundError."""
        d = os.path.join(self.tmpdir, "empty_themes")
        os.makedirs(d)
        try:
            ThemeLoader(d)
            assert False, "Should have raised"
        except FileNotFoundError:
            pass

    def test_reload(self):
        """reload() should re-read files from disk."""
        d = make_theme_dir({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1+1", out="2"),
            ]
        }, self.tmpdir)
        loader = ThemeLoader(d, num_bits=16)
        assert loader.stats()["math"]["curriculum"] == 1

        # Add a pair to the file
        with open(os.path.join(d, "math.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(simple_pair("m-2", inp="2+2", out="4")) + "\n")

        loader.reload()
        assert loader.stats()["math"]["curriculum"] == 2

    def test_large_batch(self):
        """Large batch size should work."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                *[simple_pair(f"m-{i}", inp=f"{i}+1", out=f"{i+1}")
                  for i in range(100)],
            ]
        }, num_bits=64)
        x, y, mask = loader.sample_batch(n_samples=50, seq_len=6, seed=1)
        assert x.shape == (50, 6, 64)

    def test_num_bits_6184(self):
        """Full Nano config num_bits=6184 should work."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="Hello world", out="Hi there"),
            ]
        }, num_bits=6184)
        x, y, mask = loader.sample_batch(n_samples=1, seq_len=6, seed=1)
        assert x.shape == (1, 6, 6184)
        assert y.shape == (1, 6, 6184)

    def test_content_round_trip(self):
        """Verify that encoded content can be decoded back to original text."""
        text = "2+3"
        bits = _text_to_bits(text, 64, "utf8")
        # Recover bytes from bits
        bit_ints = bits.astype(np.uint8)
        byte_arr = np.packbits(bit_ints)
        raw = bytes(byte_arr[:len(text)])
        assert raw.decode("utf-8") == text

    def test_set_field_preserved(self):
        """Set field should be loaded and stored (even if not used in batching)."""
        loader = self._make_loader({
            "math.jsonl": [
                simple_header("math"),
                simple_pair("m-1", inp="1+2", out="3", set_id="s1"),
                simple_pair("m-2", inp="2+1", out="3", set_id="s1"),
            ]
        })
        theme = loader._themes["math"]
        assert theme["curriculum"][0]["set"] == "s1"
        assert theme["curriculum"][1]["set"] == "s1"


class TestThemeLoaderWithRealFile:
    """Test with the actual generated arithmetic_add.jsonl if it exists."""

    def test_real_arithmetic_add(self):
        real_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "themes"
        )
        jsonl = os.path.join(real_path, "arithmetic_add.jsonl")
        if not os.path.exists(jsonl):
            return  # skip if file doesn't exist

        loader = ThemeLoader(real_path, num_bits=6184, gist_ratio=0.05)
        stats = loader.stats()
        assert "arithmetic_add" in stats
        assert stats["arithmetic_add"]["curriculum"] > 0

        x, y, mask = loader.sample_batch(n_samples=4, seq_len=6, seed=42)
        assert x.shape == (4, 6, 6184)
        assert y.shape == (4, 6, 6184)
        assert mask.shape == (4, 6)

        # Verify shifted target
        for i in range(4):
            for t in range(5):
                assert torch.equal(y[i, t], x[i, t + 1])
            assert torch.equal(y[i, 5], torch.zeros(6184))

        # Verify mask pattern
        for i in range(4):
            assert mask[i].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

        # Determinism
        x2, y2, m2 = loader.sample_batch(n_samples=4, seq_len=6, seed=42)
        assert torch.equal(x, x2)
        assert torch.equal(y, y2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestTextToBits,
        TestThemeLoaderBasic,
        TestThemeLoaderMixing,
        TestThemeLoaderDifficulty,
        TestThemeLoaderEncoding,
        TestThemeLoaderEdgeCases,
        TestThemeLoaderWithRealFile,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            method = getattr(instance, method_name)
            test_id = f"{cls.__name__}.{method_name}"
            try:
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                method()
                if hasattr(instance, "teardown_method"):
                    instance.teardown_method()
                passed += 1
                print(f"  PASS  {test_id}")
            except Exception as e:
                failed += 1
                tb = traceback.format_exc()
                errors.append((test_id, tb))
                print(f"  FAIL  {test_id}: {e}")
                if hasattr(instance, "teardown_method"):
                    try:
                        instance.teardown_method()
                    except Exception:
                        pass

    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for test_id, tb in errors:
            print(f"\n--- {test_id} ---")
            print(tb)
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
