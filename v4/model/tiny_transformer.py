"""Tiny GPT-style transformer baseline for fair comparison with INSTNCT.

Architecture: causal byte-level transformer sized to match INSTNCT's ~711K params.
  d_model=128, n_layers=3, n_heads=4, d_ff=576, sinusoidal positional encoding.
  Total: ~710K params.

Interface matches INSTNCT: forward(x, state=None, **kwargs) -> (logits, state)
"""

import math

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):

    def __init__(self, embed_mode=True, d_model=128, n_layers=3, n_heads=4, d_ff=576,
                 max_seq=512, dropout=0.0):
        super().__init__()
        assert embed_mode, "TinyTransformer only supports embed_mode=True (byte-level)"

        self.d_model = d_model
        self.embed = nn.Embedding(256, d_model)

        # Sinusoidal positional encoding (0 learnable params)
        pe = torch.zeros(max_seq, d_model)
        pos = torch.arange(max_seq).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq, d_model)

        # Transformer encoder with causal masking
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 256)

        # Causal mask cache (created on first forward, cached for reuse)
        self._causal_mask = None
        self._causal_mask_size = 0

    def _get_causal_mask(self, T, device):
        """Upper-triangular -inf mask for causal (autoregressive) attention."""
        if self._causal_mask_size >= T and self._causal_mask is not None:
            return self._causal_mask[:T, :T]
        mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        self._causal_mask = mask
        self._causal_mask_size = T
        return mask

    def forward(self, x, state=None, **kwargs):
        """
        Args:
            x: (B, T) long — byte indices 0-255
            state: ignored (no cross-sequence memory)
            **kwargs: absorbs S, probs, etc. from train.py

        Returns:
            logits: (B, T, 256) float
            state: None
        """
        B, T = x.shape

        # Embed + positional encoding
        h = self.embed(x) * math.sqrt(self.d_model)  # scale embedding (standard transformer practice)
        h = h + self.pe[:, :T]

        # Causal transformer
        mask = self._get_causal_mask(T, x.device)
        h = self.transformer(h, mask=mask, is_causal=True)

        # Output
        h = self.final_norm(h)
        logits = self.output_head(h)  # (B, T, 256)

        return logits, None
