# Gemini Research: Can Fixed Sinusoidal Encodings Replace Learned Embeddings?

**Source:** Gemini Deep Research (2026-02-26)

## The Parameter Bloat

In a recurrent ring-buffer pointer network with Vocab=256, hidden_dim=8192:
- Input Embedding: 2.1M params (256 × 8192)
- Output Projection: 2.1M params (8192 × 256)
- Total I/O Overhead: **79% of the model** is just I/O matrices

## Proposed Experimental Architectures

| Arm | Input | Output | I/O Params |
|-----|-------|--------|------------|
| 1. Baseline | Learned(256, 8192) | Learned(8192, 256) | 4.2M |
| 2. Sinusoidal Input | Fixed Sin/Cos | Learned(8192, 256) | 2.1M |
| 3. Weight Tying | Matrix A | Matrix A.T | 2.1M |
| 4. Combined | Fixed Sin/Cos | Fixed Sin/Cos.T | **0** |

## Key Arguments

- **"The first layer can learn any rotation."** Even if the input is fixed, the first weight matrix (W_1) can project the sinusoidal spiral into the necessary latent space for the RNN.
- Fixed encodings may have higher initial loss but are **immune to embedding overfitting**, unlike learned models which may memorize.
- Combined approach (Fixed In + Shared Out) could reduce model size by **~80%** with minimal impact on expressiveness.

## Sinusoidal Spiral Insight

Unlike learned embeddings which start random and cluster during training, sinusoidal encodings place every byte (0-255) onto a deterministic, high-dimensional manifold (represented as a spiral). The spiral structure is ordinal — nearby bytes map to nearby positions.

## Referenced Models

- **ByteNet (2017):** Dilated convolutions on raw bytes, learned embeddings. Established feasibility.
- **ByT5 (2021):** Google Transformer, learned embeddings, mitigates cost with massive encoder depth.
- **MegaByte (2023):** Byte patches to reduce sequence length, learned patch embeddings.

## Verdict

Gemini recommends testing all 4 arms experimentally. Optimistic about the Combined (Fixed Sin/Cos In + Sin/Cos.T Out) approach for ~80% parameter reduction.

**Note:** Does NOT address the cosine similarity problem between adjacent bytes (~0.97 in sinusoidal space) that Claude Research identified. Also does not mention Hadamard or random orthogonal alternatives.
