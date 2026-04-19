"""L2-v1 — 2-layer tied bottleneck autoencoder with true round-trip metrics.

Architecture:
  648 -> W1 -> hidden -> W2 -> bottleneck -> W2.T -> hidden -> W1.T -> 648
  tied (W.T on the decoder side)

Loss: through FROZEN L1 decoder into byte space.
  reconstructed_648
    -> reshape (N, 8, 81)
    -> L1 decoder: h @ W_L1.T + b2_L1 -> (N, 8, 32)
    -> split into 16 byte-embeds
    -> nearest / classifier against the frozen L0 LUT
    -> recover bytes

Metrics:
  pair per-dim sign-match rate     (diagnostic)
  pair per-pair sign-match rate    (diagnostic)
  exact-window byte rate           (main round-trip metric)
  top-code sign stability          (encode -> decode -> encode consistency)

Measured on BOTH train and val (to separate under-fit from over-fit).

Sweep (GPT-suggested starting scope):
  bottleneck: 192, 256, 384   (reconstructive target — 64 is too aggressive)
  hidden: 256
  activation: softsign, c19
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_single_w_mirror import SingleWMirror, DEVICE
from diag_byte_l2_phase0_geometry_probe import (
    load_l1_champion, load_l0_lut, load_corpus_bytes, collect_l1_hidden_windows,
)

OUT_DIR = Path("output/merger_l2_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVATIONS = ["softsign", "c19"]
BOTTLENECKS = [192, 256, 384]   # GPT: 64/96/128 too aggressive for reconstruction
HIDDEN_DIM = 256                 # fixed intermediate
N_TRAIN = 8000
N_VAL = 2000
ADAM_EPOCHS = 500
LBFGS_OUTER = 30
BYTE_CHUNK = 4096

LOSS_W_PAIR_MSE = 0.25
LOSS_W_PAIR_SIGN = 0.25
LOSS_W_CODE_MSE = 0.10
LOSS_W_CODE_SIGN = 0.05


class L2Mirror(nn.Module):
    """2-layer tied mirror with bottleneck (GPT protocol).

    Architecture:
      enc1 = act_a(x @ W1 + b1)                 (in_dim -> hidden)
      enc2 = act_b(enc1 @ W2 + b2)              (hidden -> bottleneck)  <- "latent code"
      dec1 = act_a(enc2 @ W2.T + b3)            (bottleneck -> hidden)
      y    = dec1 @ W1.T + b4                   (hidden -> in_dim)
    """

    def __init__(self, in_dim, hidden, bottleneck, activation, device, w_scale=0.05):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.activation_name = activation

        self.W1 = nn.Parameter(torch.randn(in_dim, hidden, device=device) * w_scale)
        self.W2 = nn.Parameter(torch.randn(hidden, bottleneck, device=device) * w_scale)
        self.b1 = nn.Parameter(torch.zeros(hidden, device=device))
        self.b2 = nn.Parameter(torch.zeros(bottleneck, device=device))
        self.b3 = nn.Parameter(torch.zeros(hidden, device=device))
        self.b4 = nn.Parameter(torch.zeros(in_dim, device=device))

        if activation == "c19":
            self.c19_a1 = C19Activation(hidden, device)
            self.c19_b = C19Activation(bottleneck, device)
            self.c19_a2 = C19Activation(hidden, device)

    def _act(self, x, layer):
        if self.activation_name == "softsign":
            return x / (1.0 + x.abs())
        if self.activation_name == "hard_tanh":
            return x.clamp(-1.0, 1.0)
        if self.activation_name == "c19":
            if layer == "a1": return self.c19_a1(x)
            if layer == "b":  return self.c19_b(x)
            if layer == "a2": return self.c19_a2(x)
        raise ValueError(self.activation_name)

    def forward(self, x):
        enc1 = self._act(x @ self.W1 + self.b1, "a1")
        enc2 = self._act(enc1 @ self.W2 + self.b2, "b")
        dec1 = self._act(enc2 @ self.W2.t() + self.b3, "a2")
        y = dec1 @ self.W1.t() + self.b4
        return y

    def latent(self, x):
        """The bottleneck code — what the Brain will see later."""
        enc1 = self._act(x @ self.W1 + self.b1, "a1")
        return self._act(enc1 @ self.W2 + self.b2, "b")


class C19Activation(nn.Module):
    """Same C19 as the L1 merger, ported for reuse."""

    def __init__(self, dim, device, c_init=1.0, rho_init=8.0):
        super().__init__()
        self.c_raw = nn.Parameter(torch.full((dim,), c_init, device=device))
        self.rho_raw = nn.Parameter(torch.full((dim,), rho_init, device=device))

    def forward(self, x):
        c = self.c_raw.clamp(min=0.1)
        rho = self.rho_raw.clamp(min=0.0)
        L = 6.0 * c
        scaled = x / c
        n = scaled.floor()
        t = scaled - n
        h = t * (1.0 - t)
        sgn = torch.where(n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n))
        interior = c * (sgn * h + rho * h * h)
        return torch.where(
            x >= L, x - L, torch.where(x <= -L, x + L, interior)
        )


def byte_space_loss(y_recon_648, pair_embeds, l1_model):
    """Compatibility stub kept for local helpers; use roundtrip_loss in training."""
    N = y_recon_648.shape[0]
    hiddens = y_recon_648.reshape(N, 8, 81)
    decoded = hiddens @ l1_model.W.t() + l1_model.b2  # (N, 8, 32)
    mse = ((decoded - pair_embeds) ** 2).mean()
    sign_hinge = torch.relu(-decoded * torch.sign(pair_embeds)).mean()
    return mse + 0.5 * sign_hinge


def decode_pair_embeds(y_recon_648, l1_model):
    """648-dim reconstructed L1 hiddens -> frozen L1 decoder -> 8 x 32 pair embeds."""
    N = y_recon_648.shape[0]
    hiddens = y_recon_648.reshape(N, 8, 81)
    return hiddens @ l1_model.W.t() + l1_model.b2


def split_pair_embeds_to_bytes(pair_embeds):
    """(N, 8, 32) -> (N, 16, 16)."""
    left = pair_embeds[..., :16]
    right = pair_embeds[..., 16:]
    return torch.stack([left, right], dim=2).reshape(pair_embeds.shape[0], 16, 16)


def lut_logits(byte_embeds, lut, chunk=BYTE_CHUNK):
    """Negative squared-distance logits against the frozen 256-entry LUT.

    byte_embeds: (N, 16, 16)
    lut: (256, 16)
    returns: (N*16, 256)
    """
    flat = byte_embeds.reshape(-1, 16)
    lut_sq = (lut * lut).sum(dim=1)  # (256,)
    outs = []
    for i in range(0, flat.shape[0], chunk):
        x = flat[i:i + chunk]
        x_sq = (x * x).sum(dim=1, keepdim=True)  # (B,1)
        logits = -(x_sq + lut_sq.unsqueeze(0) - 2.0 * (x @ lut.t()))
        outs.append(logits)
    return torch.cat(outs, dim=0)


def roundtrip_loss(model, x_648, pair_embeds, window_bytes, l1_model, lut):
    """True round-trip objective.

    648 -> L2 -> 648 -> frozen L1 decode -> byte logits against frozen L0 LUT.
    Plus pair-space and code-cycle regularizers.
    """
    y_recon = model(x_648)
    decoded_pairs = decode_pair_embeds(y_recon, l1_model)
    pair_mse = ((decoded_pairs - pair_embeds) ** 2).mean()
    pair_sign = torch.relu(-decoded_pairs * torch.sign(pair_embeds)).mean()

    decoded_bytes = split_pair_embeds_to_bytes(decoded_pairs)
    logits = lut_logits(decoded_bytes, lut)
    byte_targets = window_bytes.reshape(-1)
    byte_ce = F.cross_entropy(logits, byte_targets)

    z = model.latent(x_648)
    z_cycle = model.latent(y_recon)
    code_mse = ((z_cycle - z.detach()) ** 2).mean()
    code_sign = torch.relu(-z_cycle * torch.sign(z.detach())).mean()

    loss = (
        byte_ce
        + LOSS_W_PAIR_MSE * pair_mse
        + LOSS_W_PAIR_SIGN * pair_sign
        + LOSS_W_CODE_MSE * code_mse
        + LOSS_W_CODE_SIGN * code_sign
    )
    aux = {
        "byte_ce": float(byte_ce.detach().item()),
        "pair_mse": float(pair_mse.detach().item()),
        "pair_sign": float(pair_sign.detach().item()),
        "code_mse": float(code_mse.detach().item()),
        "code_sign": float(code_sign.detach().item()),
    }
    return loss, aux


def measure_roundtrip(model, x_648, pair_embeds, window_bytes, l1_model, lut):
    """Full metrics for the user's actual goal.

    Returns diagnostics on pair space plus the real byte round-trip metric and
    top-code stability after encode -> decode -> encode.
    """
    with torch.no_grad():
        y_recon_648 = model(x_648)
        decoded = decode_pair_embeds(y_recon_648, l1_model)  # (N, 8, 32)
        per_dim = (torch.sign(decoded) == torch.sign(pair_embeds))
        per_pair = per_dim.all(dim=2)  # (N, 8)
        per_dim_rate = per_dim.float().mean().item()
        per_pair_rate = per_pair.float().mean().item()
        exact_pair_window_rate = per_pair.all(dim=1).float().mean().item()

        decoded_bytes = split_pair_embeds_to_bytes(decoded)
        logits = lut_logits(decoded_bytes, lut)
        pred_bytes = logits.argmax(dim=1).reshape(-1, 16)
        byte_match = pred_bytes == window_bytes
        byte_acc = byte_match.float().mean().item()
        byte_exact_window_rate = byte_match.all(dim=1).float().mean().item()

        z = model.latent(x_648)
        z_cycle = model.latent(y_recon_648)
        code_sign = torch.sign(z_cycle) == torch.sign(z)
        code_dim_rate = code_sign.float().mean().item()
        code_sample_rate = code_sign.all(dim=1).float().mean().item()

    return {
        "pair_per_dim": per_dim_rate,
        "pair_per_pair": per_pair_rate,
        "pair_exact_window": exact_pair_window_rate,
        "byte_acc": byte_acc,
        "byte_exact_window": byte_exact_window_rate,
        "code_sign_dim": code_dim_rate,
        "code_sign_sample": code_sample_rate,
    }


def train_one(activation, bottleneck, hidden, X_train, P_train, B_train, X_val, P_val, B_val, l1, lut,
              adam_epochs=500, lbfgs_outer=30):
    """One sweep run — 2-layer tied bottleneck, byte-space target."""
    torch.manual_seed(42)
    model = L2Mirror(648, hidden, bottleneck, activation, DEVICE).to(DEVICE)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    P_t = torch.tensor(P_train, dtype=torch.float32, device=DEVICE)
    B_t = torch.tensor(B_train, dtype=torch.long, device=DEVICE)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    P_v = torch.tensor(P_val, dtype=torch.float32, device=DEVICE)
    B_v = torch.tensor(B_val, dtype=torch.long, device=DEVICE)

    # Phase 1: Adam
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(adam_epochs):
        opt.zero_grad()
        loss, aux = roundtrip_loss(model, X_t, P_t, B_t, l1, lut)
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            tr = measure_roundtrip(model, X_t, P_t, B_t, l1, lut)
            va = measure_roundtrip(model, X_v, P_v, B_v, l1, lut)
            print(f"    [{activation} b={bottleneck}] adam {ep+1}: loss={loss.item():.4f} "
                  f"byte_ce={aux['byte_ce']:.3f} "
                  f"train byte/exact/code={100*tr['byte_acc']:.1f}/{100*tr['byte_exact_window']:.2f}/{100*tr['code_sign_sample']:.2f}% "
                  f"val byte/exact/code={100*va['byte_acc']:.1f}/{100*va['byte_exact_window']:.2f}/{100*va['code_sign_sample']:.2f}%",
                  flush=True)

    # Phase 2: LBFGS
    opt = torch.optim.LBFGS(model.parameters(), max_iter=50,
                             tolerance_grad=1e-10, tolerance_change=1e-12,
                             line_search_fn="strong_wolfe")
    best_ew = 0.0
    stall = 0
    for outer in range(lbfgs_outer):
        def closure():
            opt.zero_grad()
            loss, _ = roundtrip_loss(model, X_t, P_t, B_t, l1, lut)
            loss.backward()
            return loss
        loss = opt.step(closure).item()
        va = measure_roundtrip(model, X_v, P_v, B_v, l1, lut)
        ew = va["byte_exact_window"]
        if ew > best_ew + 1e-4:
            best_ew = ew
            stall = 0
        else:
            stall += 1
        if (outer + 1) % 5 == 0:
            print(f"    [{activation} b={bottleneck}] lbfgs {outer+1}: loss={loss:.6f} "
                  f"val byte/exact/code={100*va['byte_acc']:.1f}/{100*va['byte_exact_window']:.2f}/{100*va['code_sign_sample']:.2f}% "
                  f"pair pd/pp={100*va['pair_per_dim']:.1f}/{100*va['pair_per_pair']:.1f}%",
                  flush=True)
        if stall >= 5:
            print(f"    [{activation} b={bottleneck}] plateau @ lbfgs {outer+1}", flush=True)
            break

    # Final: both train and val
    tr = measure_roundtrip(model, X_t, P_t, B_t, l1, lut)
    va = measure_roundtrip(model, X_v, P_v, B_v, l1, lut)

    n_params = sum(p.numel() for p in model.parameters())
    dt = time.time() - t0
    return {
        "activation": activation,
        "bottleneck": bottleneck,
        "hidden": hidden,
        "params": n_params,
        "train": tr,
        "val": va,
        "wall_s": dt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", nargs="+", default=ACTIVATIONS)
    parser.add_argument("--bottlenecks", nargs="+", type=int, default=BOTTLENECKS)
    parser.add_argument("--hidden", type=int, default=HIDDEN_DIM)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--adam-epochs", type=int, default=ADAM_EPOCHS)
    parser.add_argument("--lbfgs-outer", type=int, default=LBFGS_OUTER)
    args = parser.parse_args()

    print("=" * 70)
    print(f"L2-v1 SWEEP (2-layer tied bottleneck, byte-space loss)")
    print(f"  activations={args.activations}  bottlenecks={args.bottlenecks}  hidden={args.hidden}")
    print(f"  train={args.n_train}, val={args.n_val}")
    print("=" * 70)

    print(f"\n[1] Loading L1 + L0 LUT + corpus...")
    l1 = load_l1_champion()
    lut = load_l0_lut()
    corpus = load_corpus_bytes()
    print(f"    corpus: {len(corpus)} bytes")

    n_total = args.n_train + args.n_val
    print(f"\n[2] Collecting {n_total} L1 hidden windows + pair embeds...")
    X_all, P_all, B_all = collect_l1_hidden_windows(l1, lut, corpus, n_windows=n_total)
    X_train, X_val = X_all[:args.n_train], X_all[args.n_train:]
    P_train, P_val = P_all[:args.n_train], P_all[args.n_train:]
    B_train, B_val = B_all[:args.n_train], B_all[args.n_train:]
    print(f"    train X{X_train.shape} P{P_train.shape} B{B_train.shape}  "
          f"val X{X_val.shape} P{P_val.shape} B{B_val.shape}")

    # Sanity: the frozen lower stack itself must round-trip perfectly on these windows.
    X_id = torch.tensor(X_val[: min(len(X_val), 128)], dtype=torch.float32, device=DEVICE)
    P_id = torch.tensor(P_val[: min(len(P_val), 128)], dtype=torch.float32, device=DEVICE)
    B_id = torch.tensor(B_val[: min(len(B_val), 128)], dtype=torch.long, device=DEVICE)
    class IdentityModel(nn.Module):
        def forward(self, x): return x
        def latent(self, x): return x
    ident = IdentityModel()
    sanity = measure_roundtrip(ident, X_id, P_id, B_id, l1, lut)
    print(f"    sanity lower-stack byte_exact={100*sanity['byte_exact_window']:.2f}% "
          f"byte_acc={100*sanity['byte_acc']:.2f}% code_stable={100*sanity['code_sign_sample']:.2f}%")

    results = []
    for act in args.activations:
        for b in args.bottlenecks:
            print(f"\n[run] activation={act} bottleneck={b} hidden={args.hidden}")
            r = train_one(act, b, args.hidden, X_train, P_train, B_train, X_val, P_val, B_val, l1, lut,
                          adam_epochs=args.adam_epochs, lbfgs_outer=args.lbfgs_outer)
            results.append(r)
            tr = r["train"]; va = r["val"]
            print(f"  -> FINAL: train byte/exact/code = {100*tr['byte_acc']:.1f}/{100*tr['byte_exact_window']:.2f}/{100*tr['code_sign_sample']:.2f}%  "
                  f"val byte/exact/code = {100*va['byte_acc']:.1f}/{100*va['byte_exact_window']:.2f}/{100*va['code_sign_sample']:.2f}%  "
                  f"params={r['params']}  wall={r['wall_s']:.0f}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — true byte round-trip + top-code stability")
    print(f"{'='*70}")
    print(
        f"{'act':>9} {'bot':>4} {'H':>4} | "
        f"{'train byte':>10} {'train exact':>11} {'train code':>10} | "
        f"{'val byte':>8} {'val exact':>9} {'val code':>8} | "
        f"{'val ppair':>9} {'val pwin':>9} | {'params':>8} {'wall':>6}"
    )
    for r in results:
        tr = r["train"]; va = r["val"]
        print(
            f"{r['activation']:>9} {r['bottleneck']:>4} {r['hidden']:>4} | "
            f"{100*tr['byte_acc']:>9.2f}% {100*tr['byte_exact_window']:>10.2f}% {100*tr['code_sign_sample']:>9.2f}% | "
            f"{100*va['byte_acc']:>7.2f}% {100*va['byte_exact_window']:>8.2f}% {100*va['code_sign_sample']:>7.2f}% | "
            f"{100*va['pair_per_pair']:>8.2f}% {100*va['pair_exact_window']:>8.2f}% | "
            f"{r['params']:>8d} {r['wall_s']:>5.0f}s"
        )

    with open(OUT_DIR / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
