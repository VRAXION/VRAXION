"""Final push: heavy sign-hinge + long LBFGS to squeeze the last 1 bad pair.

Loads the 99.9985% model (1 bad pair) and tries to close it with
a much heavier sign-hinge weight (5.0 vs 0.5).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import torch
import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_single_w_mirror import (
    SingleWMirror, load_byte_pairs, metrics, DEVICE,
)

BEST = "output/merger_single_w_continue/best_model.pt"
OUT_DIR = Path("output/merger_single_w_final_push")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def push_lbfgs(model, data, hinge_weight=5.0, max_outer=400, tol_stall=30):
    opt = torch.optim.LBFGS(model.parameters(), max_iter=100,
                            tolerance_grad=1e-16, tolerance_change=1e-18,
                            history_size=200,
                            line_search_fn="strong_wolfe")
    best_ll = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stall = 0
    for outer in range(max_outer):
        def closure():
            opt.zero_grad()
            y = model(data)
            mse = ((y - data) ** 2).mean()
            # Heavy sign hinge
            sign_err = torch.relu(-y * torch.sign(data))
            hinge = sign_err.mean()
            loss = mse + hinge_weight * hinge
            loss.backward()
            return loss
        loss = opt.step(closure).item()
        ll, pd, bp = metrics(model, data)
        if ll > best_ll + 1e-6:
            best_ll = ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
        if (outer + 1) % 10 == 0 or ll >= 100.0:
            print(f"  [push outer {outer+1}] loss={loss:.6e} ll={ll:.6f}% bad={bp} stall={stall} best={best_ll:.6f}%", flush=True)
        if ll >= 100.0:
            print(f"  -> LOSSLESS @ outer {outer+1}", flush=True)
            return True, best_state
        if stall >= tol_stall:
            print(f"  -> plateau, best={best_ll:.6f}%", flush=True)
            model.load_state_dict(best_state)
            return False, best_state
    model.load_state_dict(best_state)
    return False, best_state


def main():
    print("=== FINAL PUSH (hinge 5.0 + long LBFGS) ===", flush=True)
    model = SingleWMirror(32, 81, DEVICE).to(DEVICE)
    state = torch.load(BEST, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    data = load_byte_pairs().to(DEVICE)

    ll0, pd0, bp0 = metrics(model, data)
    print(f"Loaded: ll={ll0:.6f}% bad={bp0}", flush=True)

    t0 = time.time()
    lossless, best_state = push_lbfgs(model, data, hinge_weight=5.0, max_outer=400, tol_stall=30)
    t = time.time() - t0

    ll, pd, bp = metrics(model, data)
    print(f"\n=== DONE === final ll={ll:.6f}% bad={bp} time={t:.0f}s", flush=True)

    torch.save(best_state, OUT_DIR / "best_model.pt")
    with open(OUT_DIR / "final_model.json", "w") as f:
        json.dump({
            "architecture": "single-W final push (hinge=5.0)",
            "H": 81, "in_dim": 32, "out_dim": 32,
            "final_lossless": ll, "final_bad_pairs": bp,
            "weights_count": 2592,
            "W": model.W.data.cpu().numpy().tolist(),
            "b1": model.b1.data.cpu().numpy().tolist(),
            "b2": model.b2.data.cpu().numpy().tolist(),
            "c19_c": model.c19.c_raw.data.cpu().numpy().tolist(),
            "c19_rho": model.c19.rho_raw.data.cpu().numpy().tolist(),
        }, f)
    print(f"Saved: {OUT_DIR / 'final_model.json'}")


if __name__ == "__main__":
    main()
