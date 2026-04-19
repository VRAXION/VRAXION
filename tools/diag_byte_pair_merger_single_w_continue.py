"""Continue LBFGS training on the best Single-W model from prior run.

Loads output/merger_single_w_test/best_model.pt and runs long LBFGS
(stall=20, max 300 outer) to try to push past 99.65% to 100% lossless.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_single_w_mirror import (
    SingleWMirror, load_byte_pairs, metrics, DEVICE,
)

BEST_STATE = "output/merger_single_w_test/best_model.pt"
OUT_DIR = Path("output/merger_single_w_continue")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_lbfgs_long(model, data, max_outer=300, tol_stall=20):
    opt = torch.optim.LBFGS(model.parameters(), max_iter=100,
                            tolerance_grad=1e-14, tolerance_change=1e-16,
                            history_size=100,
                            line_search_fn="strong_wolfe")
    best_ll = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stall = 0
    for outer in range(max_outer):
        def closure():
            opt.zero_grad()
            y = model(data)
            mse = ((y - data) ** 2).mean()
            sign_hinge = torch.relu(-y * torch.sign(data)).mean()
            loss = mse + 0.5 * sign_hinge
            loss.backward()
            return loss
        loss = opt.step(closure).item()
        ll, pd, bp = metrics(model, data)
        if ll > best_ll + 0.001:
            best_ll = ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
        if (outer + 1) % 10 == 0:
            print(f"  [long-lbfgs outer {outer+1}] loss={loss:.6e} ll={ll:.6f}% bad={bp} stall={stall} best={best_ll:.4f}%", flush=True)
        if ll >= 100.0:
            print(f"  -> LOSSLESS @ outer {outer+1}", flush=True)
            return True, best_state
        if stall >= tol_stall:
            print(f"  -> plateau (stall={stall}), best={best_ll:.4f}%", flush=True)
            model.load_state_dict(best_state)
            return False, best_state
    return False, best_state


def main():
    print("=== CONTINUE LBFGS on Best Single-W Model ===", flush=True)
    model = SingleWMirror(32, 81, DEVICE).to(DEVICE)
    state = torch.load(BEST_STATE, map_location=DEVICE)
    model.load_state_dict(state)
    data = load_byte_pairs().to(DEVICE)

    ll0, pd0, bp0 = metrics(model, data)
    print(f"Loaded state: ll={ll0:.4f}% pd={pd0:.4f}% bad={bp0}", flush=True)

    print(f"\n--- Long LBFGS (max=300, stall=20, history=100) ---", flush=True)
    t0 = time.time()
    lossless, best_state = train_lbfgs_long(model, data, max_outer=300, tol_stall=20)
    t_total = time.time() - t0

    ll, pd, bp = metrics(model, data)
    print(f"\n=== DONE ===")
    print(f"  Final: ll={ll:.4f}% pd={pd:.4f}% bad={bp}")
    print(f"  Time: {t_total:.0f}s")

    torch.save(best_state, OUT_DIR / "best_model.pt")

    # Save artifact
    state_dict = {
        "architecture": "single-W mirror tied (continued LBFGS)",
        "H": 81, "in_dim": 32, "out_dim": 32,
        "final_lossless": ll,
        "final_bad_pairs": bp,
        "weights_count": 32 * 81,
        "W": model.W.data.cpu().numpy().tolist(),
        "b1": model.b1.data.cpu().numpy().tolist(),
        "b2": model.b2.data.cpu().numpy().tolist(),
        "c19_c": model.c19.c_raw.data.cpu().numpy().tolist(),
        "c19_rho": model.c19.rho_raw.data.cpu().numpy().tolist(),
    }
    with open(OUT_DIR / "final_model.json", "w") as f:
        json.dump(state_dict, f)
    print(f"  Saved: {OUT_DIR / 'final_model.json'}")


if __name__ == "__main__":
    main()
