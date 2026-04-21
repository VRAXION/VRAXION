"""Modal app: train Block C on remote compute (CPU / T4 / L4 / A10G).

The training code lives in tools/diag_block_c_torch.py — this module wraps
it for Modal. We upload the pre-tokenised corpus (int32 npy, ~10-40 MB)
and vocab JSON once to a Modal Volume, then launch training Functions
that read from the volume.

Usage:
    # One-time (or when corpus/vocab change):
    modal run tools/modal_block_c.py::upload_data \\
        --tokens output/data/fineweb_edu_100mb.tokens.npy \\
        --vocab  output/word_tokenizer_champion/champion_vocab.json

    # Benchmark smoke run (1 epoch, 500k tokens, compare tiers):
    modal run tools/modal_block_c.py::benchmark

    # Production sweep:
    modal run tools/modal_block_c.py::sweep \\
        --gpu T4 --epochs 8 --e 32 --hidden 128 \\
        --seeds "1,2,3,4,5"

    # Pull artifacts back:
    modal volume get vraxion-block-c /out ./output/modal_pull
"""
from __future__ import annotations

import time
from pathlib import Path

import modal

APP_NAME = "vraxion-block-c"
VOL_NAME = "vraxion-block-c"

app = modal.App(APP_NAME)

# Image: slim debian + torch. Modal's pip install picks up the CUDA wheel
# automatically when the Function is scheduled with a GPU (same torch
# package, different wheel). The training script is shipped as a single
# file into /app, which we add to sys.path inside the container.
_TORCH_SCRIPT = Path(__file__).resolve().parent / "diag_block_c_torch.py"
_BYTEPAIR_SCRIPT = Path(__file__).resolve().parent / "diag_block_c_bytepair_poc.py"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy==2.1.3")
    .add_local_file(str(_TORCH_SCRIPT),    remote_path="/app/diag_block_c_torch.py")
    .add_local_file(str(_BYTEPAIR_SCRIPT), remote_path="/app/diag_block_c_bytepair_poc.py")
)

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


# --------------------------------------------------------------------------
# Upload: copies pre-tokenised corpus + vocab into the shared Volume.
# Run once (or whenever the corpus changes). Tiny payload (tokens < 50 MB).
# --------------------------------------------------------------------------
@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def _upload_blob(name: str, data: bytes) -> str:
    out = Path("/vol") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    vol.commit()
    return f"wrote {out} ({len(data)/1e6:.1f} MB)"


@app.local_entrypoint()
def upload_data(tokens: str, vocab: str):
    tokens_p = Path(tokens).expanduser().resolve()
    vocab_p  = Path(vocab).expanduser().resolve()
    if not tokens_p.exists():
        raise FileNotFoundError(tokens_p)
    if not vocab_p.exists():
        raise FileNotFoundError(vocab_p)
    print(f"Uploading tokens: {tokens_p}  ({tokens_p.stat().st_size/1e6:.1f} MB)")
    print(_upload_blob.remote("tokens.npy", tokens_p.read_bytes()))
    print(f"Uploading vocab:  {vocab_p}  ({vocab_p.stat().st_size/1e6:.1f} MB)")
    print(_upload_blob.remote("vocab.json", vocab_p.read_bytes()))
    print("Done.")


# --------------------------------------------------------------------------
# Training Function. Parametrised by GPU via Modal's `.with_options` at
# call time so we can reuse the same Function def for all tiers.
# --------------------------------------------------------------------------
def _train_impl(
    E: int, H: int, context: int, epochs: int, seeds: str,
    run_name: str, device_str: str, tokens_file: str = "tokens.npy",
    resume_dir: str = "", patience: int = -1,
) -> dict:
    """Runs inside the Modal container."""
    import subprocess
    import sys as _sys
    from pathlib import Path as _P

    out_dir = _P("/vol/runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = _P("/vol") / tokens_file
    vocab  = _P("/vol/vocab.json")
    if not tokens.exists():
        raise FileNotFoundError(f"{tokens} not on volume; run upload_data first")
    if not vocab.exists():
        raise FileNotFoundError(f"{vocab} not on volume")

    cmd = [
        _sys.executable, "-u", "/app/diag_block_c_torch.py",
        "--tokens", str(tokens),
        "--vocab",  str(vocab),
        "--e", str(E), "--hidden", str(H), "--context", str(context),
        "--epochs", str(epochs), "--seeds", seeds,
        "--device", device_str,
        "--out", str(out_dir),
        "--commit-hook", "/vol/_commit_signal",
        "--patience", str(patience),
    ]
    if resume_dir:
        cmd += ["--resume-dir", str(_P("/vol") / resume_dir)]

    # Start a background watcher that polls the signal file and calls
    # vol.commit() whenever the child script flips it. This makes per-epoch
    # checkpoints visible on the volume without the child knowing about the
    # Volume object.
    import threading
    signal_path = _P("/vol/_commit_signal")
    signal_path.write_text("0")
    stop = threading.Event()

    def _watcher():
        last = "0"
        while not stop.wait(timeout=5.0):
            try:
                cur = signal_path.read_text()
            except Exception:
                continue
            if cur != last:
                try:
                    vol.commit()
                    last = cur
                except Exception as e:
                    print(f"[watcher] commit failed: {e}", flush=True)
    t = threading.Thread(target=_watcher, daemon=True)
    t.start()
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    stop.set()
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        raise RuntimeError(f"training exited {r.returncode}")
    vol.commit()
    return {"run": run_name, "seconds": dt, "out": str(out_dir)}


# Variants of the training function parametrised by compute tier. Modal
# needs a static GPU spec at @app.function time, so we declare one per tier
# and pick at call time.
@app.function(image=image, volumes={"/vol": vol},
              cpu=8.0, memory=16384, timeout=2 * 3600)
def train_cpu(E: int, H: int, context: int, epochs: int, seeds: str,
              run_name: str, tokens_file: str = "tokens.npy",
              resume_dir: str = "", patience: int = -1) -> dict:
    return _train_impl(E, H, context, epochs, seeds, run_name, "cpu",
                       tokens_file, resume_dir, patience)


@app.function(image=image, volumes={"/vol": vol}, gpu="T4",
              timeout=6 * 3600)
def train_t4(E: int, H: int, context: int, epochs: int, seeds: str,
             run_name: str, tokens_file: str = "tokens.npy",
             resume_dir: str = "", patience: int = -1) -> dict:
    return _train_impl(E, H, context, epochs, seeds, run_name, "cuda",
                       tokens_file, resume_dir, patience)


@app.function(image=image, volumes={"/vol": vol}, gpu="L4",
              timeout=6 * 3600)
def train_l4(E: int, H: int, context: int, epochs: int, seeds: str,
             run_name: str, tokens_file: str = "tokens.npy",
             resume_dir: str = "", patience: int = -1) -> dict:
    return _train_impl(E, H, context, epochs, seeds, run_name, "cuda",
                       tokens_file, resume_dir, patience)


@app.function(image=image, volumes={"/vol": vol}, gpu="A10G",
              timeout=6 * 3600)
def train_a10g(E: int, H: int, context: int, epochs: int, seeds: str,
               run_name: str, tokens_file: str = "tokens.npy",
               resume_dir: str = "", patience: int = -1) -> dict:
    return _train_impl(E, H, context, epochs, seeds, run_name, "cuda",
                       tokens_file, resume_dir, patience)


_TIER = {
    "cpu":  train_cpu,
    "T4":   train_t4,
    "L4":   train_l4,
    "A10G": train_a10g,
}


# --------------------------------------------------------------------------
# Benchmark: same smoke run on every tier, report wall time + $ estimate.
# --------------------------------------------------------------------------
# Rough 2026 Modal prices (per hour, on-demand). Update if pricing changes.
PRICE_PER_HR = {
    "cpu":  0.24,   # 8 CPU cores; Modal bills ~$0.03/core/hr
    "T4":   0.59,
    "L4":   0.80,
    "A10G": 1.10,
}


@app.local_entrypoint()
def benchmark():
    """Launch the same 1-epoch smoke run on every tier in parallel."""
    E, H, context, epochs, seeds = 32, 128, 8, 1, "1"
    print("Launching smoke runs on cpu, T4, L4, A10G (1 epoch, seed=1)...")
    handles = {}
    for tier, fn in _TIER.items():
        run_name = f"bench_{tier}_{int(time.time())}"
        handles[tier] = (fn.spawn(E, H, context, epochs, seeds, run_name), run_name)

    results = {}
    for tier, (h, run_name) in handles.items():
        try:
            r = h.get()
            results[tier] = r
            print(f"[{tier:>4}]  {r['seconds']:7.1f} s  "
                  f"(~${r['seconds']/3600 * PRICE_PER_HR[tier]:.4f}) "
                  f"out={r['out']}")
        except Exception as e:
            print(f"[{tier:>4}]  FAILED: {e}")

    if results:
        fastest = min(results.values(), key=lambda r: r["seconds"])
        cheapest = min(
            results.items(),
            key=lambda kv: kv[1]["seconds"] / 3600 * PRICE_PER_HR[kv[0]],
        )
        print(f"\nFastest:  {fastest['run']}  ({fastest['seconds']:.1f}s)")
        print(f"Cheapest: {cheapest[0]}  "
              f"(${cheapest[1]['seconds']/3600 * PRICE_PER_HR[cheapest[0]]:.4f})")


# --------------------------------------------------------------------------
# Production sweep entry point — PARALLEL: one container per (E, seed).
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Byte-pair PoC (ABC pipeline): train Block C directly on byte-pair IDs.
# --------------------------------------------------------------------------
def _bytepair_impl(E: int, H: int, context: int, epochs: int, seeds: str,
                   run_name: str, corpus_file: str, max_bytes: int,
                   device_str: str, lr: float = 0.1,
                   warm_emb_template: str = "", lr_decay: bool = False) -> dict:
    import subprocess, sys as _sys, threading
    from pathlib import Path as _P
    out_dir = _P("/vol/runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = _P("/vol") / corpus_file
    if not corpus.exists():
        raise FileNotFoundError(f"{corpus} not on volume")

    signal_path = _P("/vol/_commit_signal")
    signal_path.write_text("0")
    stop = threading.Event()
    def _watcher():
        last = "0"
        while not stop.wait(5.0):
            try: cur = signal_path.read_text()
            except Exception: continue
            if cur != last:
                try: vol.commit(); last = cur
                except Exception as e: print(f"[watcher] {e}", flush=True)
    t = threading.Thread(target=_watcher, daemon=True); t.start()

    cmd = [_sys.executable, "-u", "/app/diag_block_c_bytepair_poc.py",
           "--corpus", str(corpus),
           "--max-bytes", str(max_bytes),
           "--e", str(E), "--hidden", str(H), "--context", str(context),
           "--epochs", str(epochs), "--seeds", seeds,
           "--device", device_str,
           "--commit-hook", "/vol/_commit_signal",
           "--lr", str(lr),
           "--out", str(out_dir)]
    if warm_emb_template:
        cmd += ["--warm-emb-template", warm_emb_template]
    if lr_decay:
        cmd += ["--lr-decay"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    stop.set()
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        raise RuntimeError(f"bytepair exit {r.returncode}")
    vol.commit()
    return {"run": run_name, "seconds": dt, "out": str(out_dir)}


@app.function(image=image, volumes={"/vol": vol}, gpu="L4", timeout=3 * 3600)
def train_bytepair_l4(E: int, H: int, context: int, epochs: int, seeds: str,
                      run_name: str, corpus_file: str = "fineweb_edu_100mb.txt",
                      max_bytes: int = 100_000_000, lr: float = 0.1,
                      warm_emb_template: str = "", lr_decay: bool = False) -> dict:
    return _bytepair_impl(E, H, context, epochs, seeds, run_name,
                          corpus_file, max_bytes, "cuda", lr,
                          warm_emb_template, lr_decay)


@app.local_entrypoint()
def bytepair_poc(e: int = 32, hidden: int = 128, context: int = 16,
                 epochs: int = 8, seeds: str = "1,3,7",
                 corpus_file: str = "fineweb_edu_100mb.txt",
                 max_bytes: int = 100_000_000,
                 lr: float = 0.1,
                 warm_emb_template: str = "",
                 lr_decay: bool = False,
                 run_prefix: str = ""):
    """Byte-pair PoC: train Block C over byte-pair IDs (vocab=65536) on L4.

    warm_emb_template may use {seed} placeholder, e.g.
    '/vol/runs/bytepair_100mb_E32_8ep/seed{seed}/seed_{seed}/emb_E32_epoch03.npy'
    to resume from the peak of a previous diverging run.
    """
    if not run_prefix:
        run_prefix = f"bytepair_poc_{int(time.time())}"
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    print(f"Byte-pair PoC: E={e} H={hidden} context={context} "
          f"epochs={epochs} seeds={seed_list} max_bytes={max_bytes/1e6:.0f}MB "
          f"lr={lr} lr_decay={lr_decay} "
          f"warm={bool(warm_emb_template)}")

    handles = {}
    for seed in seed_list:
        run_name = f"{run_prefix}/seed{seed}"
        h = train_bytepair_l4.spawn(e, hidden, context, epochs, str(seed),
                                    run_name, corpus_file, max_bytes,
                                    lr, warm_emb_template, lr_decay)
        handles[seed] = (h, run_name)

    total_sec = 0.0
    for seed, (h, run_name) in handles.items():
        r = h.get()
        total_sec += r["seconds"]
        print(f"  seed={seed}  {r['seconds']/60:.1f} min  out={r['out']}")
    print(f"\nTotal compute: {total_sec/60:.1f} min  "
          f"(cost ~${total_sec/3600 * PRICE_PER_HR['L4']:.3f})")
    print(f"Pull: modal volume get {VOL_NAME} runs/{run_prefix} ./output/modal_pull/")


@app.local_entrypoint()
def sweep(gpu: str = "L4",
          e_list: str = "32",           # e.g. "32,48,64"
          hidden: int = 128,
          context: int = 8,
          epochs: int = 8,
          seeds: str = "1,2,3,4,5",
          tokens_file: str = "tokens.npy",
          max_parallel: int = 5,
          run_prefix: str = "",
          resume_sweep: str = "",
          patience: int = -1):
    """Fan out (E x seed) grid as independent containers on the chosen tier.

    max_parallel caps how many containers are in-flight at once so we stay
    under Modal's per-account GPU concurrency limit.

    resume_sweep: name of a previous sweep dir on the volume (e.g.
    'prod_100mb_8ep_v2'). If set, each (E, seed) container resumes from
    'runs/<resume_sweep>/E<E>_seed<seed>/seed_<seed>/' which holds the
    best-state emb + W_out + W1 from that run. SGD momentum is reset.
    """
    if gpu not in _TIER:
        raise ValueError(f"Unknown tier {gpu!r}; choose from {list(_TIER)}")
    fn = _TIER[gpu]
    Es = [int(e.strip()) for e in e_list.split(",") if e.strip()]
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    if not run_prefix:
        run_prefix = f"sweep_{gpu}_{int(time.time())}"
    grid = [(E, s) for E in Es for s in seed_list]
    print(f"Launching {len(grid)} {gpu} containers "
          f"(E \u2208 {Es}, seeds \u2208 {seed_list}, {epochs} epochs each, "
          f"max_parallel={max_parallel})...")

    results = {}
    total_sec = 0.0
    i = 0
    while i < len(grid):
        batch = grid[i : i + max_parallel]
        i += max_parallel
        print(f"\n-- launching batch of {len(batch)}: {batch} --")
        handles = {}
        for E, seed in batch:
            run_name = f"{run_prefix}/E{E}_seed{seed}"
            resume_dir = ""
            if resume_sweep:
                resume_dir = (f"runs/{resume_sweep}/E{E}_seed{seed}/"
                              f"seed_{seed}")
            h = fn.spawn(E, hidden, context, epochs, str(seed),
                         run_name, tokens_file, resume_dir, patience)
            handles[(E, seed)] = (h, run_name)
        for (E, seed), (h, run_name) in handles.items():
            try:
                r = h.get()
                results[(E, seed)] = r
                total_sec += r["seconds"]
                print(f"  [E={E} seed={seed}]  {r['seconds']/60:6.1f} min  "
                      f"(~${r['seconds']/3600 * PRICE_PER_HR[gpu]:.3f})  "
                      f"out={r['out']}")
            except Exception as exc:
                print(f"  [E={E} seed={seed}]  FAILED: {exc}")

    if results:
        wall_per_batch = max(r["seconds"] for r in results.values()) / 60
        n_batches = (len(grid) + max_parallel - 1) // max_parallel
        total_cost = total_sec / 3600 * PRICE_PER_HR[gpu]
        print(f"\n== Sweep complete ==")
        print(f"  runs: {len(results)}/{len(grid)} in {n_batches} batches")
        print(f"  wall per batch (approx): {wall_per_batch:.1f} min")
        print(f"  compute total: {total_sec/60:.1f} min  (cost ~${total_cost:.2f})")
        print(f"\nPull artifacts locally:")
        print(f"  modal volume get {VOL_NAME} runs/{run_prefix} ./output/modal_pull/")
