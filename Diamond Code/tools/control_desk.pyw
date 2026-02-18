"""
VRAXION Control Desk — Standalone training control panel.
Directly reads/writes controls.json. No web server needed.
Launch: double-click or pin to taskbar.
"""

import json
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path

CONTROLS_PATH = Path(__file__).resolve().parent.parent / "logs" / "swarm" / "controls.json"

EFFORT_TIERS = {
    "Alpha":   {"tt": 0,  "lcx": False, "batch": 500, "name": "Reflex"},
    "Beta":    {"tt": 1,  "lcx": True,  "batch": 500, "name": "Recall"},
    "Gamma":   {"tt": 2,  "lcx": True,  "batch": 500, "name": "Reason"},
    "Delta":   {"tt": 4,  "lcx": True,  "batch": 500, "name": "Depth"},
    "Epsilon": {"tt": 8,  "lcx": True,  "batch": 500, "name": "Emergence"},
    "Zeta":    {"tt": 16, "lcx": True,  "batch": 250, "name": "Zenith"},
}

# ── Read / Write ──────────────────────────────────────────────

def load_controls():
    try:
        with open(CONTROLS_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_controls(data):
    CONTROLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTROLS_PATH, 'w') as f:
        json.dump(data, f, indent=2)

# ── App ───────────────────────────────────────────────────────

class ControlDesk:
    def __init__(self, root):
        self.root = root
        self.root.title("VRAXION Control Desk")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")
        self.controls = load_controls()
        self._building = True

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#1a1a2e", foreground="#e0e0e0",
                         fieldbackground="#16213e", font=("Consolas", 10))
        style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0",
                         font=("Consolas", 10))
        style.configure("Header.TLabel", background="#1a1a2e", foreground="#00d4ff",
                         font=("Consolas", 12, "bold"))
        style.configure("Section.TLabel", background="#1a1a2e", foreground="#ffa500",
                         font=("Consolas", 10, "bold"))
        style.configure("Status.TLabel", background="#0f3460", foreground="#00ff88",
                         font=("Consolas", 9), padding=4)
        style.configure("TCombobox", fieldbackground="#16213e", foreground="#e0e0e0",
                         selectbackground="#0f3460", selectforeground="#e0e0e0")
        style.configure("TCheckbutton", background="#1a1a2e", foreground="#e0e0e0",
                         font=("Consolas", 10))
        style.configure("TScale", background="#1a1a2e", troughcolor="#16213e")
        style.configure("TSpinbox", fieldbackground="#16213e", foreground="#e0e0e0")
        style.configure("Active.TCheckbutton", background="#1a1a2e", foreground="#00ff88")
        style.map("TCheckbutton", background=[("active", "#1a1a2e")])
        style.map("Active.TCheckbutton", background=[("active", "#1a1a2e")])

        main = ttk.Frame(root)
        main.pack(padx=12, pady=8)

        # ── Header ──
        ttk.Label(main, text="VRAXION Control Desk", style="Header.TLabel").pack(pady=(0, 8))

        # ── Effort Tier ──
        ef = ttk.Frame(main)
        ef.pack(fill="x", pady=4)
        ttk.Label(ef, text="EFFORT", style="Section.TLabel").pack(side="left")
        self.effort_var = tk.StringVar(value=self.controls.get("effort", "Beta"))
        cb = ttk.Combobox(ef, textvariable=self.effort_var,
                          values=list(EFFORT_TIERS.keys()), state="readonly", width=12)
        cb.pack(side="left", padx=(8, 0))
        cb.bind("<<ComboboxSelected>>", self._on_effort_change)
        self.effort_label = ttk.Label(ef, text="")
        self.effort_label.pack(side="left", padx=(8, 0))
        self._update_effort_label()

        # ── Core Controls ──
        ttk.Label(main, text="CORE", style="Section.TLabel").pack(anchor="w", pady=(8, 2))
        core = ttk.Frame(main)
        core.pack(fill="x")

        # LR
        r0 = ttk.Frame(core)
        r0.pack(fill="x", pady=2)
        ttk.Label(r0, text="LR:", width=14).pack(side="left")
        self.lr_var = tk.StringVar(value=str(self.controls.get("lr", 0.0003)))
        lr_entry = ttk.Entry(r0, textvariable=self.lr_var, width=14)
        lr_entry.pack(side="left")
        lr_entry.bind("<Return>", self._on_lr_change)
        lr_entry.bind("<FocusOut>", self._on_lr_change)

        # Think Ticks
        r1 = ttk.Frame(core)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Think Ticks:", width=14).pack(side="left")
        self.tt_var = tk.IntVar(value=self.controls.get("think_ticks", 1))
        tt_spin = ttk.Spinbox(r1, from_=0, to=32, textvariable=self.tt_var, width=6,
                               command=self._on_core_change)
        tt_spin.pack(side="left")

        # Batch Size
        r2 = ttk.Frame(core)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Batch Size:", width=14).pack(side="left")
        self.batch_var = tk.IntVar(value=self.controls.get("batch_size", 500))
        batch_spin = ttk.Spinbox(r2, from_=1, to=1000, textvariable=self.batch_var, width=6,
                                  command=self._on_core_change)
        batch_spin.pack(side="left")

        # Use LCX
        r3 = ttk.Frame(core)
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Use LCX:", width=14).pack(side="left")
        self.lcx_var = tk.BooleanVar(value=self.controls.get("use_lcx", True))
        ttk.Checkbutton(r3, variable=self.lcx_var, text="ON",
                         command=self._on_core_change).pack(side="left")

        # Eval Every
        r4 = ttk.Frame(core)
        r4.pack(fill="x", pady=2)
        ttk.Label(r4, text="Eval Every:", width=14).pack(side="left")
        self.eval_var = tk.IntVar(value=self.controls.get("eval_every", 5))
        eval_spin = ttk.Spinbox(r4, from_=1, to=500, textvariable=self.eval_var, width=6,
                                 command=self._on_core_change)
        eval_spin.pack(side="left")

        # Checkpoint Every
        r5 = ttk.Frame(core)
        r5.pack(fill="x", pady=2)
        ttk.Label(r5, text="Ckpt Every:", width=14).pack(side="left")
        self.ckpt_var = tk.IntVar(value=self.controls.get("checkpoint_every", 100))
        ckpt_spin = ttk.Spinbox(r5, from_=10, to=5000, textvariable=self.ckpt_var, width=6,
                                 command=self._on_core_change)
        ckpt_spin.pack(side="left")

        # ── Data Weights ──
        ttk.Label(main, text="DATA", style="Section.TLabel").pack(anchor="w", pady=(8, 2))
        dw_frame = ttk.Frame(main)
        dw_frame.pack(fill="x")

        self.data_vars = {}
        weights = self.controls.get("data_weights", {})
        for i, (name, w) in enumerate(sorted(weights.items())):
            short = name.replace(".traindat", "")
            var = tk.BooleanVar(value=(w > 0))
            self.data_vars[name] = var
            cb = ttk.Checkbutton(dw_frame, variable=var, text=short,
                                  style="Active.TCheckbutton" if w > 0 else "TCheckbutton",
                                  command=self._on_data_change)
            cb.grid(row=i // 2, column=i % 2, sticky="w", padx=4, pady=1)

        # ── Dream Controls ──
        ttk.Label(main, text="DREAM", style="Section.TLabel").pack(anchor="w", pady=(8, 2))
        dr = ttk.Frame(main)
        dr.pack(fill="x")

        dr0 = ttk.Frame(dr)
        dr0.pack(fill="x", pady=2)
        ttk.Label(dr0, text="Enabled:", width=14).pack(side="left")
        self.dream_var = tk.BooleanVar(value=self.controls.get("dream_enabled", False))
        ttk.Checkbutton(dr0, variable=self.dream_var, text="ON",
                         command=self._on_core_change).pack(side="left")

        dr1 = ttk.Frame(dr)
        dr1.pack(fill="x", pady=2)
        ttk.Label(dr1, text="Mode:", width=14).pack(side="left")
        self.dream_mode_var = tk.StringVar(value=self.controls.get("dream_mode", "consolidation"))
        ttk.Combobox(dr1, textvariable=self.dream_mode_var,
                      values=["consolidation", "rehearsal"], state="readonly", width=14).pack(side="left")
        self.dream_mode_var.trace_add("write", lambda *_: self._on_core_change())

        dr2 = ttk.Frame(dr)
        dr2.pack(fill="x", pady=2)
        ttk.Label(dr2, text="Frequency:", width=14).pack(side="left")
        self.dream_freq_var = tk.IntVar(value=self.controls.get("dream_frequency", 10))
        ttk.Spinbox(dr2, from_=1, to=100, textvariable=self.dream_freq_var, width=6,
                     command=self._on_core_change).pack(side="left")

        dr3 = ttk.Frame(dr)
        dr3.pack(fill="x", pady=2)
        ttk.Label(dr3, text="Steps:", width=14).pack(side="left")
        self.dream_steps_var = tk.IntVar(value=self.controls.get("dream_steps", 3))
        ttk.Spinbox(dr3, from_=1, to=20, textvariable=self.dream_steps_var, width=6,
                     command=self._on_core_change).pack(side="left")

        dr4 = ttk.Frame(dr)
        dr4.pack(fill="x", pady=2)
        ttk.Label(dr4, text="Dream TT:", width=14).pack(side="left")
        self.dream_tt_var = tk.IntVar(value=self.controls.get("dream_think_ticks", 2))
        ttk.Spinbox(dr4, from_=0, to=32, textvariable=self.dream_tt_var, width=6,
                     command=self._on_core_change).pack(side="left")

        # ── Status Bar ──
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self.status_var, style="Status.TLabel").pack(
            fill="x", pady=(8, 0))

        # ── Refresh timer (read file every 2s to stay in sync) ──
        self._building = False
        self._schedule_refresh()

    def _update_effort_label(self):
        eff = self.effort_var.get()
        tier = EFFORT_TIERS.get(eff, {})
        name = tier.get("name", "")
        tt = tier.get("tt", "?")
        batch = tier.get("batch", "?")
        lcx = "ON" if tier.get("lcx") else "OFF"
        self.effort_label.config(text=f"{name}  tt={tt} batch={batch} lcx={lcx}")

    def _on_effort_change(self, event=None):
        eff = self.effort_var.get()
        tier = EFFORT_TIERS.get(eff, {})
        if tier:
            self.tt_var.set(tier["tt"])
            self.batch_var.set(tier["batch"])
            self.lcx_var.set(tier["lcx"])
        self._update_effort_label()
        self._save()

    def _on_lr_change(self, event=None):
        self._save()

    def _on_core_change(self):
        if not self._building:
            self._save()

    def _on_data_change(self):
        if not self._building:
            self._save()

    def _save(self):
        try:
            lr_val = float(self.lr_var.get())
        except ValueError:
            lr_val = 0.0003

        data = load_controls()

        data["lr"] = lr_val
        data["think_ticks"] = self.tt_var.get()
        data["batch_size"] = self.batch_var.get()
        data["use_lcx"] = self.lcx_var.get()
        data["effort"] = self.effort_var.get()
        tier = EFFORT_TIERS.get(self.effort_var.get(), {})
        data["effort_name"] = tier.get("name", "")
        data["stage"] = self.effort_var.get().upper()
        data["eval_every"] = self.eval_var.get()
        data["checkpoint_every"] = self.ckpt_var.get()

        # Data weights
        weights = data.get("data_weights", {})
        for name, var in self.data_vars.items():
            weights[name] = 1 if var.get() else 0
        data["data_weights"] = weights

        # Dream
        data["dream_enabled"] = self.dream_var.get()
        data["dream_mode"] = self.dream_mode_var.get()
        data["dream_frequency"] = self.dream_freq_var.get()
        data["dream_steps"] = self.dream_steps_var.get()
        data["dream_think_ticks"] = self.dream_tt_var.get()

        save_controls(data)
        self.status_var.set(f"Saved  {time.strftime('%H:%M:%S')}")

    def _schedule_refresh(self):
        self._refresh_from_file()
        self.root.after(2000, self._schedule_refresh)

    def _refresh_from_file(self):
        """Re-read controls.json to stay in sync (e.g. if training loop wrote back)."""
        self._building = True
        try:
            data = load_controls()
            if not data:
                return

            eff = data.get("effort", self.effort_var.get())
            if eff != self.effort_var.get():
                self.effort_var.set(eff)
                self._update_effort_label()

            lr = data.get("lr")
            if lr is not None:
                self.lr_var.set(str(lr))

            tt = data.get("think_ticks")
            if tt is not None:
                self.tt_var.set(tt)

            bs = data.get("batch_size")
            if bs is not None:
                self.batch_var.set(bs)

            lcx = data.get("use_lcx")
            if lcx is not None:
                self.lcx_var.set(lcx)

            ev = data.get("eval_every")
            if ev is not None:
                self.eval_var.set(ev)

            ck = data.get("checkpoint_every")
            if ck is not None:
                self.ckpt_var.set(ck)

            # Dream
            de = data.get("dream_enabled")
            if de is not None:
                self.dream_var.set(de)
            dm = data.get("dream_mode")
            if dm is not None:
                self.dream_mode_var.set(dm)
            df = data.get("dream_frequency")
            if df is not None:
                self.dream_freq_var.set(df)
            ds = data.get("dream_steps")
            if ds is not None:
                self.dream_steps_var.set(ds)
            dt = data.get("dream_think_ticks")
            if dt is not None:
                self.dream_tt_var.set(dt)

            # Data weights
            weights = data.get("data_weights", {})
            for name, var in self.data_vars.items():
                w = weights.get(name, 0)
                var.set(w > 0)

        except Exception:
            pass
        finally:
            self._building = False


if __name__ == "__main__":
    root = tk.Tk()
    root.iconname("VRAXION")
    app = ControlDesk(root)
    root.mainloop()
