from __future__ import annotations

import random

import numpy as np


def _drift_loss(net, rng=random):
    if rng.randint(1, 5) == 1:
        net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + rng.randint(-3, 3))))


def _drift_intensity(value: int, rng=random, min_value: int = 1, max_value: int = 15) -> int:
    if rng.randint(1, 20) <= 7:
        value = max(min_value, min(max_value, value + rng.choice([-1, 1])))
    return int(value)


def _apply_ops(net, ops):
    undo = []
    for op in ops:
        if op == "add":
            net._add(undo)
        elif op == "remove":
            net._remove(undo)
        elif op == "rewire":
            net._rewire(undo)
        elif op == "flip":
            net._flip(undo)
        else:
            raise ValueError(f"Unsupported forced op: {op}")
    net._sync_sparse_idx()
    return undo


class BasePolicyAdapter:
    name = "policy"

    def __init__(self):
        self._state = {}

    def propose(self, net):
        raise NotImplementedError

    def on_accept(self, proposal):
        return None

    def on_reject(self, proposal):
        return None

    def after_step(self, improved: bool, attempt: int):
        return None

    def describe_state(self):
        return dict(self._state)


class DrivePolicyAdapter(BasePolicyAdapter):
    name = "drive"

    def propose(self, net):
        net.mutate()
        return None


class ModePolicyAdapter(BasePolicyAdapter):
    name = "mode"
    OPS = ["flip", "add", "remove", "rewire"]

    def __init__(self):
        super().__init__()
        self._state.update({"mode": 1, "intensity": 3})

    def propose(self, net):
        previous = dict(self._state)
        self._state["intensity"] = _drift_intensity(self._state["intensity"])
        _drift_loss(net)
        if random.randint(1, 20) <= 7:
            self._state["mode"] = (self._state["mode"] + random.choice([-1, 1])) % len(self.OPS)
        ops = [self.OPS[self._state["mode"]]] * int(self._state["intensity"])
        _apply_ops(net, ops)
        return {"previous": previous}

    def on_reject(self, proposal):
        self._state = proposal["previous"]

    def describe_state(self):
        return {
            "mode": self.OPS[self._state["mode"]],
            "intensity": int(self._state["intensity"]),
        }


class BoolMoodPolicyAdapter(BasePolicyAdapter):
    name = "bool_mood"

    def __init__(self):
        super().__init__()
        self._state.update({"grow": True, "refine": False, "aggressive": True})

    def propose(self, net):
        previous = dict(self._state)
        for key in ["grow", "refine", "aggressive"]:
            if random.random() < 0.35:
                self._state[key] = not self._state[key]
        _drift_loss(net)
        n_changes = 10 if self._state["aggressive"] else 3
        ops = []
        for _ in range(n_changes):
            if self._state["grow"] and not self._state["refine"]:
                ops.append("add" if random.random() < 0.7 else "flip")
            elif self._state["grow"] and self._state["refine"]:
                roll = random.random()
                if roll < 0.6:
                    ops.append("rewire")
                elif roll < 0.8:
                    ops.append("flip")
                else:
                    ops.append("add")
            elif (not self._state["grow"]) and self._state["refine"]:
                ops.append("flip" if random.random() < 0.8 else "rewire")
            else:
                roll = random.random()
                if roll < 0.7:
                    ops.append("remove")
                elif roll < 0.9:
                    ops.append("flip")
                else:
                    ops.append("rewire")
        _apply_ops(net, ops)
        return {"previous": previous}

    def on_reject(self, proposal):
        self._state = proposal["previous"]

    def describe_state(self):
        return {
            "grow": bool(self._state["grow"]),
            "refine": bool(self._state["refine"]),
            "aggressive": bool(self._state["aggressive"]),
        }


class AddRemovePolicyAdapter(BasePolicyAdapter):
    name = "add_remove"

    def __init__(self):
        super().__init__()
        self._state.update({"add_weight": 50, "remove_weight": 50, "intensity": 3})

    def propose(self, net):
        previous = dict(self._state)
        self._state["intensity"] = _drift_intensity(self._state["intensity"])
        _drift_loss(net)
        if random.randint(1, 20) <= 7:
            self._state["add_weight"] = max(
                1, min(99, self._state["add_weight"] + random.choice([-3, -1, 1, 3]))
            )
        if random.randint(1, 20) <= 7:
            self._state["remove_weight"] = max(
                1, min(99, self._state["remove_weight"] + random.choice([-3, -1, 1, 3]))
            )

        total = self._state["add_weight"] + self._state["remove_weight"]
        ops = []
        for _ in range(int(self._state["intensity"])):
            roll = random.randint(1, total)
            ops.append("add" if roll <= self._state["add_weight"] else "remove")
        _apply_ops(net, ops)
        return {"previous": previous}

    def on_reject(self, proposal):
        self._state = proposal["previous"]

    def describe_state(self):
        return {
            "add_weight": int(self._state["add_weight"]),
            "remove_weight": int(self._state["remove_weight"]),
            "intensity": int(self._state["intensity"]),
        }


class LegacyFlipOnRejectStrategyAdapter(BasePolicyAdapter):
    name = "flip_on_reject"

    def __init__(self):
        super().__init__()
        self._state.update({"signal": True, "grow": False, "intensity": 3})

    def _mode_ops(self):
        if self._state["signal"]:
            return ["flip"] * int(self._state["intensity"])
        if self._state["grow"]:
            return ["add"] * int(self._state["intensity"])
        return ["remove"] * int(self._state["intensity"])

    def propose(self, net):
        self._state["intensity"] = _drift_intensity(self._state["intensity"])
        _drift_loss(net)
        _apply_ops(net, self._mode_ops())
        return None

    def on_reject(self, proposal):
        if random.random() < 0.35:
            self._state["signal"] = not self._state["signal"]
        if random.random() < 0.35:
            self._state["grow"] = not self._state["grow"]

    def describe_state(self):
        mode = "signal" if self._state["signal"] else ("grow" if self._state["grow"] else "shrink")
        return {"mode": mode, "intensity": int(self._state["intensity"])}


class DarwinianStrategyAdapter(LegacyFlipOnRejectStrategyAdapter):
    name = "darwinian"

    def propose(self, net):
        if random.random() < 0.35:
            self._state["signal"] = not self._state["signal"]
        if random.random() < 0.35:
            self._state["grow"] = not self._state["grow"]
        return super().propose(net)

    def on_reject(self, proposal):
        return None


class WindowReviewStrategyAdapter(LegacyFlipOnRejectStrategyAdapter):
    name = "window_review"

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = int(window_size)
        self._accepts_in_window = 0
        self._changes = 0

    def after_step(self, improved: bool, attempt: int):
        if improved:
            self._accepts_in_window += 1
        if attempt % self.window_size != 0:
            return
        accept_rate = self._accepts_in_window / self.window_size
        if accept_rate < 0.10:
            self._state["signal"] = not self._state["signal"]
            self._state["grow"] = not self._state["grow"]
            self._changes += 1
        elif accept_rate < 0.15:
            if random.random() < 0.5:
                self._state["signal"] = not self._state["signal"]
            else:
                self._state["grow"] = not self._state["grow"]
            self._changes += 1
        self._accepts_in_window = 0

    def on_reject(self, proposal):
        return None

    def describe_state(self):
        state = super().describe_state()
        state.update({
            "window_size": self.window_size,
            "strategy_changes": self._changes,
        })
        return state


def build_policy(name: str):
    if name == "drive":
        return DrivePolicyAdapter()
    if name == "mode":
        return ModePolicyAdapter()
    if name == "bool_mood":
        return BoolMoodPolicyAdapter()
    if name == "add_remove":
        return AddRemovePolicyAdapter()
    if name == "flip_on_reject":
        return LegacyFlipOnRejectStrategyAdapter()
    if name == "darwinian":
        return DarwinianStrategyAdapter()
    if name.startswith("window_"):
        return WindowReviewStrategyAdapter(int(name.split("_", 1)[1]))
    raise ValueError(f"Unknown mutation policy: {name}")
