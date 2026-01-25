#!/usr/bin/env python
"""Offline mitosis: clone an expert and redirect addresses in the router map."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch


def _parse_addresses(arg: str) -> list[int]:
    addresses = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        addresses.append(int(token))
    if not addresses:
        raise ValueError("No target addresses provided.")
    return addresses


def _infer_num_experts(state: dict) -> int:
    max_idx = -1
    prefix = "head.experts."
    for key in state.keys():
        if key.startswith(prefix):
            try:
                idx = int(key[len(prefix):].split(".", 1)[0])
            except ValueError:
                continue
            max_idx = max(max_idx, idx)
    return max_idx + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="VRAXION mitosis: split an expert offline.")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output checkpoint (.pt)")
    parser.add_argument("--parent", type=int, help="Overloaded expert id to clone")
    parser.add_argument(
        "--addresses",
        help="Comma-separated ring addresses to redirect to the new expert",
    )
    parser.add_argument("--meta", help="Path to mitosis_meta.json for auto addresses/parent")
    parser.add_argument("--noise", type=float, default=1e-5, help="Noise scale for cloned weights")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    out_path = os.path.abspath(args.output)
    meta_parent = None
    meta_addrs = None
    if args.meta:
        meta_path = os.path.abspath(args.meta)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta_parent = meta.get("parent_expert")
        meta_addrs = meta.get("hot_addresses")
    parent_id = args.parent if args.parent is not None else meta_parent
    addresses_arg = args.addresses if args.addresses is not None else None
    if addresses_arg is None and meta_addrs is not None:
        addresses_arg = ",".join(str(a) for a in meta_addrs)
    if parent_id is None or addresses_arg is None:
        raise ValueError("Parent expert and addresses are required (via --parent/--addresses or --meta).")
    target_addrs = _parse_addresses(addresses_arg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError("Checkpoint missing 'model' state dict")
    state = ckpt["model"]

    num_experts = ckpt.get("num_experts")
    if num_experts is None:
        num_experts = _infer_num_experts(state)
    if num_experts <= 0:
        raise ValueError("Could not infer num_experts from checkpoint")
    new_id = int(num_experts)

    parent_prefix = f"head.experts.{parent_id}."
    child_prefix = f"head.experts.{new_id}."
    cloned = 0
    param_clones = []
    for key, value in list(state.items()):
        if key.startswith(parent_prefix):
            new_key = child_prefix + key[len(parent_prefix):]
            noise = torch.randn_like(value) * float(args.noise)
            state[new_key] = value.clone() + noise
            param_clones.append((key, new_key))
            cloned += 1
    if cloned == 0:
        raise ValueError(f"No expert parameters found for parent {parent_id}")

    if "router_map" not in state:
        raise KeyError("Checkpoint missing router_map; run a router-map build first.")
    router_map = state["router_map"].clone()
    max_addr = int(router_map.numel()) - 1
    for addr in target_addrs:
        if addr < 0 or addr > max_addr:
            raise ValueError(f"Address {addr} out of range [0, {max_addr}]")
        router_map[addr] = new_id
    state["router_map"] = router_map

    ckpt["model"] = state
    ckpt["num_experts"] = new_id + 1
    ckpt["mitosis"] = {
        "parent": int(parent_id),
        "child": int(new_id),
        "addresses": target_addrs,
        "noise": float(args.noise),
    }
    # Clone optimizer momentum if available and param names are present.
    param_names = ckpt.get("param_names")
    optim = ckpt.get("optim")
    if optim and param_names:
        param_groups = optim.get("param_groups", [])
        if param_groups:
            group = param_groups[0]
            params_list = list(group.get("params", []))
            name_to_pid = {name: pid for name, pid in zip(param_names, params_list)}
            next_pid = (max(params_list) + 1) if params_list else 0
            state_dict = optim.get("state", {})
            for parent_name, child_name in param_clones:
                parent_pid = name_to_pid.get(parent_name)
                if parent_pid is None:
                    continue
                child_pid = next_pid
                next_pid += 1
                params_list.append(child_pid)
                state_dict[child_pid] = copy.deepcopy(state_dict.get(parent_pid, {}))
                param_names.append(child_name)
            group["params"] = params_list
            param_groups[0] = group
            optim["param_groups"] = param_groups
            optim["state"] = state_dict
            ckpt["optim"] = optim
            ckpt["param_names"] = param_names

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(ckpt, out_path)
    print(
        f"[mitosis] cloned expert {parent_id} -> {new_id} "
        f"({cloned} tensors), redirected {len(target_addrs)} addresses -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
