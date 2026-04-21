"""Pull Block C sweep artifacts from Modal Volume to local disk.

Modal CLI `modal volume get` on directories is broken on this version
(Errno 17 / 21 regardless of target existence). Workaround: use the
Python SDK to read each file individually.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal


def pull_file(vol: modal.Volume, remote: str, local: Path) -> int:
    local.parent.mkdir(parents=True, exist_ok=True)
    with local.open("wb") as f:
        n = 0
        for chunk in vol.read_file(remote):
            f.write(chunk)
            n += len(chunk)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="vraxion-block-c")
    ap.add_argument("--remote", required=True,
                    help="Remote directory root on the volume, e.g. "
                         "runs/prod_100mb_8ep_v2")
    ap.add_argument("--local", required=True, type=Path,
                    help="Local destination directory")
    ap.add_argument("--only", default="",
                    help="Comma-separated substrings to restrict pulled "
                         "filenames (e.g. 'emb_E,training_summary.json')")
    args = ap.parse_args()

    vol = modal.Volume.from_name(args.volume)
    filters = [s for s in args.only.split(",") if s]

    def walk(remote: str):
        for e in vol.listdir(remote):
            if e.type.name == "DIRECTORY":
                yield from walk(e.path)
            else:
                yield e

    total_bytes = 0
    total_files = 0
    for f in walk(args.remote):
        if filters and not any(sub in f.path for sub in filters):
            continue
        rel = Path(f.path).relative_to(args.remote)
        local_path = args.local / rel
        n = pull_file(vol, f.path, local_path)
        total_bytes += n
        total_files += 1
        print(f"  {f.path}  -> {local_path}  ({n/1e6:.2f} MB)")

    print(f"\nPulled {total_files} files, {total_bytes/1e6:.1f} MB total")


if __name__ == "__main__":
    main()
