"""Sync a canonical classes.txt across multiple dataset directories.

Common scenario: you edit classes in dashboard's main dataset (adds a new
class entry in canonical/classes.txt) but other train/val dirs still have
the old schema → build_*_dataset.py aborts on schema mismatch.

This script copies the canonical classes.txt to every target dir, ensuring
all dirs share the same nc + names ordering. Label files themselves are
untouched — only the schema file is normalized.

Usage:
    py scripts/sync_classes.py \\
        --canonical data/raw_images/run_20260521_103956_distinct/classes.txt \\
        --to data/raw_images/run_20260527_094158 \\
             data/raw_images/run_20260527_101545 \\
             data/raw_images/_ui_val_pool

    # Default behavior: sync UI dataset dirs (those used by build_ui_dataset.py)
    py scripts/sync_classes.py
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw_images"

DEFAULT_CANONICAL = RAW / "run_20260521_103956_distinct" / "classes.txt"
DEFAULT_TARGETS = [
    RAW / "run_20260527_094158",
    RAW / "run_20260527_101545",
    RAW / "run_20260518_002646",
    RAW / "_ui_val_pool",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL,
                    help="source classes.txt (canonical schema)")
    ap.add_argument("--to", nargs="+", type=Path, default=None,
                    help="target dirs that should receive the canonical "
                         "classes.txt (defaults to the UI dataset dirs).")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would change without writing")
    args = ap.parse_args()

    src = args.canonical
    if not src.exists():
        print(f"[!] canonical not found: {src}", file=sys.stderr)
        return 1
    canon_lines = src.read_text(encoding="utf-8").splitlines()
    print(f"[canonical] {src} ({len(canon_lines)} classes)")

    targets = args.to if args.to else DEFAULT_TARGETS
    changed = 0
    same = 0
    missing = 0
    for tgt_dir in targets:
        if not tgt_dir.is_dir():
            print(f"[!] target dir missing: {tgt_dir}")
            missing += 1
            continue
        tgt_file = tgt_dir / "classes.txt"
        if tgt_file.exists():
            tgt_lines = tgt_file.read_text(encoding="utf-8").splitlines()
            if tgt_lines == canon_lines:
                print(f"  ✓ {tgt_dir.name}: already in sync")
                same += 1
                continue
            print(f"  ⚙ {tgt_dir.name}: needs sync ({len(tgt_lines)} → {len(canon_lines)} lines)")
        else:
            print(f"  ⚙ {tgt_dir.name}: no classes.txt — will create")
        if not args.dry_run:
            shutil.copy2(src, tgt_file)
        changed += 1

    print()
    print(f"[done] {changed} synced, {same} already aligned, {missing} dirs missing"
          f"{' (DRY RUN)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
