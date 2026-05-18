"""Trim the universal master class registry to a leaner subset.

Use case (2026-05-18): user accumulated 193 classes across early
experiments + today's deliberate UI labeling.  Wants to drop legacy
classes that don't appear in today's labels, keeping only what's
actually being used going forward.

Strategy:
  1. Seed set = union of classes referenced in label files of the
     given seed dataset(s).  Preserves first-seen order so indices
     stay readable.
  2. Build new master = seed set.  Add any extra `--keep` names too
     (for classes you plan to label soon but haven't yet).
  3. For every dataset (raw_images + trajectories):
     - Rewrite every label .txt: remap cls indices old→new; DROP any
       label line referencing a class not in new master.
     - If a label file becomes empty, delete it (YOLO treats missing
       .txt as zero boxes, same as empty file but cleaner).
     - Overwrite local classes.txt with new master copy.
  4. Save new master.

Safe to re-run: idempotent if seed set already matches master.

Future-compat: the universal-master add_class API still works after
trimming — appending a brand-new class just grows the (smaller) master.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"
TRAJ = REPO / "data" / "trajectories"
MASTER_FILE = RAW / "_classes.txt"
BACKUP_ROOT = RAW / "_backups"


def backup_all_labels() -> Path:
    """Snapshot every .txt file under raw_images + trajectories before mutating.

    Returns the timestamped backup dir for log/printing.  Cheap operation —
    labels are small text files.  Rollback = `cp -r <backup>/. data/`.
    """
    import shutil
    import time
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = BACKUP_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_root, prefix in ((RAW, "raw"), (TRAJ, "traj")):
        if not src_root.is_dir():
            continue
        for d in src_root.iterdir():
            if not d.is_dir():
                continue
            sub = d / "frames" if (d / "frames").is_dir() else d
            txt_files = list(sub.glob("*.txt"))
            if not txt_files:
                continue
            dst_dir = out / prefix / d.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in txt_files:
                shutil.copy2(f, dst_dir / f.name)
                n += 1
    # Also back up the master itself
    if MASTER_FILE.exists():
        shutil.copy2(MASTER_FILE, out / "_classes.txt")
    print(f"[backup] {n} label files snapshotted to {out}")
    return out


def load_classes(p: Path) -> List[str]:
    if not p.exists():
        return []
    return [c.strip() for c in p.read_text(encoding="utf-8").splitlines() if c.strip()]


def save_classes(p: Path, names: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(names) + "\n", encoding="utf-8")


def label_dirs() -> List[Path]:
    """Yield every directory that may contain label .txt files."""
    out: List[Path] = []
    if RAW.is_dir():
        for d in sorted(RAW.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith("_"):  # skip _backups, _classes.txt parent, etc.
                continue
            sub = d / "frames" if (d / "frames").is_dir() else d
            out.append(sub)
    if TRAJ.is_dir():
        for d in sorted(TRAJ.iterdir()):
            if not d.is_dir() or not d.name.startswith("run_"):
                continue
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed",
        action="append",
        required=True,
        help="Seed dataset name(s) under data/raw_images/. Use --seed run_X --seed run_Y for multiple.",
    )
    ap.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Extra class names to keep even if unused in seed datasets.",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Explicit drop list — class names to remove even if present in seed "
             "(use for misclick/duplicate classes the seed dataset accidentally has).",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Show plan, don't actually modify files.")
    args = ap.parse_args()

    old_master = load_classes(MASTER_FILE)
    if not old_master:
        print(f"[err] no master at {MASTER_FILE} — bootstrap first by accessing /api/v1/datasets/images")
        return 1
    print(f"[old] master has {len(old_master)} classes")

    # ── 1. Build new master from seed datasets ──
    new_master: List[str] = []
    seen: Set[str] = set()
    for seed_name in args.seed:
        seed_dir = (RAW / seed_name)
        if not seed_dir.is_dir():
            print(f"[err] seed not found: {seed_dir}")
            return 1
        seed_sub = seed_dir / "frames" if (seed_dir / "frames").is_dir() else seed_dir
        for lf in seed_sub.glob("*.txt"):
            if lf.name == "classes.txt":
                continue
            txt = lf.read_text(encoding="utf-8").strip()
            if not txt:
                continue
            for line in txt.splitlines():
                parts = line.strip().split()
                if not parts or not parts[0].lstrip("-").isdigit():
                    continue
                ci = int(parts[0])
                if 0 <= ci < len(old_master):
                    name = old_master[ci]
                    if name not in seen:
                        seen.add(name)
                        new_master.append(name)
        print(f"[seed] {seed_name}: cumulative {len(new_master)} unique classes")

    # extra --keep names (e.g. for things user wants but hasn't labeled yet)
    for k in args.keep:
        if k not in seen:
            seen.add(k)
            new_master.append(k)
            print(f"[keep] adding extra class: {k}")

    # explicit --exclude (e.g. misclick variants user wants gone)
    excludes = set(args.exclude or [])
    if excludes:
        before = len(new_master)
        new_master = [c for c in new_master if c not in excludes]
        seen = set(new_master)
        for x in args.exclude:
            print(f"[exclude] dropping: {x}{'' if x in (set(load_classes(MASTER_FILE)) | set([c for c in old_master if c in excludes])) else ' (not in master, ignored)'}")
        print(f"[exclude] new master {len(new_master)} (was {before} after seed)")

    print(f"[new] master will have {len(new_master)} classes "
          f"(dropping {len(old_master) - len(seen & set(old_master))} legacy classes)")
    dropped = [c for c in old_master if c not in seen]
    print(f"[drop] first 10 to be removed: {dropped[:10]}")
    print(f"[drop] last 10 to be removed: {dropped[-10:]}")

    # ── 2. Build remap: old_idx -> new_idx (or None if class is dropped) ──
    remap: Dict[int, int] = {}
    for old_idx, name in enumerate(old_master):
        if name in seen:
            remap[old_idx] = new_master.index(name)

    if args.dry_run:
        print()
        print("[dry-run] would rewrite labels in these dirs:")
        for d in label_dirs():
            n = len([f for f in d.glob("*.txt") if f.name != "classes.txt"])
            print(f"  {d}  ({n} label files)")
        print()
        print(f"[dry-run] would save {len(new_master)} classes to master")
        return 0

    # ── Safety: snapshot every label file before mutating anything ──
    backup_dir = backup_all_labels()
    print(f"[safety] rollback command if anything looks wrong:")
    print(f"         (manually restore .txt files from {backup_dir})")
    print()

    # ── 3. Migrate every dataset ──
    total_files_touched = 0
    total_lines_dropped = 0
    total_files_emptied = 0
    for dirp in label_dirs():
        for lf in dirp.glob("*.txt"):
            if lf.name == "classes.txt":
                continue
            try:
                txt = lf.read_text(encoding="utf-8")
            except Exception:
                continue
            if not txt.strip():
                continue
            new_lines: List[str] = []
            file_dropped = 0
            for line in txt.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                if not parts[0].lstrip("-").isdigit():
                    continue
                old_idx = int(parts[0])
                new_idx = remap.get(old_idx)
                if new_idx is None:
                    file_dropped += 1
                    continue
                parts[0] = str(new_idx)
                new_lines.append(" ".join(parts))
            if new_lines:
                lf.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            else:
                # Empty after dropping — remove the file entirely
                lf.unlink()
                total_files_emptied += 1
            if file_dropped:
                total_lines_dropped += file_dropped
                total_files_touched += 1

    print()
    print(f"[migrate] dropped {total_lines_dropped} label lines across "
          f"{total_files_touched} files")
    print(f"[migrate] removed {total_files_emptied} now-empty label files")

    # ── 4. Save new master ──
    save_classes(MASTER_FILE, new_master)
    print(f"[save] master → {MASTER_FILE} ({len(new_master)} classes)")

    # ── 5. Sync every dataset's local classes.txt to the new master ──
    synced = 0
    for dirp in label_dirs():
        cf = dirp / "classes.txt"
        if cf.exists() or any(dirp.glob("*.txt")):
            # only sync dirs that have labels (skip empty trajectory dirs)
            cf.write_text("\n".join(new_master) + "\n", encoding="utf-8")
            synced += 1
    print(f"[sync] updated classes.txt in {synced} dataset dirs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
