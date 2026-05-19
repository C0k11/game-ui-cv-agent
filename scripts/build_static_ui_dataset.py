"""Build a YOLO detection training dataset from all labeled BA UI captures.

Aggregates labeled (image, label) pairs from EVERY dataset under
data/raw_images/ (plus any trajectory dirs with labels) into a single
YOLO-format training tree.  Uses the universal master class registry
(data/raw_images/_classes.txt) as the canonical class index space.

Why one mega-dataset:
  Static UI overfits well — more diverse frames showing the SAME UI
  element in different contexts beats lots of duplicate frames.  Merging
  every labeled capture across runs maximizes that diversity for free.

Label cleaning:
  Existing label .txt files store extended fields beyond standard YOLO
  detection format (shape=polygon, angle, polygon points).  YOLO's
  detection trainer expects ONLY `cls cx cy w h` per line — so we strip
  the extras here when copying.  Original files are untouched.

Output layout (under ml_cache/models/yolo/dataset/static_ui_v1/):
    images/train/<ds>__<frame>.jpg
    images/val/<ds>__<frame>.jpg
    labels/train/<ds>__<frame>.txt
    labels/val/<ds>__<frame>.txt
    data.yaml
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"
TRAJ = REPO / "data" / "trajectories"
MASTER_FILE = RAW / "_classes.txt"
OUT_ROOT = Path(r"D:\Project\ml_cache\models\yolo\dataset\static_ui_v1")

VAL_RATIO = 0.20
SEED = 42


def collect_label_dirs(exclude_val_pool: bool = True) -> List[Tuple[Path, str]]:
    """Yield (label_dir, dataset_tag) for every dir potentially containing labels.

    By default, skips `_val_*` dirs (validation pools) — those are handled
    separately by collect_val_pool_dirs() so they don't bleed into train.
    """
    out: List[Tuple[Path, str]] = []
    if RAW.is_dir():
        for d in sorted(RAW.iterdir()):
            if not d.is_dir():
                continue
            if exclude_val_pool and d.name.startswith("_val_"):
                continue
            sub = d / "frames" if (d / "frames").is_dir() else d
            out.append((sub, d.name))
    if TRAJ.is_dir():
        for d in sorted(TRAJ.iterdir()):
            if not d.is_dir() or not d.name.startswith("run_"):
                continue
            # Only include trajectory dirs that have any non-classes label .txt
            if any(f for f in d.glob("*.txt") if f.name != "classes.txt"):
                out.append((d, f"traj_{d.name}"))


def collect_val_pool_dirs() -> List[Tuple[Path, str]]:
    """Return label dirs from data/raw_images/_val_static_ui/frames/.

    These are held-out frames the user labeled specifically for static_ui
    validation — they go 100% to val, never to train.  Returns empty list
    if no such pool exists.
    """
    out: List[Tuple[Path, str]] = []
    pool = RAW / "_val_static_ui" / "frames"
    if pool.is_dir():
        if any(f for f in pool.glob("*.txt") if f.name != "classes.txt"):
            out.append((pool, "_val_static_ui"))
    return out
    return out


def clean_label_line(line: str) -> str | None:
    """Strip a label line to standard YOLO detection format.

    Returns 'cls cx cy w h' or None to skip the line.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    if not parts[0].lstrip("-").isdigit():
        return None
    try:
        cls = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except (ValueError, IndexError):
        return None
    # Sanity: skip degenerate boxes (<1px equivalent)
    if w <= 0 or h <= 0:
        return None
    # YOLO format requires 0<=coord<=1, clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def collect_pairs(label_dirs: List[Tuple[Path, str]]) -> List[Tuple[Path, Path, str]]:
    """Return list of (image_path, label_path, dataset_tag) for labeled frames only."""
    pairs: List[Tuple[Path, Path, str]] = []
    for lbl_dir, tag in label_dirs:
        for lf in sorted(lbl_dir.glob("*.txt")):
            if lf.name == "classes.txt":
                continue
            try:
                txt = lf.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                # Some legacy label files saved with utf-16 BOM; try fallback.
                try:
                    txt = lf.read_text(encoding="utf-16").strip()
                except Exception:
                    print(f"[warn] skipping unreadable label file: {lf}")
                    continue
            if not txt:
                continue
            # Find matching image
            stem = lf.stem
            for ext in (".jpg", ".jpeg", ".png"):
                im = lf.with_name(stem + ext)
                if im.exists():
                    pairs.append((im, lf, tag))
                    break
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_ROOT),
                    help="Output YOLO dataset root.")
    ap.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict to specific dataset names (under data/raw_images/ or "
             "data/trajectories/).  Pass repeatedly for multiple.  Without "
             "this flag, ALL labeled datasets are aggregated.",
    )
    args = ap.parse_args()

    out_root = Path(args.out)
    master = [c.strip() for c in MASTER_FILE.read_text(encoding="utf-8").splitlines() if c.strip()]
    if not master:
        print(f"[err] master classes missing: {MASTER_FILE}")
        return 1
    print(f"[master] {len(master)} classes")

    all_dirs = collect_label_dirs(exclude_val_pool=True)
    if args.only:
        only_set = set(args.only)
        filtered = [(d, tag) for d, tag in all_dirs if d.parent.name in only_set or d.name in only_set]
        if not filtered:
            print(f"[err] --only {args.only} matched 0 dirs (available: {[d.parent.name if d.name=='frames' else d.name for d, _ in all_dirs][:10]}...)")
            return 1
        print(f"[only] filtered {len(all_dirs)} -> {len(filtered)} label dirs")
        all_dirs = filtered
    pairs = collect_pairs(all_dirs)
    print(f"[scan] {len(pairs)} labeled (image, label) pairs across "
          f"{len(set(p[2] for p in pairs))} source datasets (train pool)")
    if not pairs:
        print("[err] no labeled data found")
        return 1

    # Dedicated val pool detection — if user labeled _val_static_ui/frames/,
    # use 100% of pairs for train and val_pool for val.
    val_pool_dirs = collect_val_pool_dirs()
    val_pool_pairs = collect_pairs(val_pool_dirs) if val_pool_dirs else []

    rng = random.Random(args.seed)
    if val_pool_pairs:
        train_pairs = pairs
        val_pairs = val_pool_pairs
        print(f"[split] dedicated val: train {len(train_pairs)} (100% of recordings) "
              f"/ val {len(val_pairs)} (held-out _val_static_ui pool)")
    else:
        # Fallback: class-stratified split (rare classes pinned to train)
        class_to_frames: dict = {}
        for i, (im, lf, _) in enumerate(pairs):
            try:
                raw_lines = lf.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                try:
                    raw_lines = lf.read_text(encoding="utf-16").splitlines()
                except Exception:
                    raw_lines = []
            seen = set()
            for line in raw_lines:
                parts = line.strip().split()
                if parts and parts[0].lstrip("-").isdigit():
                    seen.add(int(parts[0]))
            for c in seen:
                class_to_frames.setdefault(c, []).append(i)

        must_train: set = set()
        for c, frame_idxs in sorted(class_to_frames.items(), key=lambda kv: len(kv[1])):
            if len(frame_idxs) <= 3:
                must_train.update(frame_idxs)
            else:
                must_train.update(frame_idxs[:2])

        remaining = [i for i in range(len(pairs)) if i not in must_train]
        rng.shuffle(remaining)
        n_val_target = max(1, int(len(pairs) * args.val_ratio))
        n_val_actual = min(n_val_target, len(remaining))
        val_idx = set(remaining[:n_val_actual])
        train_idx = set(range(len(pairs))) - val_idx

        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]
        print(f"[split] stratified fallback: train {len(train_pairs)} / val {len(val_pairs)} "
              f"({len(class_to_frames)} classes; sparse classes pinned to train)")

    if args.dry_run:
        print("[dry-run] would write to:", out_root)
        return 0

    # Clean output
    if out_root.exists():
        shutil.rmtree(out_root)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    def emit(pair_list: List[Tuple[Path, Path, str]], split: str) -> None:
        n_boxes = 0
        for im, lf, tag in pair_list:
            stem = f"{tag}__{im.stem}"
            shutil.copy2(im, out_root / "images" / split / (stem + im.suffix.lower()))
            # Clean label (with same UTF-8/UTF-16 fallback as collect_pairs)
            try:
                raw_lines = lf.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                try:
                    raw_lines = lf.read_text(encoding="utf-16").splitlines()
                except Exception:
                    raw_lines = []
            lines = []
            for line in raw_lines:
                cleaned = clean_label_line(line)
                if cleaned:
                    lines.append(cleaned)
            (out_root / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            n_boxes += len(lines)
        print(f"[emit] {split}: {len(pair_list)} files, {n_boxes} boxes")

    emit(train_pairs, "train")
    emit(val_pairs, "val")

    # data.yaml — pathstyle that ultralytics expects
    yaml_lines = [
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(master)}",
        "names:",
    ]
    for i, name in enumerate(master):
        # Escape names with special chars
        safe = name.replace("'", "\\'")
        yaml_lines.append(f"  {i}: '{safe}'")
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"[yaml] {out_root / 'data.yaml'}")

    print()
    print("Next: py scripts/train_yolo26.py static_ui")
    return 0


if __name__ == "__main__":
    sys.exit(main())
