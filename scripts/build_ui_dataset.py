"""Assemble train + val into a YOLO dataset for ui_yolo26m_v1 training.

Sources:
  TRAIN dirs (3, all .jpg + .txt symlinked in):
    - data/raw_images/run_20260521_103956_distinct  (~876 frames, includes
      oversample copies for minority classes)
    - data/raw_images/run_20260527_094158           (~149 craft补录)
    - data/raw_images/run_20260527_101545           (~15 craft补录)
  VAL dir:
    - data/raw_images/_ui_val_pool                  (~51 hand-curated frames,
      borrowed from fused_avatar manual val + old run sampling)

Output:
  D:/Project/ml_cache/models/yolo/dataset/ui_v1/
    images/train/<source>__<stem>.jpg   (symlinks)
    images/val/<stem>.jpg
    labels/train/<source>__<stem>.txt
    labels/val/<stem>.txt
    data.yaml                            (path / train / val / nc / names)

Class schema: pulled from run_20260521_103956_distinct/classes.txt (447 classes).
All 3 train dirs + val dir use the same schema — verified before build aborts
on mismatch.

Usage:
    py scripts/build_ui_dataset.py
    py scripts/build_ui_dataset.py --clean
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"
OUT_ROOT = Path("D:/Project/ml_cache/models/yolo/dataset/ui_v1")

TRAIN_SOURCES = [
    "run_20260521_103956_distinct",  # main batch (852 labeled w/ oversample)
    "run_20260527_094158",           # craft 补录 (149)
    "run_20260527_101545",           # craft 补录 (15)
    "run_20260518_002646",           # tiny补 — covers cls 97 活动剧情_已选择 (2)
    # NOTE: run_20260518_163513 / run_20260228_* / run_20260307_* are
    # fused_avatar labels (character heads), NOT UI — DO NOT include.
]
VAL_SOURCE = "_ui_val_pool"


def link_pair(src_jpg: Path, src_txt: Path, dst_jpg: Path, dst_txt: Path):
    """Symlink (or copy if FS rejects) a (jpg, txt) pair into dst paths."""
    for src, dst in [(src_jpg, dst_jpg), (src_txt, dst_txt)]:
        if dst.exists():
            dst.unlink()
        try:
            # Resolve to real file in case src is itself a symlink (oversample copies)
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="wipe output dir before building")
    args = ap.parse_args()

    # Schema source = first train dir
    schema_src = RAW / TRAIN_SOURCES[0]
    classes_file = schema_src / "classes.txt"
    if not classes_file.exists():
        print(f"[!] missing schema source: {classes_file}", file=sys.stderr)
        return 1
    classes = classes_file.read_text(encoding="utf-8").splitlines()
    print(f"[schema] {len(classes)} classes from {schema_src.name}/classes.txt")

    # Verify all sources use same schema
    for s in TRAIN_SOURCES[1:] + [VAL_SOURCE]:
        cf = RAW / s / "classes.txt"
        if cf.exists():
            other = cf.read_text(encoding="utf-8").splitlines()
            if other != classes:
                print(f"[!] schema mismatch in {s}/classes.txt ({len(other)} lines vs {len(classes)})",
                      file=sys.stderr)
                # Diff first 3
                for i in range(min(len(classes), len(other), 5)):
                    if classes[i] != other[i]:
                        print(f"  idx {i}: schema={classes[i]!r}  {s}={other[i]!r}")
                return 1

    if args.clean and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
        print(f"[clean] wiped {OUT_ROOT}")

    img_train = OUT_ROOT / "images" / "train"
    img_val = OUT_ROOT / "images" / "val"
    lbl_train = OUT_ROOT / "labels" / "train"
    lbl_val = OUT_ROOT / "labels" / "val"
    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    # Train: walk each source dir, pick every (jpg, txt) pair
    train_added = 0
    train_skipped = 0
    train_empty = 0
    train_negatives = 0  # frames with label file but no box (will be added as negatives)
    for s in TRAIN_SOURCES:
        sd = RAW / s
        if not sd.is_dir():
            print(f"[!] train source missing: {sd}")
            continue
        n_before = train_added
        for jpg in sorted(sd.glob("*.jpg")):
            txt = sd / (jpg.stem + ".txt")
            if not txt.exists():
                train_skipped += 1
                continue
            # Skip pseudo-files like _kept.jpg / classes.jpg (no such things but be safe)
            if jpg.stem.startswith("_") or jpg.stem == "classes":
                continue
            stem = f"{s}__{jpg.stem}"
            dst_jpg = img_train / f"{stem}.jpg"
            dst_txt = lbl_train / f"{stem}.txt"
            link_pair(jpg, txt, dst_jpg, dst_txt)
            # Tally empty vs non-empty
            lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                train_negatives += 1
            train_added += 1
        print(f"[train] {s}: +{train_added - n_before} frames")
    print(f"[train] total {train_added} frames added (skipped {train_skipped} unpaired, "
          f"{train_negatives} empty/negatives kept as background samples)")

    # Val: same logic for val pool
    val_added = 0
    val_skipped = 0
    val_empty = 0
    vd = RAW / VAL_SOURCE
    if not vd.is_dir():
        print(f"[!] val source missing: {vd}", file=sys.stderr)
        return 1
    for jpg in sorted(vd.glob("*.jpg")):
        txt = vd / (jpg.stem + ".txt")
        if not txt.exists():
            val_skipped += 1
            continue
        if jpg.stem.startswith("_") or jpg.stem == "classes":
            continue
        stem = jpg.stem
        dst_jpg = img_val / f"{stem}.jpg"
        dst_txt = lbl_val / f"{stem}.txt"
        link_pair(jpg, txt, dst_jpg, dst_txt)
        lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            val_empty += 1
        val_added += 1
    print(f"[val] total {val_added} frames added (skipped {val_skipped} unpaired, "
          f"{val_empty} empty)")

    # Write data.yaml
    yaml_path = OUT_ROOT / "data.yaml"
    yaml = []
    yaml.append(f"path: {OUT_ROOT.as_posix()}")
    yaml.append("train: images/train")
    yaml.append("val: images/val")
    yaml.append(f"nc: {len(classes)}")
    yaml.append("names:")
    for i, n in enumerate(classes):
        # Escape special chars by quoting; bash-friendly
        safe = n.replace("'", "\\'")
        yaml.append(f"  {i}: '{safe}'")
    yaml_path.write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"[yaml] wrote {yaml_path} ({len(classes)} classes)")

    print()
    print(f"[done] dataset ready at {OUT_ROOT}")
    print(f"       train: {train_added}, val: {val_added}, ratio {train_added/(train_added+val_added)*100:.1f}% / {val_added/(train_added+val_added)*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
