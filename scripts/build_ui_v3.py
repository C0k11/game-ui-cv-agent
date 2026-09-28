"""Build the ui_v3 dataset (from-scratch UI detector) from human-verified labels only.

Sources (all labels use the master class list data/raw_images/_classes_next.txt):
  train  data/gold_v3/<page>/            live captures, only stems listed in _verified.txt
         data/ui_v3_old/<pool>/          reviewed old material: <stem>.txt + _images.txt (absolute image paths)
  val    data/gold_v3_val/<page>/        another day / another account, only stems in _verified.txt

The old material is imported once from the review working copy with --import-clean
(scratchpad/_clean_v3 and scratchpad/_thinmine_clean); big pools are never read directly.

Output: D:/Project/ml_cache/models/yolo/dataset/ui_v3/{images,labels}/{train,val} + data.yaml.
Images are hardlinked when possible (same volume), copied otherwise. A dhash leak check drops
val frames that are near-duplicates of any train frame and reports them.

Usage:
  py -X utf8 scripts/build_ui_v3.py --import-clean          # refresh data/ui_v3_old from the working copies
  py -X utf8 scripts/build_ui_v3.py [--clean] [--hd-leak 4]  # assemble the dataset
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "raw_images" / "_classes_next.txt"
GOLD = REPO / "data" / "gold_v3"
GOLD_VAL = REPO / "data" / "gold_v3_val"
OLD = REPO / "data" / "ui_v3_old"
CLEAN_SOURCES = [REPO / "scratchpad" / "_clean_v3", REPO / "scratchpad" / "_thinmine_clean"]
OUT_ROOT = Path("D:/Project/ml_cache/models/yolo/dataset/ui_v3")
AVATAR_SPAN = (143, 394)   # head-avatar ids live in the fused_avatar model, never in the UI set


def read_names() -> list[str]:
    return MASTER.read_text(encoding="utf-8").splitlines()


def verified_stems(page_dir: Path) -> list[str]:
    vf = page_dir / "_verified.txt"
    if not vf.is_file():
        return []
    return [s for s in vf.read_text(encoding="utf-8").split() if (page_dir / (s + ".txt")).is_file()]


def find_image(folder: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".png", ".jpeg"):
        p = folder / (stem + ext)
        if p.is_file():
            return p
    return None


def gold_items(root: Path, split: str) -> list[dict]:
    items = []
    if not root.is_dir():
        return items
    for page_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for stem in verified_stems(page_dir):
            img = find_image(page_dir, stem)
            if img is None:
                print(f"  [warn] no image for {page_dir.name}/{stem}")
                continue
            items.append({"src": page_dir.name, "stem": stem, "label": page_dir / (stem + ".txt"), "img": img, "split": split})
    return items


def old_items() -> list[dict]:
    items = []
    if not OLD.is_dir():
        return items
    for pool in sorted(p for p in OLD.iterdir() if p.is_dir() and not p.name.startswith("_")):
        lst = pool / "_images.txt"
        imgs = {}
        if lst.is_file():
            for line in lst.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    imgs[Path(line).stem] = Path(line)
        for lab in sorted(pool.glob("*.txt")):
            if lab.name.startswith("_") or lab.name == "classes.txt":
                continue
            img = imgs.get(lab.stem) or find_image(pool, lab.stem)
            if img is None or not img.is_file():
                print(f"  [warn] missing image for {pool.name}/{lab.stem}")
                continue
            items.append({"src": pool.name, "stem": lab.stem, "label": lab, "img": img, "split": "train"})
    return items


def import_clean() -> None:
    """Copy the reviewed working copies into data/ui_v3_old (labels + image lists only).

    Pools that exist in several sources (a _clean_v3 pool that also got thin-mine frames) are merged:
    the destination is rebuilt from scratch, then every source adds its label files and image paths.
    """
    if OLD.exists():
        shutil.rmtree(OLD)
    OLD.mkdir(parents=True)
    n_lab = 0
    pools = set()
    for src in CLEAN_SOURCES:
        if not src.is_dir():
            continue
        for pool in sorted(p for p in src.iterdir() if p.is_dir() and not p.name.startswith("_apply") and p.name not in ("_review", "_fit")):
            labs = [p for p in pool.glob("*.txt") if not p.name.startswith("_") and p.name != "classes.txt"]
            if not labs:
                continue
            dst = OLD / pool.name
            dst.mkdir(exist_ok=True)
            for lab in labs:
                shutil.copyfile(lab, dst / lab.name)
            if (pool / "classes.txt").is_file() and not (dst / "classes.txt").is_file():
                shutil.copyfile(pool / "classes.txt", dst / "classes.txt")
            lst = pool / "_images.txt"
            if lst.is_file():
                have = set()
                dl = dst / "_images.txt"
                if dl.is_file():
                    have = {ln.strip() for ln in dl.read_text(encoding="utf-8").splitlines() if ln.strip()}
                add = [ln.strip() for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip() and ln.strip() not in have]
                with open(dl, "a", encoding="utf-8", newline="\n") as fo:
                    for ln in add:
                        fo.write(ln + "\n")
            pools.add(pool.name)
            n_lab += len(labs)
    print(f"imported {len(pools)} pools / {n_lab} label files into {OLD}")


def parse_label(path: Path, nc: int) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    rows, problems = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        q = ln.split()
        if len(q) < 5:
            if ln.strip():
                problems.append("short line")
            continue
        try:
            k = int(float(q[0]))
            cx, cy, w, h = map(float, q[1:5])
        except ValueError:
            problems.append("parse")
            continue
        if not (0 <= k < nc):
            problems.append(f"cls {k} out of range")
            continue
        if AVATAR_SPAN[0] <= k <= AVATAR_SPAN[1]:
            continue   # avatar ids belong to the other model
        if w <= 0 or h <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1:
            problems.append("bad box")
            continue
        cx, cy = min(max(cx, w / 2), 1 - w / 2), min(max(cy, h / 2), 1 - h / 2)
        rows.append((k, cx, cy, min(w, 1.0), min(h, 1.0)))
    return rows, problems


def dhash64(path: str) -> int | None:
    cv2.setNumThreads(1)
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    s = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def gray_small(path: str) -> np.ndarray | None:
    """96x54 grayscale vector for the pixel-level duplicate check (mean absolute difference)."""
    cv2.setNumThreads(1)
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if im is None:
        return None
    return cv2.resize(im, (96, 54), interpolation=cv2.INTER_AREA).astype(np.int16).ravel()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def out_name(item: dict) -> str:
    return f"{item['src']}__{item['stem']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-clean", action="store_true", help="refresh data/ui_v3_old from the review working copies")
    ap.add_argument("--clean", action="store_true", help="wipe the output dataset first")
    ap.add_argument("--hd-leak", type=int, default=4, help="dhash mode: drop val frames within this dhash distance of a train frame")
    ap.add_argument("--leak-mode", choices=("dhash", "pixel"), default="pixel",
                    help="pixel (default, 09-14): drop a val frame only when it is a near-exact copy of a train frame "
                         "(96x54 grayscale mean abs diff <= --pixel-mad); dhash: the old coarse 8x8 rule")
    ap.add_argument("--pixel-mad", type=float, default=2.0, help="pixel mode threshold (0-255 scale)")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()
    if args.import_clean:
        import_clean()
        return 0

    names = read_names()
    nc = len(names)
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    items = gold_items(GOLD, "train") + old_items() + gold_items(GOLD_VAL, "val")
    print(f"sources: gold {sum(1 for i in items if i['split'] == 'train' and (GOLD / i['src']).is_dir())}"
          f" / old {sum(1 for i in items if i['split'] == 'train' and (OLD / i['src']).is_dir())}"
          f" / val {sum(1 for i in items if i['split'] == 'val')}")

    # 1 parse every label, keep the frame even when it has zero boxes (verified negatives)
    parsed = {}
    n_problem = 0
    for it in items:
        rows, problems = parse_label(it["label"], nc)
        if problems:
            n_problem += 1
            print(f"  [label] {it['src']}/{it['stem']}: {Counter(problems)}")
        parsed[id(it)] = rows

    # 2 leak check: val vs train. 09-14 user ruling: the UI never changes, so a val frame of the same page as a
    #    train frame is legitimate; only near-exact copies (same page AND same content, e.g. a re-recording of a
    #    static list) are leaks. pixel mode measures that directly; the old dhash rule dropped 213 of 489 val frames.
    paths = sorted({str(it["img"]) for it in items})
    leaked = []
    kept = []
    if args.leak_mode == "pixel":
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            vecs = dict(zip(paths, ex.map(gray_small, paths, chunksize=16)))
        train_m = np.stack([vecs[str(it["img"])] for it in items
                            if it["split"] == "train" and vecs.get(str(it["img"])) is not None])
        for it in items:
            if it["split"] == "val":
                v = vecs.get(str(it["img"]))
                if v is None or float(np.abs(train_m - v[None, :]).mean(axis=1).min()) <= args.pixel_mad:
                    leaked.append(it)
                    continue
            kept.append(it)
        print(f"leak check: dropped {len(leaked)} val frames that are near-exact copies of a train frame (MAD <= {args.pixel_mad})")
    else:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            hashes = dict(zip(paths, ex.map(dhash64, paths, chunksize=16)))
        train_h = [hashes[str(it["img"])] for it in items if it["split"] == "train" and hashes.get(str(it["img"])) is not None]
        for it in items:
            if it["split"] == "val":
                h = hashes.get(str(it["img"]))
                if h is None or any((h ^ t).bit_count() <= args.hd_leak for t in train_h):
                    leaked.append(it)
                    continue
            kept.append(it)
        print(f"leak check: dropped {len(leaked)} val frames within dhash {args.hd_leak} of train")

    # 3 duplicate stems inside a split would overwrite each other
    seen = Counter(out_name(it) for it in kept)
    dups = [k for k, v in seen.items() if v > 1]
    if dups:
        print(f"  [error] duplicate output names: {dups[:5]}")
        return 1

    # 4 write
    per_cls = {"train": Counter(), "val": Counter()}
    per_cls_frames = {"train": Counter(), "val": Counter()}
    n_frames = Counter()
    n_boxes = Counter()
    manifest = []
    for it in kept:
        split = it["split"]
        rows = parsed[id(it)]
        name = out_name(it)
        link_or_copy(it["img"], out / "images" / split / (name + it["img"].suffix.lower()))
        with open(out / "labels" / split / (name + ".txt"), "w", encoding="utf-8", newline="\n") as fo:
            for k, cx, cy, w, h in rows:
                fo.write(f"{k} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        n_frames[split] += 1
        n_boxes[split] += len(rows)
        for k in {r[0] for r in rows}:
            per_cls_frames[split][k] += 1
        for r in rows:
            per_cls[split][r[0]] += 1
        manifest.append((split, it["src"], it["stem"], str(it["img"]), len(rows)))

    with open(out / "_manifest.csv", "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["split", "src", "stem", "img", "boxes"])
        w.writerows(manifest)
    with open(out / "_leaked_val.csv", "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["src", "stem", "img"])
        for it in leaked:
            w.writerow([it["src"], it["stem"], str(it["img"])])

    yaml_lines = [f"path: {out.as_posix()}", "train: images/train", "val: images/val", f"nc: {nc}", "names:"]
    for i, n in enumerate(names):
        yaml_lines.append(f"  {i}: '{n}'")
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # 5 per-class report
    ui_ids = [i for i in range(nc) if not (AVATAR_SPAN[0] <= i <= AVATAR_SPAN[1]) and not names[i].startswith("_")]
    with open(out / "_classes_report.csv", "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["id", "name", "train_boxes", "train_frames", "val_boxes", "val_frames"])
        for i in ui_ids:
            w.writerow([i, names[i], per_cls["train"][i], per_cls_frames["train"][i], per_cls["val"][i], per_cls_frames["val"][i]])
    thin = [(per_cls["train"][i], i) for i in ui_ids if per_cls["train"][i] < 10]
    noval = [i for i in ui_ids if per_cls["train"][i] >= 10 and per_cls["val"][i] == 0]
    print(f"train {n_frames['train']} frames / {n_boxes['train']} boxes; val {n_frames['val']} frames / {n_boxes['val']} boxes; label problems {n_problem}")
    print(f"UI classes {len(ui_ids)}: train<10 {len(thin)}, train>=10 but val=0 {len(noval)}")
    for c, i in sorted(thin):
        print(f"  thin {i:4d} {names[i]:28s} train {c:3d} val {per_cls['val'][i]:3d}")
    print(f"data.yaml -> {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
