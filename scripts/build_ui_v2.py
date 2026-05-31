"""Build the ui_v2 dataset for ui_yolo26m_v5 (warm-start fine-tune).

v5 plan (vs ui_v1 which v4 trained on):
  • Schema = MASTER _classes.txt (451 cls, incl 450 选择购买). ui_v1 was 450.
  • REAL sources are DEDUPED by content md5 — run_20260521_103956_distinct was
    628 unique frames blown to 12003 via 200x oversample copies (95% dup). We
    keep the 628 uniques. Heavy dup was the v1/v2/v3 overfit driver; v4 only
    survived it via aug+synth. Cut it.
  • NEW real data merged: run_20260531_110516 (1052) + run_20260531_143201 (37
    shop frames — fills 选择购买/已选中/绿勾).
  • cls92 战术大赛对战选择区域 KEPT (no DROP_CLASSES).
  • MODERATE re-oversample: classes with 8..TARGET unique frames → duplicate to
    TARGET (~30) so dedup doesn't starve the old-only event/battle classes that
    aren't in the new captures. Classes with <8 unique are NOT duped (that's the
    overfit trap — they rely on synth / future real captures instead).
  • 450 选择购买: explicit higher target (50 ≈ ×3 of its 17 real frames) per user.
  • SYNTH kept (441/442/449 bond): _synth_bond{,_goto,_enter} — proven in v4.
  • Val = _ui_val_pool (blind to weak cls; only for early-stop mechanics — real
    judgement is dashboard visual on the shipped model).

Output: D:/Project/ml_cache/models/yolo/dataset/ui_v2/
Usage: py scripts/build_ui_v2.py [--clean]
"""
from __future__ import annotations
import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"
MASTER = RAW / "_classes.txt"
OUT_ROOT = Path("D:/Project/ml_cache/models/yolo/dataset/ui_v2")

sys.path.insert(0, str(REPO / "scripts"))
from build_ui_dataset import sanitize_label_text  # noqa: E402

REAL_SOURCES = [
    "run_20260521_103956_distinct",  # 628 unique (dedup from 12003)
    "run_20260527_094158",
    "run_20260527_101545",
    "run_20260518_002646",
    "run_20260529_000756",
    "run_20260529_123209",
    "run_20260531_110516",           # NEW 1052
    "run_20260531_143201",           # NEW 37 shop
]
SYNTH_SOURCES = ["_synth_bond", "_synth_bond_goto", "_synth_bond_enter"]
VAL_SOURCE = "_ui_val_pool"

TARGET = 30            # moderate oversample floor (was 200 — the overfit driver)
TARGET_OVERRIDE = {450: 50}   # 选择购买: ~x3 its 17 real frames
MIN_UNIQUE = 8         # don't oversample classes thinner than this (overfit trap)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def label_classes(cleaned: str):
    cs = set()
    for ln in cleaned.splitlines():
        ln = ln.strip()
        if ln:
            cs.add(int(ln.split()[0]))
    return cs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    master = MASTER.read_text(encoding="utf-8").splitlines()
    nc = len(master)
    print(f"[schema] master = {nc} classes (last: {master[-1]!r})")

    # Verify each source's classes.txt is a PREFIX of master (no reorder drift).
    for s in REAL_SOURCES + SYNTH_SOURCES + [VAL_SOURCE]:
        cf = RAW / s / "classes.txt"
        if not cf.exists():
            continue
        sc = cf.read_text(encoding="utf-8").splitlines()
        if sc != master[:len(sc)]:
            for i in range(min(len(sc), nc)):
                if sc[i] != master[i]:
                    print(f"[!] SCHEMA DRIFT in {s}: idx {i} = {sc[i]!r} vs master {master[i]!r}")
                    return 1
    print("[schema] all sources prefix-match master ✓")

    if args.clean and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    img_tr = OUT_ROOT / "images" / "train"
    lbl_tr = OUT_ROOT / "labels" / "train"
    img_va = OUT_ROOT / "images" / "val"
    lbl_va = OUT_ROOT / "labels" / "val"
    for d in (img_tr, lbl_tr, img_va, lbl_va):
        d.mkdir(parents=True, exist_ok=True)

    # ── gather REAL frames, dedup by content md5 ──
    seen_md5 = {}
    uniques = []   # (src, stem, jpg, cleaned_txt, classes)
    dropped_overflow = 0
    n_raw = 0
    for s in REAL_SOURCES:
        sd = RAW / s
        if not sd.is_dir():
            print(f"[!] missing real source {sd}")
            continue
        for jpg in sorted(sd.glob("*.jpg")):
            txt = sd / (jpg.stem + ".txt")
            if not txt.exists():
                continue
            n_raw += 1
            h = md5(jpg)
            if h in seen_md5:
                continue
            seen_md5[h] = True
            cleaned = sanitize_label_text(txt.read_text(encoding="utf-8"))
            # drop any out-of-range cls (>= nc)
            keep = [ln for ln in cleaned.splitlines()
                    if ln.strip() and int(ln.split()[0]) < nc]
            dropped_overflow += len(cleaned.splitlines()) - len(keep)
            cleaned = ("\n".join(keep) + "\n") if keep else ""
            uniques.append((s, jpg.stem, jpg, cleaned, label_classes(cleaned)))
    print(f"[real] {n_raw} raw frames → {len(uniques)} unique (dedup removed "
          f"{n_raw - len(uniques)} dup copies); dropped {dropped_overflow} out-of-range boxes")

    # ── synth frames (no dedup) ──
    synth = []
    for s in SYNTH_SOURCES:
        sd = RAW / s
        if not sd.is_dir():
            continue
        for jpg in sorted(sd.glob("*.jpg")):
            txt = sd / (jpg.stem + ".txt")
            if not txt.exists():
                continue
            cleaned = sanitize_label_text(txt.read_text(encoding="utf-8"))
            keep = [ln for ln in cleaned.splitlines()
                    if ln.strip() and int(ln.split()[0]) < nc]
            cleaned = ("\n".join(keep) + "\n") if keep else ""
            synth.append((s, jpg.stem, jpg, cleaned, label_classes(cleaned)))
    print(f"[synth] {len(synth)} frames")

    base = uniques + synth   # one entry each (the de-duped real + synth)

    # ── moderate oversample: bring 8..TARGET classes up to TARGET ──
    cls_to_frames = defaultdict(list)
    for i, (_, _, _, _, cs) in enumerate(base):
        for c in cs:
            cls_to_frames[c].append(i)
    dup_entries = []   # indices into base, duplicated
    for c, idxs in cls_to_frames.items():
        n = len(idxs)
        tgt = TARGET_OVERRIDE.get(c, TARGET)
        if n < MIN_UNIQUE or n >= tgt:
            continue
        need = tgt - n
        for k in range(need):
            dup_entries.append(idxs[k % n])
    print(f"[oversample] +{len(dup_entries)} dup entries (target {TARGET}, "
          f"450→{TARGET_OVERRIDE[450]}, min_unique {MIN_UNIQUE})")

    # ── write train (uniques + synth once, then dups with __dN suffix) ──
    def write_entry(entry, dst_stem):
        s, stem, jpg, cleaned, _ = entry
        dj = img_tr / f"{dst_stem}.jpg"
        if dj.exists():
            dj.unlink()
        try:
            dj.symlink_to(jpg.resolve())
        except OSError:
            shutil.copy2(jpg, dj)
        (lbl_tr / f"{dst_stem}.txt").write_text(cleaned, encoding="utf-8")

    neg = 0
    for entry in base:
        s, stem = entry[0], entry[1]
        write_entry(entry, f"{s}__{stem}")
        if not entry[3].strip():
            neg += 1
    dup_count = defaultdict(int)
    for bi in dup_entries:
        entry = base[bi]
        s, stem = entry[0], entry[1]
        dup_count[bi] += 1
        write_entry(entry, f"{s}__{stem}__d{dup_count[bi]}")
    n_train = len(base) + len(dup_entries)
    print(f"[train] {n_train} frames ({len(base)} unique+synth, {len(dup_entries)} dups, "
          f"{neg} negatives)")

    # ── val from _ui_val_pool ──
    vd = RAW / VAL_SOURCE
    n_val = 0
    for jpg in sorted(vd.glob("*.jpg")):
        txt = vd / (jpg.stem + ".txt")
        if not txt.exists():
            continue
        cleaned = sanitize_label_text(txt.read_text(encoding="utf-8"))
        keep = [ln for ln in cleaned.splitlines()
                if ln.strip() and int(ln.split()[0]) < nc]
        cleaned = ("\n".join(keep) + "\n") if keep else ""
        dj = img_va / f"{jpg.stem}.jpg"
        if dj.exists():
            dj.unlink()
        try:
            dj.symlink_to(jpg.resolve())
        except OSError:
            shutil.copy2(jpg, dj)
        (lbl_va / f"{jpg.stem}.txt").write_text(cleaned, encoding="utf-8")
        n_val += 1
    print(f"[val] {n_val} frames (_ui_val_pool)")

    # ── data.yaml ──
    yaml = [f"path: {OUT_ROOT.as_posix()}", "train: images/train", "val: images/val",
            f"nc: {nc}", "names:"]
    for i, n in enumerate(master):
        yaml.append(f"  {i}: '{n.replace(chr(39), chr(92) + chr(39))}'")
    (OUT_ROOT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"[done] {OUT_ROOT}  (nc={nc}, train={n_train}, val={n_val})")

    # ── report key class counts (instances in train) ──
    inst = defaultdict(int)
    for entry in base:
        for ln in entry[3].splitlines():
            if ln.strip():
                inst[int(ln.split()[0])] += 1
    for bi in dup_entries:
        for ln in base[bi][3].splitlines():
            if ln.strip():
                inst[int(ln.split()[0])] += 1
    print("[key cls instances in train]")
    for c in (92, 450, 54, 403, 55, 441, 442, 449):
        print(f"  {c} {master[c][:22]:<22} {inst.get(c,0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
