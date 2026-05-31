"""Evaluate the active (or a given) UI model on the hand-captured WEAK-class
gold val pool (data/raw_images/_val_weak), reporting recall/precision/AP50 for
the bond/momotalk weak classes 441/442/449 specifically.

WHY a dedicated script: the ui_v1 dataset's 51-frame val pool contains ZERO
instances of 441/442/449, so its val mAP is blind to exactly the classes we
retrained to fix.  The only trustworthy signal is FRESH real captures (not in
train, not oversample-duplicated).  Capture them via the dashboard
(Capture → Split=Val → Purpose=UI Weak → frames land in _val_weak/frames/),
label 441/442/449 in the Annotate page, then run this.

It assembles _val_weak into a clean YOLO val structure (sanitizing labels the
same way build_ui_dataset does — truncate to 5 tokens, drop bad lines) using
the EXACT 450-class schema the model trained on, then runs model.val().

Usage:
  py scripts/eval_weak_val.py                      # active ui model
  py scripts/eval_weak_val.py --weights <best.pt>  # a specific checkpoint
  py scripts/eval_weak_val.py --conf 0.05          # lower conf = recall view
Output: console table + per-weak-class GT counts (so a forgotten label shows
as GT=0 rather than a silent 0 recall).
"""
from __future__ import annotations
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VAL_SRC = REPO / "data" / "raw_images" / "_val_weak"
UI_YAML = Path(r"D:\Project\ml_cache\models\yolo\dataset\ui_v1\data.yaml")
OUT_ROOT = Path(r"D:\Project\ml_cache\models\yolo\dataset\_val_weak_eval")
WEAK = (441, 442, 449, 92)  # 92 = arena 对战选择区域 (kept for v5; watch too)

# Reuse the dataset builder's label sanitizer (truncate to 5 tokens, drop
# DROP_CLASSES + malformed lines) so frontend-written labels with trailing
# tokens don't get silently treated as corrupt.
sys.path.insert(0, str(REPO / "scripts"))
from build_ui_dataset import sanitize_label_text  # noqa: E402


def parse_names(yaml_path: Path) -> dict:
    names = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(\d+):\s*['\"](.*)['\"]\s*$", line)
        if m:
            names[int(m.group(1))] = m.group(2)
    return names


def main() -> int:
    argv = sys.argv[1:]
    weights = None
    conf = 0.10
    iou = 0.5
    src_name = "_val_weak"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--weights":
            weights = argv[i + 1]; i += 2; continue
        if a == "--conf":
            conf = float(argv[i + 1]); i += 2; continue
        if a == "--iou":
            iou = float(argv[i + 1]); i += 2; continue
        if a == "--src":
            src_name = argv[i + 1]; i += 2; continue
        i += 1

    if not weights:
        reg = json.loads((REPO / "data" / "model_registry.json").read_text(encoding="utf-8"))
        ui = reg["ui"]
        weights = ui["versions"][ui["active"]]["path"]

    val_src = REPO / "data" / "raw_images" / src_name
    src = val_src / "frames" if (val_src / "frames").is_dir() else val_src
    pairs = []
    for jpg in sorted(src.glob("*.jpg")):
        txt = jpg.with_suffix(".txt")
        if txt.exists():
            pairs.append((jpg, txt))
    if not pairs:
        print(f"[!] no labeled (jpg+txt) frames in {src}")
        print("    Capture via dashboard (Split=Val, Purpose=UI Weak) and label "
              "441/442/449 in the Annotate page first.")
        return 1

    names = parse_names(UI_YAML)
    nc = (max(names) + 1) if names else 450

    # Assemble clean YOLO val structure (wipe + rebuild — idempotent)
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    img_val = OUT_ROOT / "images" / "val"
    lbl_val = OUT_ROOT / "labels" / "val"
    img_val.mkdir(parents=True, exist_ok=True)
    lbl_val.mkdir(parents=True, exist_ok=True)

    gt_counts = {c: 0 for c in WEAK}
    total_inst = 0
    for jpg, txt in pairs:
        stem = jpg.stem
        dst_jpg = img_val / f"{stem}.jpg"
        try:
            dst_jpg.symlink_to(jpg.resolve())
        except OSError:
            shutil.copy2(jpg, dst_jpg)
        cleaned = sanitize_label_text(txt.read_text(encoding="utf-8"))
        (lbl_val / f"{stem}.txt").write_text(cleaned, encoding="utf-8")
        for line in cleaned.splitlines():
            if line.strip():
                c = int(line.split()[0]); total_inst += 1
                if c in gt_counts:
                    gt_counts[c] += 1

    # data.yaml — train points at val too (val-only eval; YOLO won't train)
    yaml = [f"path: {OUT_ROOT.as_posix()}", "train: images/val", "val: images/val",
            f"nc: {nc}", "names:"]
    for idx in range(nc):
        safe = names.get(idx, str(idx)).replace("'", "\\'")
        yaml.append(f"  {idx}: '{safe}'")
    yaml_path = OUT_ROOT / "data.yaml"
    yaml_path.write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"weights = {weights}")
    print(f"val pool: {len(pairs)} labeled frames, {total_inst} instances total")
    print("WEAK-class GT instance counts in this val pool:")
    for c in WEAK:
        flag = "  <-- NOT LABELED (or not present)" if gt_counts[c] == 0 else ""
        print(f"  cls {c} {names.get(c, '?')[:24]:<24} GT={gt_counts[c]}{flag}")
    if all(v == 0 for v in gt_counts.values()):
        print("[!] none of 441/442/449 are labeled in the val pool — eval would be "
              "meaningless. Label them in the Annotate page first.")
        return 1
    print()

    from ultralytics import YOLO
    m = YOLO(weights)
    names_m = m.names
    metrics = m.val(data=str(yaml_path), split="val", imgsz=960, device=0,
                    conf=conf, iou=iou, plots=False, verbose=False)
    box = metrics.box
    idxs = [int(c) for c in box.ap_class_index]
    print(f"\n=== eval on _val_weak (conf={conf} iou={iou}, {len(idxs)} cls with GT) ===")
    print(f"{'cls':>4} {'name':<24} {'recall':>6} {'prec':>6} {'ap50':>6}  flag")

    def row(cid):
        if cid not in idxs:
            return None
        j = idxs.index(cid)
        return float(box.r[j]), float(box.p[j]), float(box.ap50[j])

    # WEAK classes first (the whole point), then the rest
    def fmt(cid, r, p, ap):
        nm = (names_m.get(cid) if isinstance(names_m, dict) else names_m[cid]) or str(cid)
        flag = ("BROKEN" if r < 0.30 else "WEAK" if r < 0.60 else
                "MEH" if r < 0.85 else "ok")
        print(f"{cid:>4} {str(nm)[:24]:<24} {r:>6.3f} {p:>6.3f} {ap:>6.3f}  {flag}")

    print("-- WEAK classes (target) --")
    for c in WEAK:
        rr = row(c)
        if rr:
            fmt(c, *rr)
        else:
            print(f"{c:>4} {names.get(c, '?')[:24]:<24}   (no GT in val — not labeled)")
    others = [c for c in idxs if c not in WEAK]
    if others:
        print("-- other classes present in val --")
        for c in sorted(others):
            fmt(c, *row(c))
    print(f"\noverall mAP50={float(box.map50):.3f}  mAP50-95={float(box.map):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
