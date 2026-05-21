"""Evaluate a fused_avatar checkpoint on the MANUAL val subset only.

Why: our val set is 29 manual + 450 synth + 19 negatives. The synth subset
shares distribution with train synth (same templates, same char pool, same
aug pipeline), so default Ultralytics .val() reports an inflated mAP50.
The 29 manual frames are real game screenshots — they're the only signal
that means anything for production.

Usage:
    py scripts/eval_manual_val.py                # default: v4 best.pt
    py scripts/eval_manual_val.py --weights ...  # custom checkpoint
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile

import yaml
from ultralytics import YOLO

DATASET = Path("D:/Project/ml_cache/models/yolo/dataset/fused_avatar_v1")
RUNS = Path("D:/Project/ml_cache/models/yolo/runs")


def make_manual_yaml() -> Path:
    """Write a temporary data.yaml whose val: points only to manual__ frames."""
    src_yaml = DATASET / "data.yaml"
    cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))

    val_img_dir = DATASET / "images" / "val"
    manual_imgs = sorted(p for p in val_img_dir.iterdir() if p.name.startswith("manual__"))
    if not manual_imgs:
        print(f"[!] no manual__ images in {val_img_dir}", file=sys.stderr)
        sys.exit(2)

    list_path = DATASET / "_manual_val_list.txt"
    list_path.write_text(
        "\n".join(str(p).replace("\\", "/") for p in manual_imgs),
        encoding="utf-8",
    )

    new_yaml = DATASET / "data_manual_val.yaml"
    cfg["val"] = str(list_path).replace("\\", "/")
    new_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    print(f"[+] wrote {new_yaml} (val list = {len(manual_imgs)} manual frames)")
    return new_yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        default=str(RUNS / "fused_avatar_yolo26x_v4" / "weights" / "best.pt"),
    )
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.001, help="ultralytics default")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--name", default="manual_val_eval")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[!] weights not found: {weights_path}", file=sys.stderr)
        return 2

    yaml_path = make_manual_yaml()

    print(f"[+] loading {weights_path}")
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(yaml_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        plots=False,
        verbose=False,
        name=args.name,
        exist_ok=True,
    )

    box = metrics.box
    out = {
        "weights": str(weights_path),
        "val_subset": "manual_only",
        "n_images": 29,
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }

    # Per-class breakdown (top movers vs baseline 0)
    if hasattr(box, "ap_class_index") and len(box.ap_class_index) > 0:
        names = model.names
        per_class = []
        for i, ci in enumerate(box.ap_class_index):
            per_class.append({
                "class": names[int(ci)] if isinstance(names, dict) else names[int(ci)],
                "ap50": float(box.ap50[i]),
                "ap50_95": float(box.ap[i]),
            })
        per_class.sort(key=lambda x: x["ap50"])
        out["worst_10"] = per_class[:10]
        out["best_10"] = per_class[-10:][::-1]

    print()
    print("=" * 60)
    print(f"📊 Manual-only val (29 frames) — {weights_path.name}")
    print("=" * 60)
    print(f"  mAP50    = {out['mAP50']:.4f}")
    print(f"  mAP50-95 = {out['mAP50_95']:.4f}")
    print(f"  P        = {out['precision']:.4f}")
    print(f"  R        = {out['recall']:.4f}")
    if "worst_10" in out:
        print()
        print("  Worst 10 classes (AP50):")
        for c in out["worst_10"]:
            print(f"    {c['class']:<8s}  AP50={c['ap50']:.3f}  AP50-95={c['ap50_95']:.3f}")

    out_path = Path("D:/Project/ai game secretary") / f"_{args.name}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[+] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
