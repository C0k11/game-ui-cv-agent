"""Visualize v4 best_manual.pt predictions on the 29 manual val frames.

Why: test_inference_real.py uses user-curated trajectory frames, many of
which turned out to be transition/loading screens with no avatars.
Manual val frames are guaranteed to have GT — perfect for sanity-check.

Outputs:
  - _test_v4_manual/pred/manual__*.jpg     (boxes drawn, ultralytics auto-save)
  - _test_v4_manual/_summary.md            (per-image counts + top-3 conf + GT count)
"""
from __future__ import annotations
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = Path(
    "D:/Project/ml_cache/models/yolo/runs/fused_avatar_yolo26x_v4/weights/best_manual.pt"
)
DATASET = Path("D:/Project/ml_cache/models/yolo/dataset/fused_avatar_v1")
OUT_DIR = REPO / "_test_v4_manual"


def count_gt(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), type=Path)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[err] weights missing: {weights}")
        return 1
    OUT_DIR.mkdir(exist_ok=True)

    val_img_dir = DATASET / "images" / "val"
    val_lbl_dir = DATASET / "labels" / "val"
    manual_imgs = sorted(p for p in val_img_dir.iterdir() if p.name.startswith("manual__"))
    if not manual_imgs:
        print(f"[err] no manual frames in {val_img_dir}")
        return 1

    from ultralytics import YOLO
    model = YOLO(str(weights))
    classes = model.names

    summary = [f"# v4 manual val sanity check — {weights.name}", ""]
    summary.append(f"- weights: `{weights}`")
    summary.append(f"- conf: {args.conf}, imgsz: {args.imgsz}, frames: {len(manual_imgs)}")
    summary.append("")
    summary.append("| frame | GT boxes | pred boxes | top-3 (conf) |")
    summary.append("|---|---:|---:|---|")

    total_gt = 0
    total_pred = 0
    perfect_recall_count = 0
    for src in manual_imgs:
        gt_n = count_gt(val_lbl_dir / (src.stem + ".txt"))
        r = model.predict(
            source=str(src),
            conf=args.conf,
            imgsz=args.imgsz,
            save=True,
            project=str(OUT_DIR),
            name="pred",
            exist_ok=True,
            verbose=False,
        )[0]
        boxes = r.boxes
        pred_n = len(boxes) if boxes is not None else 0
        top3 = []
        if boxes is not None and pred_n > 0:
            confs = boxes.conf.cpu().numpy().tolist()
            clss = boxes.cls.cpu().numpy().astype(int).tolist()
            pairs = sorted(zip(confs, clss), reverse=True)[:3]
            top3 = [f"{classes[c]}({co:.2f})" for co, c in pairs]
        if pred_n >= gt_n and gt_n > 0:
            perfect_recall_count += 1
        total_gt += gt_n
        total_pred += pred_n
        summary.append(f"| {src.name.replace('manual__frames__', '')} | {gt_n} | {pred_n} | {', '.join(top3) or '(none)'} |")
        print(f"[{src.name}] GT={gt_n}, pred={pred_n}")

    summary.append("")
    summary.append(f"## Totals")
    summary.append(f"- GT box count: {total_gt}")
    summary.append(f"- pred box count: {total_pred} (ratio {total_pred/max(total_gt,1):.2f})")
    summary.append(f"- frames with pred_n >= GT_n: {perfect_recall_count}/{len(manual_imgs)}")
    summary.append("")
    summary.append(f"## View")
    summary.append(f"- Annotated images: `{OUT_DIR}/pred/manual__*.jpg`")

    (OUT_DIR / "_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"\n[done] summary -> {OUT_DIR / '_summary.md'}")
    print(f"        images  -> {OUT_DIR / 'pred'}")
    print(f"\nTotals: {total_pred} pred / {total_gt} GT, {perfect_recall_count}/{len(manual_imgs)} frames with full coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
