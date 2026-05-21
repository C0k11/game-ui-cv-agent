"""Test fused_avatar_yolo26x best.pt on real trajectory frames — no labeling
required, just sanity-check what the model predicts on actual game UI.

Outputs:
  - _test_inference/{cafe,schedule}/<tick>__pred.jpg   (boxes drawn)
  - _test_inference/_summary.md                        (per-image detection counts + top-5 conf)

Usage:
    py scripts/test_inference_real.py
    py scripts/test_inference_real.py --conf 0.20  (lower to see borderline detections)
"""
from __future__ import annotations
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = Path("D:/Project/ml_cache/models/yolo/runs/fused_avatar_yolo26x/weights/best.pt")
OUT_DIR = REPO / "_test_inference"

# User-curated test frames
CAFE_FRAMES = [
    "data/trajectories/run_20260516_234050/tick_0042.jpg",
    "data/trajectories/run_20260516_234050/tick_0043.jpg",
    "data/trajectories/run_20260516_234945/tick_0010.jpg",
    "data/trajectories/run_20260516_234945/tick_0011.jpg",
    "data/trajectories/run_20260517_043723/tick_0011.jpg",
    "data/trajectories/run_20260517_043723/tick_0012.jpg",
]
SCHEDULE_FRAMES = [
    "data/trajectories/run_20260516_234945/tick_0067.jpg",
    "data/trajectories/run_20260516_234050/tick_0098.jpg",
    "data/trajectories/run_20260415_215901/tick_0042.jpg",
    "data/trajectories/run_20260504_230936/tick_0073.jpg",
    "data/trajectories/run_20260504_232633/tick_0064.jpg",
    "data/trajectories/run_20260305_171137/tick_0069.jpg",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    if not WEIGHTS.exists():
        print(f"[err] weights missing: {WEIGHTS}")
        return 1
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "cafe").mkdir(exist_ok=True)
    (OUT_DIR / "schedule").mkdir(exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    classes = model.names

    summary = ["# fused_avatar_yolo26x best.pt — real-frame inference test", ""]
    summary.append(f"- weights: `{WEIGHTS}`")
    summary.append(f"- conf threshold: {args.conf}, imgsz: {args.imgsz}")
    summary.append("")

    for tag, frames in (("cafe", CAFE_FRAMES), ("schedule", SCHEDULE_FRAMES)):
        summary.append(f"## {tag.upper()}")
        summary.append("")
        for rel in frames:
            src = REPO / rel
            if not src.exists():
                print(f"[skip] {rel} missing")
                summary.append(f"- ⚠️ `{rel}` MISSING")
                continue
            r = model.predict(
                source=str(src),
                conf=args.conf,
                imgsz=args.imgsz,
                save=True,
                project=str(OUT_DIR),
                name=tag,
                exist_ok=True,
                verbose=False,
            )[0]
            boxes = r.boxes
            n = len(boxes) if boxes is not None else 0
            top5 = []
            if boxes is not None and n > 0:
                confs = boxes.conf.cpu().numpy().tolist()
                clss = boxes.cls.cpu().numpy().astype(int).tolist()
                pairs = sorted(zip(confs, clss), reverse=True)[:5]
                top5 = [f"{classes[c]}({co:.2f})" for co, c in pairs]
            summary.append(f"- **{rel.split('/')[-2]}/{rel.split('/')[-1]}**: {n} detections, top: {', '.join(top5) if top5 else '(none)'}")
            print(f"[{tag}] {rel}: {n} detections")
        summary.append("")

    summary.append(f"## Where to view")
    summary.append("")
    summary.append("- Annotated images: `_test_inference/cafe/*.jpg` and `_test_inference/schedule/*.jpg`")
    summary.append("- Each box shows: class name + confidence")
    (OUT_DIR / "_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"\n[done] summary -> {OUT_DIR / '_summary.md'}")
    print(f"        images  -> {OUT_DIR}/cafe and {OUT_DIR}/schedule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
