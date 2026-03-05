"""Export YOLO models to TensorRT (.engine) for faster inference on RTX 4090.

Usage:
    py scripts/export_tensorrt.py                    # Export full.pt
    py scripts/export_tensorrt.py --validate         # Export + validate on sample images
    py scripts/export_tensorrt.py --model avatar     # Export avatar_augmented.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

ML_CACHE = Path(r"D:\Project\ml_cache\models\yolo")
REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = REPO_ROOT / "data" / "raw_images" / "run_20260228_235254"

MODELS = {
    "full": ML_CACHE / "full.pt",
    "avatar": ML_CACHE / "avatar_augmented.pt",
}


def export_model(name: str, imgsz: int = 640, half: bool = True) -> Path:
    """Export a .pt model to TensorRT .engine format."""
    from ultralytics import YOLO

    pt_path = MODELS[name]
    if not pt_path.is_file():
        raise FileNotFoundError(f"Model not found: {pt_path}")

    print(f"\n{'='*60}")
    print(f"Exporting {name} -> TensorRT (imgsz={imgsz}, half={half})")
    print(f"Source: {pt_path}")
    print(f"{'='*60}")

    model = YOLO(str(pt_path))
    engine_path = model.export(
        format="engine",
        half=half,
        device=0,
        imgsz=imgsz,
        workspace=8,        # 8 GB workspace for RTX 4090
        verbose=True,
    )
    engine_path = Path(engine_path)
    print(f"\nExported: {engine_path} ({engine_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return engine_path


def validate(name: str, engine_path: Path, n_images: int = 20) -> None:
    """Compare .pt vs .engine inference speed and detections on sample images."""
    from ultralytics import YOLO

    pt_path = MODELS[name]
    if not pt_path.is_file() or not engine_path.is_file():
        print("Skipping validation: model files missing")
        return

    # Collect sample images
    images = []
    if VAL_DIR.is_dir():
        for ext in ("*.jpg", "*.png"):
            images.extend(sorted(VAL_DIR.glob(ext)))
    if not images:
        print(f"No validation images found in {VAL_DIR}")
        return
    images = images[:n_images]
    print(f"\nValidating on {len(images)} images from {VAL_DIR.name}")

    # Load both models
    pt_model = YOLO(str(pt_path))
    trt_model = YOLO(str(engine_path))

    # Warm up
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(3):
        pt_model(dummy, verbose=False)
        trt_model(dummy, verbose=False)

    # Benchmark
    pt_times = []
    trt_times = []
    pt_total_dets = 0
    trt_total_dets = 0

    for img_path in images:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # PyTorch inference
        t0 = time.perf_counter()
        pt_res = pt_model(img, conf=0.15, verbose=False, half=True)
        pt_times.append(time.perf_counter() - t0)
        pt_total_dets += sum(len(r.boxes) for r in pt_res)

        # TensorRT inference
        t0 = time.perf_counter()
        trt_res = trt_model(img, conf=0.15, verbose=False)
        trt_times.append(time.perf_counter() - t0)
        trt_total_dets += sum(len(r.boxes) for r in trt_res)

    print(f"\n{'='*60}")
    print(f"Results ({len(pt_times)} images):")
    print(f"  PyTorch (.pt):   avg={1000*sum(pt_times)/len(pt_times):.1f}ms  "
          f"total_dets={pt_total_dets}")
    print(f"  TensorRT (.engine): avg={1000*sum(trt_times)/len(trt_times):.1f}ms  "
          f"total_dets={trt_total_dets}")
    speedup = sum(pt_times) / max(sum(trt_times), 1e-9)
    print(f"  Speedup: {speedup:.2f}x")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO to TensorRT")
    parser.add_argument("--model", choices=["full", "avatar", "all"], default="full",
                        help="Which model to export")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--validate", action="store_true", help="Run validation benchmark")
    args = parser.parse_args()

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]

    for name in targets:
        if not MODELS[name].is_file():
            print(f"Skipping {name}: {MODELS[name]} not found")
            continue
        engine_path = export_model(name, imgsz=args.imgsz)
        if args.validate:
            validate(name, engine_path)


if __name__ == "__main__":
    main()
