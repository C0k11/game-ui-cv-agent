"""Step 5: Evaluate OCR model accuracy on trajectory data.

Compares the fine-tuned model vs the default model on real trajectory crops.
Reports per-category accuracy and common failure patterns.

Usage:
    py -3 scripts/ocr_training/05_evaluate.py [--sample 2000]
"""
import argparse
import json
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocr_training"))
from ba_vocab import CORRECTIONS, get_all_vocab

DATA_DIR = REPO / "data" / "ocr_training"
TRAJ_DIR = REPO / "data" / "trajectories"
CUSTOM_MODEL = REPO / "data" / "ocr_model" / "ba_rec.onnx"


def load_ocr_engine(use_custom: bool):
    """Load RapidOCR with either custom or default model."""
    from rapidocr_onnxruntime import RapidOCR

    if use_custom and CUSTOM_MODEL.exists():
        print(f"Loading CUSTOM model: {CUSTOM_MODEL}")
        engine = RapidOCR(rec_model_path=str(CUSTOM_MODEL))
    else:
        if use_custom:
            print(f"[WARN] Custom model not found at {CUSTOM_MODEL}, using default")
        print("Loading DEFAULT model")
        engine = RapidOCR()
    return engine


def apply_corrections(text: str) -> str:
    """Apply known corrections to get ground truth."""
    if text in CORRECTIONS:
        return CORRECTIONS[text]
    for wrong, right in sorted(CORRECTIONS.items(), key=lambda x: -len(x[0])):
        if wrong in text:
            text = text.replace(wrong, right)
    return text


def collect_eval_samples(max_samples: int) -> list[dict]:
    """Collect evaluation samples from trajectory data."""
    runs = sorted(TRAJ_DIR.glob("run_*"))
    all_samples = []

    for run_dir in runs:
        for tick_file in sorted(run_dir.glob("tick_*.json")):
            img_file = tick_file.with_suffix(".jpg")
            if not img_file.exists():
                continue

            try:
                tick_data = json.loads(tick_file.read_text("utf-8"))
            except Exception:
                continue

            for box in tick_data.get("ocr_boxes", []):
                text = box.get("text", "").strip()
                conf = box.get("conf", 0.0)
                if text and conf >= 0.5:
                    all_samples.append({
                        "img_path": str(img_file),
                        "box": box,
                        "original_text": text,
                        "corrected_text": apply_corrections(text),
                    })

    random.shuffle(all_samples)
    samples = all_samples[:max_samples]
    print(f"Collected {len(samples)} eval samples from {len(all_samples)} total")
    return samples


def crop_region(img: np.ndarray, box: dict) -> np.ndarray:
    """Crop text region from image."""
    h, w = img.shape[:2]
    x1 = max(0, int(box["x1"] * w) - 2)
    y1 = max(0, int(box["y1"] * h) - 2)
    x2 = min(w, int(box["x2"] * w) + 2)
    y2 = min(h, int(box["y2"] * h) + 2)
    return img[y1:y2, x1:x2]


def run_ocr_on_crop(engine, crop: np.ndarray) -> str:
    """Run OCR engine on a single crop and return recognized text."""
    if crop is None or crop.size == 0:
        return ""
    try:
        result, _ = engine(crop)
        if result:
            return "".join(line[1] for line in result)
    except Exception:
        pass
    return ""


def evaluate_model(engine, samples: list[dict], label: str) -> dict:
    """Evaluate OCR engine on samples, return metrics."""
    exact_match = 0
    contains_match = 0
    total = 0
    failures = []
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    vocab_set = set(get_all_vocab())
    img_cache = {}

    for i, sample in enumerate(samples):
        img_path = sample["img_path"]
        if img_path not in img_cache:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img_cache[img_path] = img
            # Keep cache manageable
            if len(img_cache) > 50:
                oldest = next(iter(img_cache))
                del img_cache[oldest]

        img = img_cache.get(img_path)
        if img is None:
            continue

        crop = crop_region(img, sample["box"])
        if crop.size == 0:
            continue

        predicted = run_ocr_on_crop(engine, crop)
        expected = sample["corrected_text"]

        total += 1
        is_exact = predicted.strip() == expected.strip()
        is_contains = expected.strip() in predicted.strip() or predicted.strip() in expected.strip()

        if is_exact:
            exact_match += 1
        if is_contains:
            contains_match += 1

        # Categorize
        cat = "other"
        if expected in vocab_set:
            cat = "vocab"
        elif any(c in expected for c in "0123456789"):
            cat = "numeric"
        category_stats[cat]["total"] += 1
        if is_exact:
            category_stats[cat]["correct"] += 1

        if not is_exact and len(failures) < 100:
            failures.append({
                "expected": expected,
                "predicted": predicted,
                "conf": sample["box"].get("conf", 0),
            })

        if (i + 1) % 500 == 0:
            print(f"  [{label}] {i+1}/{len(samples)} ...")

    return {
        "label": label,
        "total": total,
        "exact_match": exact_match,
        "contains_match": contains_match,
        "exact_acc": exact_match / max(total, 1),
        "contains_acc": contains_match / max(total, 1),
        "category_stats": dict(category_stats),
        "failures": failures,
    }


def print_report(result: dict):
    """Print evaluation report."""
    print(f"\n{'='*60}")
    print(f"Model: {result['label']}")
    print(f"{'='*60}")
    print(f"  Samples:        {result['total']}")
    print(f"  Exact match:    {result['exact_match']}/{result['total']} ({result['exact_acc']:.1%})")
    print(f"  Contains match: {result['contains_match']}/{result['total']} ({result['contains_acc']:.1%})")

    print(f"\n  Per-category:")
    for cat, stats in sorted(result["category_stats"].items()):
        acc = stats["correct"] / max(stats["total"], 1)
        print(f"    {cat:12s}: {stats['correct']}/{stats['total']} ({acc:.1%})")

    if result["failures"]:
        print(f"\n  Top failures:")
        for f in result["failures"][:20]:
            print(f"    '{f['expected']}' → '{f['predicted']}' (conf={f['conf']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR model on trajectory data")
    parser.add_argument("--sample", type=int, default=2000,
                        help="Number of samples to evaluate (default: 2000)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare custom vs default model")
    args = parser.parse_args()

    samples = collect_eval_samples(args.sample)
    if not samples:
        print("[ERROR] No evaluation samples found. Check trajectory data.")
        sys.exit(1)

    if args.compare and CUSTOM_MODEL.exists():
        # Compare both models
        print("\n--- Evaluating DEFAULT model ---")
        default_engine = load_ocr_engine(use_custom=False)
        default_result = evaluate_model(default_engine, samples, "DEFAULT (PP-OCRv3)")
        del default_engine

        print("\n--- Evaluating CUSTOM model ---")
        custom_engine = load_ocr_engine(use_custom=True)
        custom_result = evaluate_model(custom_engine, samples, "CUSTOM (BA fine-tuned)")
        del custom_engine

        print_report(default_result)
        print_report(custom_result)

        # Delta
        delta = custom_result["exact_acc"] - default_result["exact_acc"]
        print(f"\n{'='*60}")
        print(f"Accuracy improvement: {delta:+.1%}")
        if delta > 0:
            print(f"  Custom model is BETTER by {delta:.1%}")
        elif delta < 0:
            print(f"  Custom model is WORSE by {abs(delta):.1%} — consider more training")
        else:
            print(f"  Models are equal — consider more/different training data")
    else:
        # Evaluate whichever model is available
        use_custom = CUSTOM_MODEL.exists()
        engine = load_ocr_engine(use_custom=use_custom)
        label = "CUSTOM (BA fine-tuned)" if use_custom else "DEFAULT (PP-OCRv3)"
        result = evaluate_model(engine, samples, label)
        print_report(result)


if __name__ == "__main__":
    main()
