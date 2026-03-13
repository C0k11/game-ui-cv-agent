import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.florence_vision import expand_florence_queries


_RAW_TO_CANONICAL = {
    "叉叉": "关闭按钮",
    "叉叉1": "关闭按钮",
    "叉叉2": "关闭按钮",
    "momotalk的叉叉": "关闭按钮",
    "公告叉叉": "关闭按钮",
    "内嵌公告的叉": "关闭按钮",
    "游戏内很多页面窗口的叉": "关闭按钮",
    "邮件箱": "邮件箱",
    "邮箱": "邮件箱",
    "返回键": "返回键",
    "返回按钮": "返回键",
    "主界面按钮": "主界面按钮",
    "Home按钮": "主界面按钮",
    "左切换": "左切换",
    "右切换": "右切换",
    "锁": "锁",
    "课程表锁": "锁",
    "全体课程表": "全体课程表",
}

_DEFAULT_LABELS = ["关闭按钮", "邮件箱", "返回键", "主界面按钮", "左切换", "右切换", "锁", "全体课程表"]


def _pick_queries(label: str, limit: int) -> List[str]:
    queries = [alias for _, alias in expand_florence_queries([label]) if alias]
    uniq: List[str] = []
    seen = set()
    for q in queries:
        key = str(q).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(str(q).strip())
    return uniq[: max(1, limit)]


def _iter_images(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(root.glob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _load_classes(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm_stem(stem: str) -> str:
    return str(stem or "").replace("（", "(").replace("）", ")").strip()


def _capture_to_canonical(path: Path) -> Optional[str]:
    stem = _norm_stem(path.stem)
    if "左切换" in stem:
        return "左切换"
    if "右切换" in stem:
        return "右切换"
    if "邮箱" in stem or "邮件" in stem:
        return "邮件箱"
    if "返回" in stem:
        return "返回键"
    if "Home按钮" in stem or "主界面" in stem:
        return "主界面按钮"
    if "锁" in stem:
        return "锁"
    if "课程表" in stem and "全体" in stem:
        return "全体课程表"
    if "叉" in stem or "关闭" in stem:
        return "关闭按钮"
    return _RAW_TO_CANONICAL.get(stem)


def _yolo_xywh_to_xyxy(parts: List[str], width: int, height: int) -> List[int]:
    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
    x1 = max(0, int(round((xc - bw / 2.0) * width)))
    y1 = max(0, int(round((yc - bh / 2.0) * height)))
    x2 = min(width, int(round((xc + bw / 2.0) * width)))
    y2 = min(height, int(round((yc + bh / 2.0) * height)))
    return [x1, y1, x2, y2]


def _serialize_boxes(query: str, boxes: List[List[int]], size: Tuple[int, int], processor) -> str:
    if not boxes:
        return ""
    quantized = processor.post_processor.box_quantizer.quantize(torch.tensor(boxes, dtype=torch.float32), size)
    parts: List[str] = []
    for locs in quantized.tolist():
        parts.append(f"{query}<loc_{locs[0]}><loc_{locs[1]}><loc_{locs[2]}><loc_{locs[3]}>")
    return "".join(parts)


def _build_yolo_records(root: Path, selected: set[str]) -> List[Dict[str, Any]]:
    classes = _load_classes(root / "classes.txt")
    records: List[Dict[str, Any]] = []
    for img_path in _iter_images(root):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        raw = txt_path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        with Image.open(img_path) as im:
            width, height = im.size
        annotations: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id < 0 or cls_id >= len(classes):
                continue
            raw_label = classes[cls_id].strip()
            canonical = _RAW_TO_CANONICAL.get(raw_label, raw_label)
            if canonical not in selected:
                continue
            box = _yolo_xywh_to_xyxy(parts, width, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            annotations.append({
                "label": raw_label,
                "canonical_label": canonical,
                "bbox": box,
            })
        if annotations:
            records.append({
                "image": str(img_path.resolve()),
                "width": width,
                "height": height,
                "source": "yolo_raw",
                "annotations": annotations,
            })
    return records


def _load_backgrounds(root: Path, limit: int) -> List[Path]:
    images = list(_iter_images(root))
    if limit > 0:
        return images[:limit]
    return images


def _build_synthetic_records(captures_dir: Path, selected: set[str], backgrounds: List[Path], out_dir: Path, synthetic_per_capture: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    synth_dir = out_dir / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    capture_images = [p for p in captures_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]
    for cap_path in sorted(capture_images):
        canonical = _capture_to_canonical(cap_path)
        if canonical not in selected:
            continue
        icon = Image.open(cap_path).convert("RGBA")
        if icon.width < 2 or icon.height < 2:
            continue
        for idx in range(max(1, synthetic_per_capture)):
            if backgrounds:
                bg_path = backgrounds[(idx + rng.randint(0, max(0, len(backgrounds) - 1))) % len(backgrounds)]
                bg = Image.open(bg_path).convert("RGBA")
            else:
                bg = Image.new("RGBA", (1280, 720), (240, 240, 240, 255))
            scale = rng.uniform(0.85, 1.2)
            new_w = max(8, int(round(icon.width * scale)))
            new_h = max(8, int(round(icon.height * scale)))
            icon_resized = icon.resize((new_w, new_h), Image.Resampling.LANCZOS)
            max_x = max(0, bg.width - new_w)
            max_y = max(0, bg.height - new_h)
            x = rng.randint(0, max_x) if max_x > 0 else 0
            y = rng.randint(0, max_y) if max_y > 0 else 0
            canvas = bg.copy()
            canvas.alpha_composite(icon_resized, (x, y))
            out_path = synth_dir / canonical / f"{_norm_stem(cap_path.stem)}_{idx:03d}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.convert("RGB").save(out_path)
            records.append({
                "image": str(out_path.resolve()),
                "width": canvas.width,
                "height": canvas.height,
                "source": "capture_synthetic",
                "annotations": [{
                    "label": _norm_stem(cap_path.stem),
                    "canonical_label": canonical,
                    "bbox": [x, y, x + new_w, y + new_h],
                }],
            })
    return records


def _records_to_train(records: List[Dict[str, Any]], processor, query_limit: int) -> List[Dict[str, Any]]:
    train: List[Dict[str, Any]] = []
    for rec in records:
        grouped: Dict[str, List[List[int]]] = {}
        for ann in rec.get("annotations") or []:
            grouped.setdefault(str(ann.get("canonical_label") or ann.get("label") or ""), []).append(list(ann.get("bbox") or []))
        size = (int(rec["width"]), int(rec["height"]))
        for canonical, boxes in grouped.items():
            boxes = [b for b in boxes if isinstance(b, list) and len(b) == 4 and b[2] > b[0] and b[3] > b[1]]
            if not boxes:
                continue
            for query in _pick_queries(canonical, query_limit):
                target = _serialize_boxes(query, boxes, size, processor)
                if not target:
                    continue
                train.append({
                    "image": rec["image"],
                    "prompt": f"<OPEN_VOCABULARY_DETECTION> {query}",
                    "target_text": target,
                    "query": query,
                    "canonical_label": canonical,
                    "source": rec.get("source") or "unknown",
                    "width": rec["width"],
                    "height": rec["height"],
                    "boxes": boxes,
                })
    return train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-root", default=r"D:\Project\ai game secretary\data\raw_images\run_20260228_235254")
    ap.add_argument("--captures-dir", default=r"D:\Project\ai game secretary\data\captures")
    ap.add_argument("--out", default=r"D:\Project\ai game secretary\data\florence_ui_dataset")
    ap.add_argument("--model-id", default="microsoft/Florence-2-large-ft")
    ap.add_argument("--cache-dir", default=r"D:\Project\ml_cache\models")
    ap.add_argument("--labels", default=",".join(_DEFAULT_LABELS))
    ap.add_argument("--synthetic-per-capture", type=int, default=12)
    ap.add_argument("--background-limit", type=int, default=256)
    ap.add_argument("--queries-per-label", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    yolo_root = Path(args.yolo_root)
    captures_dir = Path(args.captures_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {x.strip() for x in str(args.labels or "").split(",") if x.strip()}
    if not selected:
        selected = set(_DEFAULT_LABELS)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, cache_dir=args.cache_dir)

    records = _build_yolo_records(yolo_root, selected)
    backgrounds = _load_backgrounds(yolo_root, args.background_limit) if yolo_root.exists() else []
    records.extend(_build_synthetic_records(captures_dir, selected, backgrounds, out_dir, args.synthetic_per_capture, args.seed))

    labels_jsonl = out_dir / "labels.jsonl"
    with labels_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train_records = _records_to_train(records, processor, args.queries_per_label)
    florence_jsonl = out_dir / "florence_lora.jsonl"
    with florence_jsonl.open("w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats: Dict[str, int] = {}
    for rec in records:
        for ann in rec.get("annotations") or []:
            key = str(ann.get("canonical_label") or ann.get("label") or "")
            stats[key] = stats.get(key, 0) + 1
    summary = {
        "out_dir": str(out_dir.resolve()),
        "labels_jsonl": str(labels_jsonl.resolve()),
        "florence_lora_jsonl": str(florence_jsonl.resolve()),
        "image_records": len(records),
        "train_records": len(train_records),
        "selected_labels": sorted(selected),
        "annotation_counts": stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
