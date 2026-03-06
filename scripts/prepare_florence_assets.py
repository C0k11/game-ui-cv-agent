import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _iter_images(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _safe_rel(base: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return p.name


def _blur_avatar(src: Path, dst: Path) -> bool:
    img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        blurred = cv2.GaussianBlur(bgr, (0, 0), 6.0)
        out = np.dstack([blurred, alpha])
    else:
        out = cv2.GaussianBlur(img, (0, 0), 6.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imencode(dst.suffix or ".png", out)[1].tofile(str(dst)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--raw-run", default=r"D:\Project\ai game secretary\data\raw_images\run_20260228_235254")
    ap.add_argument("--captures-dir", default=r"D:\Project\ai game secretary\data\captures")
    ap.add_argument("--out", default=r"D:\Project\ai game secretary\data\florence_assets")
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    raw_run = Path(args.raw_run)
    captures_dir = Path(args.captures_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    blurred_dir = out_dir / "blurred_avatars"
    manifest_path = out_dir / "manifest.jsonl"

    entries = []

    if raw_run.exists():
        for img in _iter_images(raw_run):
            entries.append({
                "type": "raw_run",
                "path": str(img.resolve()),
                "rel": _safe_rel(raw_run, img),
                "source": raw_run.name,
            })

    if captures_dir.exists():
        for img in _iter_images(captures_dir):
            pstr = str(img.resolve())
            label = img.stem
            rec_type = "capture_template"
            if "角色头像" in pstr:
                rec_type = "avatar_reference"
            entries.append({
                "type": rec_type,
                "path": pstr,
                "rel": _safe_rel(captures_dir, img),
                "label": label,
            })

    avatar_dir = captures_dir / "角色头像"
    if avatar_dir.exists():
        for img in sorted(avatar_dir.glob("*.png")):
            out_path = blurred_dir / img.name
            if _blur_avatar(img, out_path):
                entries.append({
                    "type": "avatar_blurred",
                    "path": str(out_path.resolve()),
                    "rel": _safe_rel(out_dir, out_path),
                    "label": img.stem,
                    "source": str(img.resolve()),
                })

    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in entries:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "out_dir": str(out_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "entries": len(entries),
        "raw_run_exists": raw_run.exists(),
        "captures_exists": captures_dir.exists(),
        "blurred_avatar_count": len(list(blurred_dir.glob("*.png"))) if blurred_dir.exists() else 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
