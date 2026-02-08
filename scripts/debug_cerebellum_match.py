import argparse
from pathlib import Path
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--assets", default=str(Path("data") / "captures"))
    ap.add_argument("--confidence", type=float, default=0.8)
    ap.add_argument("--roi", default="")
    ap.add_argument("--templates", nargs="*", default=["内嵌公告的叉.png", "游戏内很多页面窗口的叉.png", "点击开始.png"])
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from cerebellum import Cerebellum

    roi = None
    if args.roi:
        parts = [p.strip() for p in str(args.roi).split(",")]
        if len(parts) == 4:
            roi = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))

    c = Cerebellum(assets_dir=args.assets, confidence=float(args.confidence))

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    shot = str(Path(args.shot).resolve())
    scr = cv2.imread(shot, cv2.IMREAD_COLOR)
    if scr is None:
        raise SystemExit(f"cannot read screenshot: {shot}")
    sh, sw = scr.shape[:2]
    print(f"screenshot={shot} size={sw}x{sh}")
    print(f"assets_dir={Path(args.assets).resolve()} confidence={float(args.confidence)} roi={roi}")

    for t in args.templates:
        p = Path(args.assets) / t
        img = None
        try:
            data = p.read_bytes()
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None
        if img is None:
            print(f"template={t} path={p} read=None")
        else:
            ch = 1
            try:
                ch = 1 if len(img.shape) == 2 else int(img.shape[2])
            except Exception:
                ch = 1
            th, tw = img.shape[:2]
            print(f"template={t} path={p} size={tw}x{th} channels={ch}")

        m = c.best_match(screenshot_path=shot, template_name=t, roi=roi)
        if m is None:
            print(f"  best_match=None")
        else:
            print(f"  best_score={m.score:.4f} center={m.center} bbox={m.bbox}")


if __name__ == "__main__":
    main()
