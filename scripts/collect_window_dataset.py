import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.win_capture import capture_client, find_window_by_title_substring, get_client_rect_on_screen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True, help="window title substring")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max", type=int, default=0, help="max frames (0=unlimited)")
    args = ap.parse_args()

    hwnd = find_window_by_title_substring(args.title)
    if hwnd is None:
        raise SystemExit(f"window not found: {args.title}")

    root = Path(args.out)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / f"session_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.jsonl"

    interval = 1.0 / max(0.1, float(args.fps))
    i = 0
    t0 = time.time()

    with meta_path.open("a", encoding="utf-8") as f:
        while True:
            # Re-check window existence/rect each frame in case it moved
            if not hwnd:
                break
            
            try:
                rect = get_client_rect_on_screen(hwnd)
                rect_info = {
                    "left": rect.left,
                    "top": rect.top,
                    "width": rect.width,
                    "height": rect.height,
                    "x": rect.left,
                    "y": rect.top,
                }
            except Exception:
                rect_info = None

            try:
                img = capture_client(hwnd)
            except Exception as e:
                print(f"Capture failed: {e}")
                time.sleep(1)
                continue

            name = f"frame_{i:06d}.png"
            img_path = out_dir / name
            img.save(img_path)

            rec = {
                "image": str(img_path.relative_to(root)).replace("\\", "/"),
                "timestamp": time.time(),
                "index": i,
                "client_rect": rect_info,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            i += 1
            if args.max and i >= args.max:
                break

            dt = time.time() - t0
            target = i * interval
            sleep_s = max(0.0, target - dt)
            time.sleep(sleep_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
