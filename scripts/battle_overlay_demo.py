"""Real-time battle overlay demo — DXcam + YOLO player_head + SORT tracker.

Captures via DXcam (fast desktop duplication), runs YOLOv8n battle_heads.pt
for player character head detection, feeds through BoxTracker for smooth
lock-on, renders on YoloOverlay at 250Hz.

Usage:
    python scripts/battle_overlay_demo.py [--fps 15]

Press Ctrl+C to stop.
"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BATTLE_MODEL = Path(r"D:\Project\ml_cache\models\yolo\battle_heads.pt")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=240, help="Target detection FPS")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold")
    args = parser.parse_args()

    import dxcam
    import numpy as np
    from ultralytics import YOLO
    from scripts.win_capture import find_window_by_title_substring, find_largest_visible_child
    from scripts.yolo_overlay import YoloOverlay

    # ── 1. Find MuMu window ──
    hwnd = find_window_by_title_substring("MuMu")
    if not hwnd:
        print("[Error] MuMu window not found")
        return
    child = find_largest_visible_child(hwnd)
    render_hwnd = child if child else hwnd

    # Get window screen rect for DXcam region
    import ctypes
    import ctypes.wintypes as wt
    user32 = ctypes.windll.user32

    rc = wt.RECT()
    user32.GetWindowRect(render_hwnd, ctypes.byref(rc))
    region = (rc.left, rc.top, rc.right, rc.bottom)
    print(f"[Info] Window hwnd={render_hwnd} region={region}")
    print(f"[Info] Capture size: {region[2]-region[0]}x{region[3]-region[1]}")

    # ── 2. Init DXcam ──
    camera = dxcam.create(output_idx=0, output_color="BGR")
    test_frame = camera.grab(region=region)
    if test_frame is None:
        print("[Error] DXcam grab returned None — is the window visible?")
        return
    print(f"[Info] DXcam OK: frame {test_frame.shape[1]}x{test_frame.shape[0]}")

    # ── 3. Load YOLO model ──
    if not BATTLE_MODEL.exists():
        print(f"[Error] Model not found: {BATTLE_MODEL}")
        return
    model = YOLO(str(BATTLE_MODEL))
    print(f"[Info] YOLO battle_heads loaded: {BATTLE_MODEL.name}")

    # ── 4. Start overlay ──
    overlay = YoloOverlay(render_hwnd)
    overlay.start()
    print(f"[Info] Overlay started (250Hz render, {args.fps} FPS detection)")
    print("[Info] Press Ctrl+C to stop\n")

    interval = 1.0 / args.fps
    frame_count = 0
    det_total = 0

    try:
        while True:
            t0 = time.perf_counter()

            # Grab frame via DXcam (< 2ms)
            frame = camera.grab(region=region)
            if frame is None:
                time.sleep(0.005)
                continue

            # YOLO inference (< 5ms on 4090 nano)
            results = model.predict(frame, imgsz=640, conf=args.conf,
                                    verbose=False, device=0)

            # Parse detections
            boxes_out = []
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = r.names.get(cls_id, f"cls{cls_id}")
                        # Normalize to 0-1
                        fh, fw = frame.shape[:2]
                        boxes_out.append({
                            "cls": cls_name,
                            "conf": conf,
                            "x1": x1 / fw,
                            "y1": y1 / fh,
                            "x2": x2 / fw,
                            "y2": y2 / fh,
                        })

            # Feed to overlay tracker
            overlay.update(boxes_out)

            frame_count += 1
            n = len(boxes_out)
            det_total += n
            elapsed = time.perf_counter() - t0
            fps_actual = 1.0 / max(elapsed, 0.001)
            print(f"[{frame_count:5d}] {n} heads | {elapsed*1000:5.1f}ms ({fps_actual:5.1f} FPS) | avg {det_total/frame_count:.1f}/frame", end='\r')

            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n[Info] Stopped after {frame_count} frames, avg {det_total/max(1,frame_count):.1f} detections/frame")
    finally:
        overlay.stop()
        print("[Info] Overlay stopped")


if __name__ == "__main__":
    main()
