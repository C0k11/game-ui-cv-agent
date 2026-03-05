"""MuMu Runner: High-FPS capture → YOLO+OCR → Pipeline → ADB click.

Standalone runner that captures from the MuMu emulator window,
runs the DailyPipeline skill loop, displays YOLO bounding box overlay,
and sends click/swipe/back actions via ADB.

Usage:
    py mumu_runner.py                        # auto-detect MuMu window
    py mumu_runner.py --title "MuMu"         # custom window title
    py mumu_runner.py --adb-port 7555        # custom ADB port
    py mumu_runner.py --dry-run              # no clicks, just overlay
    py mumu_runner.py --fps 60               # lower capture FPS
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# ── Repo setup ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.win_capture import (
    find_window_by_title_substring,
    find_largest_visible_child,
    get_client_rect_on_screen,
    capture_client,
)


# ── ADB Input ───────────────────────────────────────────────────────────

_MUMU_ADB_CANDIDATES = [
    Path(r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"),
    Path(r"C:\Program Files\Netease\MuMu\nx_main\adb.exe"),
    Path(r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"),
]


def _find_adb() -> str:
    """Find adb executable: MuMu bundled first, then PATH."""
    for p in _MUMU_ADB_CANDIDATES:
        if p.is_file():
            print(f"[ADB] Using MuMu bundled: {p}")
            return str(p)
    # fallback: adb in PATH
    return "adb"


class AdbInput:
    """Send touch/key events to MuMu via ADB."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7555):
        self.addr = f"{host}:{port}"
        self._connected = False
        self._adb = _find_adb()

    def connect(self) -> bool:
        try:
            r = subprocess.run(
                [self._adb, "connect", self.addr],
                capture_output=True, text=True, timeout=5,
            )
            self._connected = "connected" in r.stdout.lower() or "already" in r.stdout.lower()
            print(f"[ADB] connect {self.addr}: {r.stdout.strip()}")
            return self._connected
        except FileNotFoundError:
            print(f"[ADB] ERROR: adb not found at '{self._adb}'")
            return False
        except Exception as e:
            print(f"[ADB] connect error: {e}")
            return False

    def _shell(self, cmd: str, timeout: float = 3.0) -> bool:
        try:
            subprocess.run(
                [self._adb, "-s", self.addr, "shell", cmd],
                capture_output=True, timeout=timeout,
            )
            return True
        except Exception:
            return False

    def tap(self, x: int, y: int) -> bool:
        return self._shell(f"input tap {int(x)} {int(y)}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, dur_ms: int = 400) -> bool:
        return self._shell(
            f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(dur_ms)}",
            timeout=max(5.0, dur_ms / 1000.0 + 2.0),
        )

    def back(self) -> bool:
        return self._shell("input keyevent 4")  # KEYCODE_BACK

    def screen_size(self) -> Tuple[int, int]:
        """Get the Android screen resolution via ADB (landscape-corrected).

        MuMu reports physical size as portrait (e.g. 720x1280) but Blue Archive
        runs in landscape (rotation=1). ADB `input tap` uses the landscape
        coordinate system (1280x720). We detect this via dumpsys display.
        """
        w, h = 1280, 720  # fallback (landscape)
        # Try wm size first
        try:
            r = subprocess.run(
                [self._adb, "-s", self.addr, "shell", "wm", "size"],
                capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.strip().splitlines():
                if "size" in line.lower():
                    parts = line.split(":")[-1].strip().split("x")
                    if len(parts) == 2:
                        w, h = int(parts[0]), int(parts[1])
                        break
        except Exception:
            pass

        # Check override display (active after rotation) via dumpsys
        try:
            r = subprocess.run(
                [self._adb, "-s", self.addr, "shell",
                 "dumpsys", "display"],
                capture_output=True, text=True, timeout=5,
            )
            import re
            # Look for mOverrideDisplayInfo with "real WxH"
            m = re.search(r'mOverrideDisplayInfo.*?real\s+(\d+)\s*x\s*(\d+)', r.stdout)
            if m:
                ow, oh = int(m.group(1)), int(m.group(2))
                if ow > 0 and oh > 0:
                    w, h = ow, oh
                    print(f"[ADB] Override display: {w}x{h}")
                    return w, h
        except Exception:
            pass

        # If portrait (w < h), swap for landscape (Blue Archive is always landscape)
        if w < h:
            w, h = h, w
        return w, h


# ── Window Capture ──────────────────────────────────────────────────────

class MuMuCapture:
    """Capture frames from MuMu emulator window using BitBlt."""

    def __init__(self, title_substring: str = "MuMu"):
        self.title = title_substring
        self.hwnd: Optional[int] = None
        self.render_hwnd: Optional[int] = None
        self._client_w = 0
        self._client_h = 0

    def find_window(self) -> bool:
        self.hwnd = find_window_by_title_substring(self.title)
        if self.hwnd is None:
            return False
        child = find_largest_visible_child(int(self.hwnd))
        self.render_hwnd = int(child) if child else int(self.hwnd)
        self._update_size()
        print(f"[Capture] Found MuMu window: hwnd={self.hwnd} render={self.render_hwnd} size={self._client_w}x{self._client_h}")
        return True

    def _update_size(self) -> None:
        try:
            r = get_client_rect_on_screen(self.render_hwnd)
            self._client_w = max(0, r.right - r.left)
            self._client_h = max(0, r.bottom - r.top)
        except Exception:
            self._client_w = 0
            self._client_h = 0

    @property
    def client_size(self) -> Tuple[int, int]:
        return self._client_w, self._client_h

    def grab(self) -> Optional[np.ndarray]:
        """Capture one frame as BGR numpy array."""
        if self.render_hwnd is None:
            return None
        try:
            pil_img = capture_client(self.render_hwnd)
            if pil_img is None:
                return None
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            if w != self._client_w or h != self._client_h:
                self._client_w = w
                self._client_h = h
            return frame
        except Exception as e:
            print(f"[Capture] grab error: {e}")
            return None


# ── Overlay Drawing ─────────────────────────────────────────────────────

# Class-based color palette (consistent colors per class)
_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {}

def _color_for_class(cls_name: str) -> Tuple[int, int, int]:
    if cls_name not in _CLASS_COLORS:
        h = hash(cls_name) % 360
        # HSV → BGR for vivid colors
        hsv = np.array([[[h / 2, 200, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        _CLASS_COLORS[cls_name] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return _CLASS_COLORS[cls_name]


def draw_overlay(
    frame: np.ndarray,
    yolo_boxes: list,
    ocr_boxes: list,
    action: Dict[str, Any],
    skill_name: str = "",
    sub_state: str = "",
    fps: float = 0.0,
) -> np.ndarray:
    """Draw YOLO boxes, OCR text, action target, and HUD on frame."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # ── YOLO bounding boxes ──
    for box in yolo_boxes:
        x1 = int(box.x1 * w)
        y1 = int(box.y1 * h)
        x2 = int(box.x2 * w)
        y2 = int(box.y2 * h)
        color = _color_for_class(box.cls_name)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{box.cls_name} {box.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── OCR text (small, semi-transparent) ──
    for box in ocr_boxes:
        x1 = int(box.x1 * w)
        y1 = int(box.y1 * h)
        x2 = int(box.x2 * w)
        y2 = int(box.y2 * h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 1)

    # ── Action click target crosshair ──
    action_type = action.get("action", "")
    if action_type == "click":
        tx, ty = action.get("target", [0.5, 0.5])
        px, py = int(tx * w), int(ty * h)
        cv2.drawMarker(overlay, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.circle(overlay, (px, py), 20, (0, 0, 255), 2)

    # ── HUD bar ──
    hud_h = 36
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (0, 0, 0), -1)
    reason = action.get("reason", "")[:80]
    hud = f"FPS:{fps:.0f} | {skill_name}/{sub_state} | {action_type}: {reason}"
    cv2.putText(overlay, hud, (8, hud_h - 10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    return overlay


# ── Action Executor ─────────────────────────────────────────────────────

def execute_action(
    action: Dict[str, Any],
    adb: AdbInput,
    android_w: int,
    android_h: int,
) -> None:
    """Convert normalized pipeline action to ADB commands."""
    action_type = action.get("action", "")

    if action_type == "click":
        nx, ny = action.get("target", [0.5, 0.5])
        ax = int(nx * android_w)
        ay = int(ny * android_h)
        adb.tap(ax, ay)

    elif action_type == "back":
        adb.back()

    elif action_type == "swipe":
        frm = action.get("from", [0.5, 0.5])
        to = action.get("to", [0.5, 0.5])
        dur_ms = action.get("duration_ms", 400)
        adb.swipe(
            int(frm[0] * android_w), int(frm[1] * android_h),
            int(to[0] * android_w), int(to[1] * android_h),
            dur_ms,
        )


# ── Main Loop ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MuMu Runner: BA automation via emulator")
    parser.add_argument("--title", default="MuMu", help="Window title substring (default: MuMu)")
    parser.add_argument("--adb-host", default="127.0.0.1")
    parser.add_argument("--adb-port", type=int, default=7555, help="ADB port (MuMu12=7555)")
    parser.add_argument("--fps", type=int, default=240, help="Target capture FPS")
    parser.add_argument("--dry-run", action="store_true", help="Overlay only, no clicks")
    parser.add_argument("--no-overlay", action="store_true", help="Disable overlay window")
    parser.add_argument("--overlay-scale", type=float, default=0.5, help="Overlay window scale (default 0.5)")
    args = parser.parse_args()

    # 1. Find MuMu window
    cap = MuMuCapture(title_substring=args.title)
    if not cap.find_window():
        print(f"[ERROR] MuMu window not found (title contains '{args.title}')")
        print("  Make sure MuMu Player is running.")
        sys.exit(1)

    # 2. Connect ADB
    adb = AdbInput(host=args.adb_host, port=args.adb_port)
    if not args.dry_run:
        if not adb.connect():
            print("[WARNING] ADB connection failed. Running in dry-run mode.")
            args.dry_run = True

    # Get Android screen resolution for coordinate mapping
    android_w, android_h = adb.screen_size()
    print(f"[Info] Android resolution: {android_w}x{android_h}")

    # 3. Start pipeline
    from brain.pipeline import DailyPipeline
    pipe = DailyPipeline()
    pipe.start()
    print(f"[Info] Pipeline started with {len(pipe._skill_order)} skills")

    # 3b. Start YOLO overlay on game window
    overlay = None
    if not args.no_overlay:
        try:
            from scripts.yolo_overlay import YoloOverlay
            overlay = YoloOverlay(cap.render_hwnd)
            overlay.start()
            print(f"[Info] YOLO overlay started on render_hwnd={cap.render_hwnd}")
        except Exception as e:
            print(f"[WARN] YOLO overlay failed: {e}")

    # 4. Main loop
    window_name = "BA Bot - MuMu"
    frame_interval = 1.0 / max(1, args.fps)
    tick_interval = 0.5  # Pipeline tick every 500ms (OCR+YOLO is ~200ms)
    last_tick_time = 0.0
    last_action: Dict[str, Any] = {"action": "wait", "reason": "starting"}
    fps_counter = 0
    fps_time = time.perf_counter()
    fps_display = 0.0

    print(f"[Info] Running at {args.fps}fps capture, {'DRY RUN' if args.dry_run else 'LIVE'}. Press Q to quit.")

    try:
        while True:
            t0 = time.perf_counter()

            # Capture frame
            frame = cap.grab()
            if frame is None:
                time.sleep(0.1)
                if not cap.find_window():
                    print("[WARN] MuMu window lost, waiting...")
                continue

            # FPS counter
            fps_counter += 1
            elapsed_fps = t0 - fps_time
            if elapsed_fps >= 1.0:
                fps_display = fps_counter / elapsed_fps
                fps_counter = 0
                fps_time = t0

            # Pipeline tick (rate-limited)
            now = time.perf_counter()
            if now - last_tick_time >= tick_interval and pipe.is_running:
                last_tick_time = now

                # Save frame to temp for trajectory (pipeline needs a path)
                tmp_path = str(REPO_ROOT / "data" / "_mumu_frame.jpg")
                cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                action = pipe.tick_from_frame(frame, screenshot_path=tmp_path)
                last_action = action
                action_type = action.get("action", "")
                reason = action.get("reason", "")

                skill = pipe.current_skill
                skill_name = skill.name if skill else "—"
                sub_state = skill.sub_state if skill else ""

                print(f"[t{pipe._total_ticks:04d}] {skill_name}/{sub_state}: {action_type} — {reason[:60]}")

                # Update native overlay with YOLO boxes
                if overlay and overlay.is_alive:
                    screen = pipe.last_screen
                    if screen and screen.yolo_boxes:
                        overlay.update(screen.yolo_boxes)
                    else:
                        overlay.update([])

                # Execute action
                if not args.dry_run and action_type not in ("done", "wait"):
                    execute_action(action, adb, android_w, android_h)

                # Handle wait durations
                if action_type == "wait":
                    wait_s = action.get("duration_ms", 300) / 1000.0
                    tick_interval = max(0.3, wait_s)
                elif action_type == "done":
                    if not pipe.is_running:
                        print("[Info] Pipeline complete!")
                        # Keep overlay running so user can see final state
                else:
                    tick_interval = 0.5

            # Draw overlay
            if not args.no_overlay:
                from brain.pipeline import read_screen_from_frame
                # Use cached screen state from last tick for overlay
                # (avoid running YOLO again just for drawing)
                skill = pipe.current_skill
                skill_name = skill.name if skill else "—"
                sub_state = skill.sub_state if skill else ""

                # Quick YOLO-only pass for real-time tracking overlay
                from brain.pipeline import _run_yolo_on_image
                fh, fw = frame.shape[:2]
                yolo_boxes = _run_yolo_on_image(frame, fw, fh)

                overlay = draw_overlay(
                    frame,
                    yolo_boxes=yolo_boxes,
                    ocr_boxes=[],  # skip OCR boxes in overlay (too noisy)
                    action=last_action,
                    skill_name=skill_name,
                    sub_state=sub_state,
                    fps=fps_display,
                )

                # Scale down for display
                if args.overlay_scale != 1.0:
                    dw = int(overlay.shape[1] * args.overlay_scale)
                    dh = int(overlay.shape[0] * args.overlay_scale)
                    overlay = cv2.resize(overlay, (dw, dh), interpolation=cv2.INTER_AREA)

                cv2.imshow(window_name, overlay)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("[Info] Quit requested.")
                break

            # FPS limiter
            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("[Info] Interrupted.")
    finally:
        if overlay:
            try:
                overlay.stop()
            except Exception:
                pass
        pipe.stop()
        cv2.destroyAllWindows()
        print("[Info] MuMu Runner stopped.")


if __name__ == "__main__":
    main()
