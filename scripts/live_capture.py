"""Live screenshot capture for YOLO training dataset.

Takes periodic screenshots of the game window while running.
Uses Win32 GDI PrintWindow (PW_RENDERFULLCONTENT) — works with
DirectX/Unity hardware-accelerated windows. Zero extra dependencies
(ctypes only, no pywin32).

Frame-diff dedup: skips near-identical frames to avoid wasting disk
and labeling effort on static scenes.

Usage:
    python scripts/live_capture.py
    python scripts/live_capture.py --interval 1.2 --limit 800
    python scripts/live_capture.py --window "Blue Archive"

Press Ctrl+C to stop early.
"""

import argparse
import ctypes
import ctypes.wintypes
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path(r"D:\Project\ml_cache\models\yolo\dataset\images\train")
DIFF_THRESHOLD = 8  # mean pixel diff below this → skip (near-duplicate)

# Win32 GDI via ctypes (zero extra dependencies)
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


def find_window(title_fragment: str = "Blue Archive"):
    """Find game window by partial title match."""
    result = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_cb(hwnd, _):
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if title_fragment.lower() in buf.value.lower():
                result.append(hwnd)
        return True

    user32.EnumWindows(enum_cb, 0)
    return result[0] if result else None


def capture_window(hwnd) -> np.ndarray | None:
    """Capture window client area to BGR numpy array via GDI PrintWindow."""
    rect = ctypes.wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right, rect.bottom
    if w <= 0 or h <= 0:
        return None

    hdc_src = user32.GetDC(hwnd)
    hdc_dst = gdi32.CreateCompatibleDC(hdc_src)
    bmp = gdi32.CreateCompatibleBitmap(hdc_src, w, h)
    gdi32.SelectObject(hdc_dst, bmp)

    # PW_RENDERFULLCONTENT (flag=2) for hardware-accelerated windows
    user32.PrintWindow(hwnd, hdc_dst, 2)

    bmi = _BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h  # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0

    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(hdc_dst, bmp, 0, h, buf, ctypes.byref(bmi), 0)

    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(hdc_dst)
    user32.ReleaseDC(hwnd, hdc_src)

    # BGRA → BGR numpy array (cv2 native format)
    frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def main():
    parser = argparse.ArgumentParser(description="Live game screenshot capture for YOLO dataset")
    parser.add_argument("--interval", type=float, default=1.5, help="capture interval in seconds")
    parser.add_argument("--limit", type=int, default=500, help="max screenshots to capture")
    parser.add_argument("--window", type=str, default="Blue Archive", help="game window title keyword")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUTPUT_DIR.glob("*.jpg"))) + len(list(OUTPUT_DIR.glob("*.png")))

    hwnd = find_window(args.window)
    if hwnd is None:
        print(f"ERROR: no window matching '{args.window}'. Make sure the game is running.")
        return

    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    print(f"Window: '{buf.value}'")
    print(f"Output: {OUTPUT_DIR} ({existing} existing)")
    print(f"Settings: interval={args.interval}s limit={args.limit} diff_threshold={DIFF_THRESHOLD}")
    print("Press Ctrl+C to stop.\n")

    count = 0
    skipped = 0
    last_gray = None

    try:
        while count < args.limit:
            frame = capture_window(hwnd)
            if frame is None:
                print("  (empty capture, window minimized?)")
                time.sleep(args.interval)
                continue

            # Frame-diff dedup: skip near-identical frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if last_gray is not None:
                diff = cv2.absdiff(gray, last_gray)
                if np.mean(diff) < DIFF_THRESHOLD:
                    skipped += 1
                    time.sleep(0.2)
                    continue
            last_gray = gray

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"headpat_{ts}.jpg"
            cv2.imwrite(str(OUTPUT_DIR / filename), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 98])
            count += 1
            print(f"  [{count:03d}/{args.limit}] {filename}  {frame.shape[1]}x{frame.shape[0]}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    total = len(list(OUTPUT_DIR.glob("*.jpg"))) + len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\nCaptured {count} frames ({skipped} duplicates skipped)")
    print(f"Total in dataset: {total} images")
    print(f"\nNext: label with AnyLabeling -> python scripts/train_yolo.py")


if __name__ == "__main__":
    main()
