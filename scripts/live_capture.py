"""Live screenshot capture for YOLO training dataset.

Takes periodic screenshots of the game window while running.
Use this while the game is in cafe to collect diverse headpat training data.

Usage:
    python scripts/live_capture.py                  # default: 2s interval, 200 max
    python scripts/live_capture.py --interval 1.5   # faster capture
    python scripts/live_capture.py --limit 500      # more images
    python scripts/live_capture.py --window "BlueArchive"  # custom window title

Press Ctrl+C to stop early.
"""

import ctypes
import ctypes.wintypes
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(r"D:\Project\ml_cache\models\yolo\dataset\images\train")

# Win32 screenshot via GDI (no extra dependencies)
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

SRCCOPY = 0x00CC0020


def find_window(title_fragment: str = "BlueArchive"):
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


def capture_window(hwnd) -> bytes:
    """Capture window client area to PNG bytes via GDI."""
    from PIL import Image
    import io

    rect = ctypes.wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right, rect.bottom
    if w <= 0 or h <= 0:
        return b""

    hdc_src = user32.GetDC(hwnd)
    hdc_dst = gdi32.CreateCompatibleDC(hdc_src)
    bmp = gdi32.CreateCompatibleBitmap(hdc_src, w, h)
    gdi32.SelectObject(hdc_dst, bmp)

    # PrintWindow with PW_RENDERFULLCONTENT (flag=2) for hardware-accelerated windows
    user32.PrintWindow(hwnd, hdc_dst, 2)

    # Read bitmap bits
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
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

    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1)
    img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def main():
    interval = 2.0
    limit = 200
    window_title = "BlueArchive"

    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--interval" and i + 1 < len(args):
            interval = float(args[i + 1])
        elif a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
        elif a == "--window" and i + 1 < len(args):
            window_title = args[i + 1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUTPUT_DIR.glob("live_*.png")))
    print(f"Live capture: window='{window_title}' interval={interval}s limit={limit}")
    print(f"Output: {OUTPUT_DIR} ({existing} existing live captures)")
    print("Press Ctrl+C to stop.\n")

    hwnd = find_window(window_title)
    if hwnd is None:
        print(f"ERROR: No window found matching '{window_title}'")
        print("Make sure the game is running.")
        sys.exit(1)

    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    print(f"Found window: '{buf.value}'")

    count = 0
    try:
        while count < limit:
            ts = int(time.time() * 1000)
            data = capture_window(hwnd)
            if not data:
                print("  (empty capture, window minimized?)")
                time.sleep(interval)
                continue

            filename = f"live_{ts}.png"
            (OUTPUT_DIR / filename).write_bytes(data)
            count += 1
            if count % 10 == 0:
                print(f"  captured {count}/{limit}...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\nStopped by user.")

    print(f"\nDone. Captured {count} screenshots.")
    print(f"Total in dataset: {len(list(OUTPUT_DIR.glob('*.png')))} images")
    print(f"\nNext: label with AnyLabeling → python scripts/train_yolo.py")


if __name__ == "__main__":
    main()
