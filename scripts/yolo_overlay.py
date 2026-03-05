"""Transparent YOLO bounding-box overlay on game window.

Creates a Win32 layered window positioned exactly on top of a target
game window.  Draws YOLO detection boxes with class labels in real-time.
The overlay is click-through (WS_EX_TRANSPARENT) so it never interferes
with game input.

Usage (standalone test — draws random boxes on MuMu):
    py scripts/yolo_overlay.py --title "MuMu"

Usage (from code):
    from scripts.yolo_overlay import YoloOverlay
    overlay = YoloOverlay(render_hwnd)
    overlay.start()
    overlay.update(yolo_boxes)        # list of YoloBox or dicts
    overlay.stop()
"""
from __future__ import annotations

import colorsys
import ctypes
import ctypes.wintypes as wt
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# ── Win32 via ctypes ────────────────────────────────────────────────────
_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32
_kernel32 = ctypes.windll.kernel32

# Window style constants
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000

LWA_COLORKEY = 0x00000001
HWND_TOPMOST = -1
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

CS_HREDRAW = 0x0002
CS_VREDRAW = 0x0001

WM_PAINT = 0x000F
WM_DESTROY = 0x0002
WM_TIMER = 0x0113
WM_CLOSE = 0x0010
WM_ERASEBKGND = 0x0014
WM_USER = 0x0400
WM_APP_REDRAW = WM_USER + 1

IDC_ARROW = 32512
PS_SOLID = 0
NULL_BRUSH = 5
TRANSPARENT_BK = 1
DT_LEFT = 0x0000
DT_NOCLIP = 0x0100
FW_BOLD = 700

WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long, wt.HWND, ctypes.c_uint, wt.WPARAM, wt.LPARAM
)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("style", ctypes.c_uint),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wt.HINSTANCE),
        ("hIcon", wt.HICON),
        ("hCursor", wt.HANDLE),
        ("hbrBackground", wt.HBRUSH),
        ("lpszMenuName", wt.LPCWSTR),
        ("lpszClassName", wt.LPCWSTR),
        ("hIconSm", wt.HICON),
    ]


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", wt.HDC),
        ("fErase", wt.BOOL),
        ("rcPaint", wt.RECT),
        ("fRestore", wt.BOOL),
        ("fIncUpdate", wt.BOOL),
        ("rgbReserved", ctypes.c_byte * 32),
    ]


# ── Color key & palette ────────────────────────────────────────────────
# RGB(1,2,3) as COLORREF (0x00BBGGRR).  Every pixel filled with this
# colour becomes fully transparent via SetLayeredWindowAttributes.
_CK = 0x00030201

_cls_colors: Dict[str, int] = {}


def _colorref(r: int, g: int, b: int) -> int:
    return r | (g << 8) | (b << 16)


def _class_colorref(cls_name: str) -> int:
    """Vivid, deterministic colour for a YOLO class name."""
    if cls_name not in _cls_colors:
        h = hash(cls_name) % 360
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.9, 1.0)
        cr = _colorref(int(r * 255), int(g * 255), int(b * 255))
        if cr == _CK:
            cr = _colorref(255, 0, 0)
        _cls_colors[cls_name] = cr
    return _cls_colors[cls_name]


# ── Overlay class ──────────────────────────────────────────────────────

class YoloOverlay:
    """Transparent overlay window for real-time YOLO bbox visualisation."""

    def __init__(self, target_hwnd: int):
        self._target_hwnd = int(target_hwnd)
        self._overlay_hwnd: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._boxes: List[Tuple[str, float, float, float, float, float]] = []
        self._lock = threading.Lock()
        self._wndproc_ref = None          # prevent GC of callback
        self._tx = self._ty = 0
        self._tw = self._th = 0
        self._cls_name = f"YoloOverlay_{id(self)}"

    # ── Public API ──────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        hwnd = self._overlay_hwnd
        if hwnd and _user32.IsWindow(hwnd):
            _user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update(self, yolo_boxes: list) -> None:
        """Push new detection boxes — thread-safe."""
        parsed: List[Tuple[str, float, float, float, float, float]] = []
        for b in yolo_boxes:
            if hasattr(b, "cls_name"):
                parsed.append((b.cls_name, b.confidence,
                                b.x1, b.y1, b.x2, b.y2))
            elif isinstance(b, dict):
                parsed.append((
                    b.get("cls", ""), b.get("conf", 0),
                    b.get("x1", 0), b.get("y1", 0),
                    b.get("x2", 0), b.get("y2", 0),
                ))
        with self._lock:
            self._boxes = parsed
        hwnd = self._overlay_hwnd
        if hwnd and _user32.IsWindow(hwnd):
            _user32.InvalidateRect(hwnd, None, True)

    # ── Win32 internals ─────────────────────────────────────────────

    def _target_rect(self) -> Tuple[int, int, int, int]:
        """Screen-space client rect of target window."""
        try:
            from scripts.win_capture import get_client_rect_on_screen
            r = get_client_rect_on_screen(self._target_hwnd)
            return r.left, r.top, r.right - r.left, r.bottom - r.top
        except Exception:
            return 0, 0, 0, 0

    def _sync_position(self) -> None:
        x, y, w, h = self._target_rect()
        if w <= 0 or h <= 0:
            return
        if (x, y, w, h) != (self._tx, self._ty, self._tw, self._th):
            self._tx, self._ty, self._tw, self._th = x, y, w, h
            _user32.SetWindowPos(
                self._overlay_hwnd, HWND_TOPMOST,
                x, y, w, h, SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )
            _user32.InvalidateRect(self._overlay_hwnd, None, True)

    def _on_paint(self, hwnd: int) -> None:
        ps = PAINTSTRUCT()
        hdc = _user32.BeginPaint(hwnd, ctypes.byref(ps))

        rc = wt.RECT()
        _user32.GetClientRect(hwnd, ctypes.byref(rc))
        w, h = rc.right, rc.bottom

        # Fill entire client area with colour key → transparent
        ck_brush = _gdi32.CreateSolidBrush(_CK)
        _user32.FillRect(hdc, ctypes.byref(rc), ck_brush)
        _gdi32.DeleteObject(ck_brush)

        if w <= 0 or h <= 0:
            _user32.EndPaint(hwnd, ctypes.byref(ps))
            return

        with self._lock:
            boxes = list(self._boxes)

        _gdi32.SetBkMode(hdc, TRANSPARENT_BK)

        # Font for labels
        font_h = max(14, h // 50)
        font = _gdi32.CreateFontW(
            font_h, 0, 0, 0, FW_BOLD, 0, 0, 0,
            0, 0, 0, 0, 0, "Consolas",
        )
        old_font = _gdi32.SelectObject(hdc, font)

        null_brush = _gdi32.GetStockObject(NULL_BRUSH)

        for cls_name, conf, bx1, by1, bx2, by2 in boxes:
            px1, py1 = int(bx1 * w), int(by1 * h)
            px2, py2 = int(bx2 * w), int(by2 * h)
            color = _class_colorref(cls_name)

            # Bounding box
            pen = _gdi32.CreatePen(PS_SOLID, 3, color)
            old_pen = _gdi32.SelectObject(hdc, pen)
            old_br = _gdi32.SelectObject(hdc, null_brush)
            _gdi32.Rectangle(hdc, px1, py1, px2, py2)
            _gdi32.SelectObject(hdc, old_pen)
            _gdi32.SelectObject(hdc, old_br)
            _gdi32.DeleteObject(pen)

            # Label: background + text
            label = f"{cls_name} {conf:.2f}"
            lbl_w = len(label) * (font_h * 6 // 10) + 6
            lbl_h = font_h + 4
            lbl_rc = wt.RECT(px1, max(0, py1 - lbl_h), px1 + lbl_w, py1)
            bg = _gdi32.CreateSolidBrush(_colorref(30, 30, 30))
            _user32.FillRect(hdc, ctypes.byref(lbl_rc), bg)
            _gdi32.DeleteObject(bg)

            _gdi32.SetTextColor(hdc, color)
            txt_rc = wt.RECT(px1 + 3, max(0, py1 - lbl_h + 2),
                             px1 + lbl_w, py1)
            _user32.DrawTextW(hdc, label, -1,
                              ctypes.byref(txt_rc), DT_LEFT | DT_NOCLIP)

        _gdi32.SelectObject(hdc, old_font)
        _gdi32.DeleteObject(font)
        _user32.EndPaint(hwnd, ctypes.byref(ps))

    # ── Thread entry point ──────────────────────────────────────────

    def _run(self) -> None:
        # DPI awareness
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

        hinstance = _kernel32.GetModuleHandleW(None)

        def _wndproc(hwnd, msg, wp, lp):
            if msg == WM_PAINT:
                self._on_paint(hwnd)
                return 0
            if msg == WM_ERASEBKGND:
                return 1
            if msg == WM_TIMER:
                self._sync_position()
                return 0
            if msg == WM_CLOSE:
                _user32.KillTimer(hwnd, 1)
                _user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                _user32.PostQuitMessage(0)
                return 0
            return _user32.DefWindowProcW(hwnd, msg, wp, lp)

        self._wndproc_ref = WNDPROC(_wndproc)

        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.style = CS_HREDRAW | CS_VREDRAW
        wc.lpfnWndProc = self._wndproc_ref
        wc.hInstance = hinstance
        wc.hCursor = _user32.LoadCursorW(None, IDC_ARROW)
        wc.lpszClassName = self._cls_name

        atom = _user32.RegisterClassExW(ctypes.byref(wc))
        if not atom:
            print(f"[Overlay] RegisterClassExW failed: {ctypes.get_last_error()}")
            return

        x, y, w, h = self._target_rect()
        self._tx, self._ty, self._tw, self._th = x, y, w, h

        ex = (WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
              | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
        self._overlay_hwnd = _user32.CreateWindowExW(
            ex, self._cls_name, "YOLO Overlay",
            WS_POPUP | WS_VISIBLE,
            x, y, max(w, 100), max(h, 100),
            None, None, hinstance, None,
        )
        if not self._overlay_hwnd:
            print(f"[Overlay] CreateWindowExW failed: {ctypes.get_last_error()}")
            _user32.UnregisterClassW(self._cls_name, hinstance)
            return

        _user32.SetLayeredWindowAttributes(
            self._overlay_hwnd, _CK, 0, LWA_COLORKEY,
        )

        # Timer: re-sync overlay position every 200 ms
        _user32.SetTimer(self._overlay_hwnd, 1, 200, None)

        print(f"[Overlay] Started overlay={self._overlay_hwnd} "
              f"target={self._target_hwnd} size={w}x{h}")

        # Message pump (blocks until WM_QUIT)
        msg = wt.MSG()
        while _user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            _user32.TranslateMessage(ctypes.byref(msg))
            _user32.DispatchMessageW(ctypes.byref(msg))

        self._overlay_hwnd = None
        _user32.UnregisterClassW(self._cls_name, hinstance)
        print("[Overlay] Stopped.")


# ── Standalone test ────────────────────────────────────────────────────

def _test_main():
    import argparse, sys, os
    sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from scripts.win_capture import (
        find_window_by_title_substring, find_largest_visible_child,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="MuMu")
    args = parser.parse_args()

    hwnd = find_window_by_title_substring(args.title)
    if not hwnd:
        print(f"Window '{args.title}' not found")
        sys.exit(1)
    child = find_largest_visible_child(int(hwnd))
    render = int(child) if child else int(hwnd)
    print(f"Target: parent={hwnd} render={render}")

    overlay = YoloOverlay(render)
    overlay.start()
    time.sleep(0.5)

    # Feed YOLO detections from the model in a loop
    print("Running YOLO overlay (Ctrl+C to stop)...")
    try:
        from brain.pipeline import _run_yolo_on_image
        from scripts.win_capture import capture_client
        import numpy as np, cv2

        while True:
            pil = capture_client(render)
            if pil is None:
                time.sleep(0.1)
                continue
            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            yolo = _run_yolo_on_image(frame, w, h)
            overlay.update(yolo)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        overlay.stop()


if __name__ == "__main__":
    _test_main()
