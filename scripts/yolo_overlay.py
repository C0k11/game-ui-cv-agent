"""Transparent YOLO bounding-box overlay on game window.

Architecture (v2 — "cheat-grade" render pipeline):
  - Detection boxes are stored as **normalized 0-1 coordinates** relative
    to the game client area.  They are never converted to absolute screen
    coordinates until the instant of drawing.
  - A Win32 timer fires at ~240 Hz (4 ms).  Every tick it:
        1. Reads the game window's current screen position (GetWindowRect).
        2. Moves the overlay to match (SetWindowPos).
        3. Invalidates the overlay for a WM_PAINT.
    Because position is re-read every tick, the overlay "sticks" to the
    game window with zero perceptible lag — even while dragging.
  - Box data is pushed from the AI thread via ``update()`` (lock-free
    swap of a list reference).  The render loop never waits on AI.

Usage (standalone test — draws live YOLO boxes on MuMu):
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

_LRESULT = getattr(wt, "LRESULT", ctypes.c_ssize_t)

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
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004

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

# Timer interval in ms.  4 ms ≈ 250 Hz render refresh.
_TIMER_MS = 4

WNDPROC = ctypes.WINFUNCTYPE(
    _LRESULT, wt.HWND, ctypes.c_uint, wt.WPARAM, wt.LPARAM
)

_user32.DefWindowProcW.argtypes = [wt.HWND, ctypes.c_uint, wt.WPARAM, wt.LPARAM]
_user32.DefWindowProcW.restype = _LRESULT

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


def _is_florence_label(cls_name: str) -> bool:
    return str(cls_name).startswith("Florence:")


def _display_label(cls_name: str) -> str:
    raw = str(cls_name or "")
    if _is_florence_label(raw):
        return f"Florence | {raw.split(':', 1)[1].strip()}"
    return raw


def _class_colorref(cls_name: str) -> int:
    """Vivid, deterministic colour for a YOLO class name."""
    if _is_florence_label(str(cls_name)):
        return _colorref(127, 255, 0)
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
    """Transparent overlay window for real-time YOLO bbox visualisation.

    Render loop runs at ~250 Hz via Win32 timer.  Every tick:
      1. Query target window's screen rect (zero-cost Win32 call).
      2. SetWindowPos to keep overlay pixel-locked on top.
      3. InvalidateRect → WM_PAINT draws boxes using cached normalized
         coordinates multiplied by the *current* client size.

    Detection data is pushed from the AI thread via ``update()`` using a
    simple reference swap (no contention with the render thread).
    """

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
        # SORT tracker + velocity prediction + EMA smoothing
        # max_age=4: survive brief detection gaps
        # alpha=0.85: tight lock-on (85% new position + 15% old)
        try:
            from scripts.box_tracker import BoxTracker
            self._tracker = BoxTracker(max_age=5, min_hits=1, max_center_dist=1.5, alpha=0.85, high_conf=0.25)
        except Exception:
            self._tracker = None
        self._cls_name = f"YoloOverlay_{id(self)}"
        self._dirty = True                # force first paint

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
        """Push new detection boxes — thread-safe (lock-free swap).

        Runs SORT tracker + exponential smoothing so boxes glide
        instead of jumping between frames.
        """
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
        # Apply tracker for smooth lock-on effect
        if self._tracker is not None:
            parsed = self._tracker.update(parsed)
        with self._lock:
            self._boxes = parsed
            self._dirty = True

    # ── Win32 internals ─────────────────────────────────────────────

    def _target_rect(self) -> Tuple[int, int, int, int]:
        """Screen-space client rect of target window.

        Uses ClientToScreen for accurate per-monitor-DPI positioning.
        """
        try:
            from scripts.win_capture import get_client_rect_on_screen
            r = get_client_rect_on_screen(self._target_hwnd)
            return r.left, r.top, r.right - r.left, r.bottom - r.top
        except Exception:
            return 0, 0, 0, 0

    def _sync_position(self) -> None:
        """Reposition overlay to match target window — called at 250 Hz.

        Always queries the *current* window position so the overlay
        tracks window drags with sub-frame latency.  Only issues
        SetWindowPos when position or size actually changed (avoids
        unnecessary compositor work).
        """
        x, y, w, h = self._target_rect()
        if w <= 0 or h <= 0:
            return

        moved = (x != self._tx or y != self._ty)
        resized = (w != self._tw or h != self._th)

        if moved or resized:
            self._tx, self._ty, self._tw, self._th = x, y, w, h
            _user32.SetWindowPos(
                self._overlay_hwnd, HWND_TOPMOST,
                x, y, w, h, SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )
            # Size changed → must repaint (boxes scale with client area)
            self._dirty = True

        # Always repaint every tick for persistent lock-on visual.
        # Boxes are drawn fresh each frame and cleared next frame,
        # giving a continuous "locked on" effect at 250Hz.
        _user32.InvalidateRect(self._overlay_hwnd, None, False)

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

        # Snapshot current boxes (reference read is atomic in CPython)
        boxes = self._boxes

        _gdi32.SetBkMode(hdc, TRANSPARENT_BK)

        # Font for labels
        font_h = max(14, h // 50)
        font = _gdi32.CreateFontW(
            font_h, 0, 0, 0, FW_BOLD, 0, 0, 0,
            0, 0, 0, 0, 0, "Consolas",
        )
        old_font = _gdi32.SelectObject(hdc, font)

        null_brush = _gdi32.GetStockObject(NULL_BRUSH)

        def _draw_line(x1: int, y1: int, x2: int, y2: int) -> None:
            pt = wt.POINT()
            _gdi32.MoveToEx(hdc, int(x1), int(y1), ctypes.byref(pt))
            _gdi32.LineTo(hdc, int(x2), int(y2))

        for cls_name, conf, bx1, by1, bx2, by2 in boxes:
            px1, py1 = int(bx1 * w), int(by1 * h)
            px2, py2 = int(bx2 * w), int(by2 * h)
            color = _class_colorref(cls_name)
            label = _display_label(cls_name)
            is_florence = _is_florence_label(cls_name)

            pen = _gdi32.CreatePen(PS_SOLID, 3 if is_florence else 3, color)
            old_pen = _gdi32.SelectObject(hdc, pen)
            old_br = _gdi32.SelectObject(hdc, null_brush)
            if is_florence:
                bw = max(1, px2 - px1)
                bh = max(1, py2 - py1)
                seg = max(12, min(28, bw // 4, bh // 4))
                _draw_line(px1, py1, px1 + seg, py1)
                _draw_line(px1, py1, px1, py1 + seg)
                _draw_line(px2, py1, px2 - seg, py1)
                _draw_line(px2, py1, px2, py1 + seg)
                _draw_line(px1, py2, px1 + seg, py2)
                _draw_line(px1, py2, px1, py2 - seg)
                _draw_line(px2, py2, px2 - seg, py2)
                _draw_line(px2, py2, px2, py2 - seg)
            else:
                _gdi32.Rectangle(hdc, px1, py1, px2, py2)
            _gdi32.SelectObject(hdc, old_pen)
            _gdi32.SelectObject(hdc, old_br)
            _gdi32.DeleteObject(pen)

            label = f"{label} {conf:.2f}"
            lbl_w = len(label) * (font_h * 6 // 10) + 6
            lbl_h = font_h + 4
            lbl_top = max(0, py1 - lbl_h - (2 if is_florence else 0))
            lbl_bottom = lbl_top + lbl_h
            lbl_rc = wt.RECT(px1, lbl_top, px1 + lbl_w, lbl_bottom)
            bg_color = _colorref(12, 24, 12) if is_florence else _colorref(30, 30, 30)
            bg = _gdi32.CreateSolidBrush(bg_color)
            _user32.FillRect(hdc, ctypes.byref(lbl_rc), bg)
            _gdi32.DeleteObject(bg)

            if is_florence:
                accent_rc = wt.RECT(px1, lbl_top, px1 + lbl_w, min(lbl_bottom, lbl_top + 3))
                accent = _gdi32.CreateSolidBrush(color)
                _user32.FillRect(hdc, ctypes.byref(accent_rc), accent)
                _gdi32.DeleteObject(accent)

            _gdi32.SetTextColor(hdc, color)
            txt_rc = wt.RECT(px1 + 3, lbl_top + 2,
                             px1 + lbl_w, lbl_bottom)
            _user32.DrawTextW(hdc, label, -1,
                              ctypes.byref(txt_rc), DT_LEFT | DT_NOCLIP)

        _gdi32.SelectObject(hdc, old_font)
        _gdi32.DeleteObject(font)
        _user32.EndPaint(hwnd, ctypes.byref(ps))

    # ── Thread entry point ──────────────────────────────────────────

    def _run(self) -> None:
        # DPI awareness (per-monitor v2 for accurate positioning)
        try:
            _user32.SetThreadDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
            _user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
            _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
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
            if msg == WM_APP_REDRAW:
                self._dirty = True
                return 0
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

        # 250 Hz timer — drives position sync + repaint
        _user32.SetTimer(self._overlay_hwnd, 1, _TIMER_MS, None)

        print(f"[Overlay] Started overlay={self._overlay_hwnd} "
              f"target={self._target_hwnd} size={w}x{h} timer={_TIMER_MS}ms")

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
