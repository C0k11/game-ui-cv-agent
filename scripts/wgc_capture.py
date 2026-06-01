"""WGC (Windows.Graphics.Capture) screen-capture backend.

Captures a target window (e.g. the MuMu emulator render window) at high frame
rate even when the window is occluded or in the background, using the
``windows-capture`` library (Windows.Graphics.Capture API).

Why this exists
---------------
``scripts/win_capture.py`` offers PrintWindow / BitBlt / desktop BitBlt modes.
Their ceiling is ~32 fps (PrintWindow) and BitBlt returns black for GPU-rendered
windows on a secondary monitor. WGC hardware-composites the window and delivers
50+ fps regardless of occlusion.

API notes (windows_capture 2.0.0)
---------------------------------
* ``WindowsCapture(window_hwnd=..., window_name=..., monitor_index=...)`` selects
  the target. ``window_hwnd`` is the most reliable (no CJK-title issues) and the
  captured surface auto-follows the window. WGC captures the *whole window*
  (including title bar / borders), so the frame is slightly larger than the
  client area and must be cropped.
* The capture loop is blocking: ``cap.start()`` runs until stopped. We run it on
  a daemon thread. ``cap.event`` registers ``on_frame_arrived(frame, ctrl)`` and
  ``on_closed()``. To stop the blocking loop we call ``ctrl.stop()`` from inside
  the callback (there is no external stop on ``WindowsCapture`` itself for the
  ``start()`` path).
* ``frame.frame_buffer`` is a ``numpy.ndarray`` of shape ``(height, width, 4)``,
  dtype ``uint8``, channel order **BGRA**. ``frame.frame_buffer[:, :, :3]`` is
  the BGR image. The array aliases native memory, so we copy it.

This module only *adds* a backend; it does not touch the existing win_capture
modes.
"""
from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from typing import Optional, Tuple

import numpy as np

try:
    from windows_capture import WindowsCapture
    _WGC_AVAILABLE = True
    _WGC_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover - import guard
    WindowsCapture = None  # type: ignore
    _WGC_AVAILABLE = False
    _WGC_IMPORT_ERROR = _e

# Reuse the screen-coordinate / DPI helpers from the existing capture module
# (read-only use; we do not modify win_capture).
from scripts.win_capture import get_client_rect_on_screen, _dpi_aware_context

_user32 = ctypes.WinDLL("user32", use_last_error=True)


def wgc_available() -> bool:
    """True if the windows_capture backend imported successfully."""
    return _WGC_AVAILABLE


# ── monitor enumeration (for monitor-index fallback crop) ─────────────────

_MONITORENUMPROC = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
    ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
)


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


def _enum_monitor_rects() -> list[Tuple[int, int, int, int]]:
    """Return monitor rects (l, t, r, b) in EnumDisplayMonitors order.

    This order matches the ``monitor_index`` used by windows_capture in
    practice (verified against MuMu on a multi-monitor setup).
    """
    rects: list[Tuple[int, int, int, int]] = []

    def _cb(hmon, hdc, lprc, lp):  # noqa: ANN001
        mi = _MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(_MONITORINFOEXW)
        try:
            _user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
            rm = mi.rcMonitor
            rects.append((int(rm.left), int(rm.top), int(rm.right), int(rm.bottom)))
        except Exception:
            pass
        return True

    try:
        with _dpi_aware_context():
            _user32.EnumDisplayMonitors(0, None, _MONITORENUMPROC(_cb), 0)
    except Exception:
        return []
    return rects


def _window_rect_on_screen(hwnd: int) -> Tuple[int, int, int, int]:
    """GetWindowRect in physical pixels (l, t, r, b)."""
    with _dpi_aware_context():
        wr = wintypes.RECT()
        if not _user32.GetWindowRect(int(hwnd), ctypes.byref(wr)):
            raise OSError(ctypes.get_last_error())
    return int(wr.left), int(wr.top), int(wr.right), int(wr.bottom)


def monitor_index_for_window(hwnd: int) -> Optional[int]:
    """Best-effort: which monitor_index the window's center sits on."""
    try:
        rc = get_client_rect_on_screen(int(hwnd))
        cx = (rc.left + rc.right) // 2
        cy = (rc.top + rc.bottom) // 2
    except Exception:
        return None
    for i, (l, t, r, b) in enumerate(_enum_monitor_rects()):
        if l <= cx < r and t <= cy < b:
            return i
    return None


# ── the capture backend ───────────────────────────────────────────────────


class WgcCapture:
    """Threaded Windows.Graphics.Capture backend.

    A daemon thread runs the blocking WGC loop; each arriving frame is stored
    (BGRA, copied) under a lock. ``grab()`` synchronously returns the latest
    frame cropped to the MuMu client area as a BGR ``numpy.ndarray``.

    Parameters
    ----------
    render_hwnd:
        The MuMu render child window. Used to compute the client-area crop
        (its client rect on screen). When ``capture_hwnd`` is not given, this
        is also tried as the capture target first, then its top-level parent.
    capture_hwnd:
        Explicit window to hand to WGC. WGC cannot always build a
        GraphicsCaptureItem from a deep child window; the top-level MuMu window
        usually works. If omitted, the constructor auto-resolves a working
        target (render_hwnd -> top-level ancestor -> monitor).
    monitor_index:
        If set (and no hwnd target works), capture this monitor full-screen and
        crop out the MuMu client area by screen coordinates. ``None`` = auto.
    client_size:
        Optional ``(w, h)`` to force the crop size. If omitted it is read from
        ``render_hwnd``'s client rect and refreshed when it changes.
    """

    def __init__(
        self,
        render_hwnd: int,
        *,
        capture_hwnd: Optional[int] = None,
        monitor_index: Optional[int] = None,
        client_size: Optional[Tuple[int, int]] = None,
        cursor_capture: bool = False,
        draw_border: bool = False,
    ) -> None:
        if not _WGC_AVAILABLE:
            raise RuntimeError(
                f"windows_capture is not available: {_WGC_IMPORT_ERROR!r}"
            )

        self.render_hwnd = int(render_hwnd)
        self._cursor_capture = bool(cursor_capture)
        self._draw_border = bool(draw_border)

        # crop target (client area)
        if client_size is not None:
            self._client_w, self._client_h = int(client_size[0]), int(client_size[1])
        else:
            self._client_w, self._client_h = self._read_client_size()

        # latest frame state
        self._lock = threading.Lock()
        self._latest_bgra: Optional[np.ndarray] = None  # full WGC frame (BGRA)
        self._frame_count = 0
        self._stop = threading.Event()
        self._started = threading.Event()
        self._closed = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None
        self._start_error: Optional[BaseException] = None

        # capture geometry: either ("window", hwnd) or ("monitor", index)
        self._mode: str = "window"
        self._target_hwnd: Optional[int] = None
        self._monitor_index: Optional[int] = None
        self._monitor_origin: Tuple[int, int] = (0, 0)

        # cached crop geometry (refreshed lazily; Win32 rect calls are costly
        # relative to the >50fps grab rate and geometry rarely changes)
        self._crop_cache: Tuple[int, int] = (0, 0)
        self._crop_cache_ts: float = 0.0
        self._geom_refresh_interval: float = 0.5

        self._resolve_target(capture_hwnd, monitor_index)
        self._start_thread()

    # ── geometry helpers ──────────────────────────────────────────────────

    def _read_client_size(self) -> Tuple[int, int]:
        try:
            r = get_client_rect_on_screen(self.render_hwnd)
            return max(0, r.width), max(0, r.height)
        except Exception:
            return 0, 0

    @staticmethod
    def _top_level_ancestor(hwnd: int) -> int:
        GA_ROOT = 2
        try:
            top = int(_user32.GetAncestor(int(hwnd), GA_ROOT) or 0)
            return top if top else int(hwnd)
        except Exception:
            return int(hwnd)

    def _resolve_target(
        self, capture_hwnd: Optional[int], monitor_index: Optional[int]
    ) -> None:
        """Pick a WGC target that can actually be captured.

        Order: explicit capture_hwnd -> render_hwnd -> top-level parent ->
        monitor (explicit or auto-detected from the window position).
        """
        candidates: list[int] = []
        if capture_hwnd:
            candidates.append(int(capture_hwnd))
        candidates.append(self.render_hwnd)
        top = self._top_level_ancestor(self.render_hwnd)
        if top not in candidates:
            candidates.append(top)

        for hwnd in candidates:
            if self._probe_window_target(hwnd):
                self._mode = "window"
                self._target_hwnd = int(hwnd)
                return

        # window targets all failed -> monitor fallback
        mi = monitor_index
        if mi is None:
            mi = monitor_index_for_window(self.render_hwnd)
        if mi is None:
            mi = 0
        self._mode = "monitor"
        self._monitor_index = int(mi)
        rects = _enum_monitor_rects()
        if 0 <= self._monitor_index < len(rects):
            l, t, _, _ = rects[self._monitor_index]
            self._monitor_origin = (l, t)
        else:
            self._monitor_origin = (0, 0)

    def _probe_window_target(self, hwnd: int) -> bool:
        """Briefly try to start WGC on a window hwnd; True if a frame arrives."""
        if not hwnd:
            return False
        got = {"ok": False}
        ev = threading.Event()
        err: dict = {"e": None}
        try:
            cap = WindowsCapture(
                cursor_capture=self._cursor_capture,
                draw_border=self._draw_border,
                window_hwnd=int(hwnd),
            )
        except Exception as e:
            return False

        @cap.event
        def on_frame_arrived(frame, ctrl):  # noqa: ANN001
            got["ok"] = True
            ev.set()
            ctrl.stop()

        @cap.event
        def on_closed():  # noqa: ANN001
            ev.set()

        def run():
            try:
                cap.start()
            except Exception as e:  # GraphicsCaptureItem conversion failure, etc.
                err["e"] = e
                ev.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        ev.wait(timeout=1.5)
        t.join(timeout=1.0)
        return bool(got["ok"]) and err["e"] is None

    def _crop_offset(self) -> Tuple[int, int]:
        """(x, y) offset of the client area inside the current WGC frame."""
        try:
            rc = get_client_rect_on_screen(self.render_hwnd)
        except Exception:
            return 0, 0
        if self._mode == "window" and self._target_hwnd is not None:
            try:
                wl, wt, _, _ = _window_rect_on_screen(self._target_hwnd)
            except Exception:
                return 0, 0
            return rc.left - wl, rc.top - wt
        # monitor mode
        mox, moy = self._monitor_origin
        return rc.left - mox, rc.top - moy

    # ── capture thread ────────────────────────────────────────────────────

    def _build_capture(self):
        if self._mode == "window":
            return WindowsCapture(
                cursor_capture=self._cursor_capture,
                draw_border=self._draw_border,
                window_hwnd=int(self._target_hwnd),
            )
        return WindowsCapture(
            cursor_capture=self._cursor_capture,
            draw_border=self._draw_border,
            monitor_index=int(self._monitor_index or 0),
        )

    def _start_thread(self) -> None:
        cap = self._build_capture()
        self._cap = cap

        @cap.event
        def on_frame_arrived(frame, ctrl):  # noqa: ANN001
            # frame.frame_buffer: (h, w, 4) uint8 BGRA, aliases native memory.
            try:
                buf = frame.frame_buffer
                with self._lock:
                    self._latest_bgra = buf.copy()
                    self._frame_count += 1
            except Exception:
                pass
            if self._stop.is_set():
                try:
                    ctrl.stop()
                except Exception:
                    pass

        @cap.event
        def on_closed():  # noqa: ANN001
            self._closed.set()

        def run():
            self._started.set()
            try:
                cap.start()
            except Exception as e:
                self._start_error = e
            finally:
                self._closed.set()

        self._thread = threading.Thread(
            target=run, name="WgcCapture", daemon=True
        )
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """'window' or 'monitor' — how WGC is sourcing pixels."""
        return self._mode

    @property
    def client_size(self) -> Tuple[int, int]:
        return self._client_w, self._client_h

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    def wait_first_frame(self, timeout: float = 3.0) -> bool:
        """Block until at least one frame has been captured (or timeout)."""
        import time as _time
        deadline = _time.time() + max(0.0, timeout)
        while _time.time() < deadline:
            with self._lock:
                if self._latest_bgra is not None:
                    return True
            if self._start_error is not None:
                return False
            _time.sleep(0.01)
        with self._lock:
            return self._latest_bgra is not None

    def _refresh_geometry(self) -> None:
        """Re-read client size + crop offset (throttled). Cheap calls are
        skipped between refreshes to keep grab() at full stream rate."""
        import time as _time
        now = _time.time()
        if now - self._crop_cache_ts < self._geom_refresh_interval:
            return
        self._crop_cache_ts = now
        cw, ch = self._read_client_size()
        if cw > 0 and ch > 0:
            self._client_w, self._client_h = cw, ch
        self._crop_cache = self._crop_offset()

    def grab(self) -> Optional[np.ndarray]:
        """Return the latest frame as BGR ``numpy.ndarray`` cropped to the
        MuMu client area. ``None`` if no frame yet."""
        with self._lock:
            frame = self._latest_bgra
        if frame is None:
            return None

        fh, fw = frame.shape[0], frame.shape[1]

        self._refresh_geometry()
        ox, oy = self._crop_cache

        # If we somehow have no valid client size, return full BGR frame.
        if self._client_w <= 0 or self._client_h <= 0:
            return np.ascontiguousarray(frame[:, :, :3])

        x0 = min(max(0, ox), max(0, fw - 1))
        y0 = min(max(0, oy), max(0, fh - 1))
        x1 = min(fw, x0 + self._client_w)
        y1 = min(fh, y0 + self._client_h)
        crop = frame[y0:y1, x0:x1, :3]
        return np.ascontiguousarray(crop)

    def start_error(self) -> Optional[BaseException]:
        """Exception raised by the WGC start loop, if any."""
        return self._start_error

    def is_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def stop(self) -> None:
        """Signal the capture loop to stop and join the thread."""
        self._stop.set()
        # The loop only checks the stop flag on the next frame; if frames have
        # stopped arriving (e.g. window closed) it will exit on its own.
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
