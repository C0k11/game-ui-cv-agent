import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from PIL import Image


user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)


def _ensure_dpi_awareness() -> None:
    try:
        user32.SetProcessDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
        user32.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = (DPI_AWARENESS_CONTEXT)-4
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:
        pass
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.SetProcessDpiAwareness.argtypes = (ctypes.c_int,)
        shcore.SetProcessDpiAwareness.restype = ctypes.c_int
        shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        user32.SetProcessDPIAware.argtypes = ()
        user32.SetProcessDPIAware.restype = ctypes.c_bool
        user32.SetProcessDPIAware()
    except Exception:
        pass


_ensure_dpi_awareness()


EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


EnumChildProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


@dataclass
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


def _get_window_text(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _is_window_visible(hwnd: int) -> bool:
    return bool(user32.IsWindowVisible(hwnd))


def _client_area(hwnd: int) -> int:
    try:
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return 0
        w = int(rect.right - rect.left)
        h = int(rect.bottom - rect.top)
        return max(0, w) * max(0, h)
    except Exception:
        return 0


def find_largest_visible_child(hwnd_parent: int) -> Optional[int]:
    best_hwnd: Optional[int] = None
    best_area = 0

    def _cb(hwnd: int, lparam: int) -> bool:
        nonlocal best_hwnd, best_area
        try:
            if not _is_window_visible(hwnd):
                return True
            area = _client_area(hwnd)
            if area > best_area:
                best_area = area
                best_hwnd = int(hwnd)
        except Exception:
            pass
        return True

    try:
        user32.EnumChildWindows(int(hwnd_parent), EnumChildProc(_cb), 0)
    except Exception:
        return None
    return best_hwnd


def find_window_by_title_substring(title_substring: str) -> Optional[int]:
    title_substring = (title_substring or "").strip().lower()
    if not title_substring:
        return None

    found: Optional[int] = None

    def _cb(hwnd: int, lparam: int) -> bool:
        nonlocal found
        if found is not None:
            return False
        if not _is_window_visible(hwnd):
            return True
        title = _get_window_text(hwnd).strip()
        if not title:
            return True
        if title_substring in title.lower():
            found = hwnd
            return False
        return True

    user32.EnumWindows(EnumWindowsProc(_cb), 0)
    return found


def get_client_rect_on_screen(hwnd: int) -> Rect:
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise OSError(ctypes.get_last_error())

    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        raise OSError(ctypes.get_last_error())

    left = int(pt.x)
    top = int(pt.y)
    right = left + int(rect.right - rect.left)
    bottom = top + int(rect.bottom - rect.top)
    return Rect(left=left, top=top, right=right, bottom=bottom)


def capture_client(hwnd: int) -> Image.Image:
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise OSError(ctypes.get_last_error())

    w = int(rect.right - rect.left)
    h = int(rect.bottom - rect.top)
    if w <= 0 or h <= 0:
        raise ValueError("window client rect is empty")

    hdc_screen = user32.GetDC(hwnd)
    if not hdc_screen:
        raise OSError(ctypes.get_last_error())

    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    if not hdc_mem:
        user32.ReleaseDC(hwnd, hdc_screen)
        raise OSError(ctypes.get_last_error())

    hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
    if not hbmp:
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_screen)
        raise OSError(ctypes.get_last_error())

    old = gdi32.SelectObject(hdc_mem, hbmp)
    SRCCOPY = 0x00CC0020

    if not gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, 0, 0, SRCCOPY):
        gdi32.SelectObject(hdc_mem, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_screen)
        raise OSError(ctypes.get_last_error())

    bmi = ctypes.create_string_buffer(40)
    ctypes.memset(bmi, 0, 40)
    ctypes.cast(bmi, ctypes.POINTER(ctypes.c_uint32))[0] = 40
    ctypes.cast(bmi, ctypes.POINTER(ctypes.c_int32))[1] = w
    ctypes.cast(bmi, ctypes.POINTER(ctypes.c_int32))[2] = -h
    ctypes.cast(bmi, ctypes.POINTER(ctypes.c_uint16))[6] = 1
    ctypes.cast(bmi, ctypes.POINTER(ctypes.c_uint16))[7] = 32

    buf = ctypes.create_string_buffer(w * h * 4)
    bits = gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buf, bmi, 0)
    if bits == 0:
        gdi32.SelectObject(hdc_mem, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_screen)
        raise OSError(ctypes.get_last_error())

    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")

    gdi32.SelectObject(hdc_mem, old)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(hwnd, hdc_screen)

    return img
