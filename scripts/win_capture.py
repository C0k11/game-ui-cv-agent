import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from PIL import Image


user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

try:
    user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetClassNameW.restype = ctypes.c_int
except Exception:
    pass

try:
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
except Exception:
    pass

try:
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
except Exception:
    pass

try:
    kernel32.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
except Exception:
    pass


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


def _get_class_name(hwnd: int) -> str:
    try:
        buf = ctypes.create_unicode_buffer(256)
        if int(user32.GetClassNameW(int(hwnd), buf, 256) or 0) > 0:
            return str(buf.value or "")
    except Exception:
        pass
    return ""


def _get_window_pid(hwnd: int) -> int:
    try:
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(wintypes.HWND(int(hwnd)), ctypes.byref(pid))
        return int(pid.value)
    except Exception:
        return 0


def _process_basename(pid: int) -> str:
    pid = int(pid)
    if pid <= 0:
        return ""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = None
    try:
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid))
    except Exception:
        h = None
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(2048)
        size = wintypes.DWORD(len(buf))
        try:
            kernel32.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
            kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        except Exception:
            pass
        ok = bool(kernel32.QueryFullProcessImageNameW(wintypes.HANDLE(h), wintypes.DWORD(0), buf, ctypes.byref(size)))
        if not ok:
            return ""
        p = str(buf.value or "")
        base = p.replace("/", "\\").rsplit("\\", 1)[-1]
        return str(base or "")
    except Exception:
        return ""
    finally:
        try:
            kernel32.CloseHandle(wintypes.HANDLE(h))
        except Exception:
            pass


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

    bad_exes = {
        "chrome.exe",
        "msedge.exe",
        "firefox.exe",
        "brave.exe",
        "opera.exe",
        "iexplore.exe",
        "steam.exe",
        "steamwebhelper.exe",
    }

    bad_classes = {
        "chrome_widgetwin_0",
        "chrome_widgetwin_1",
        "mozillawindowclass",
        "applicationframewindow",
    }

    candidates: list[tuple[int, int, int]] = []

    def _cb(hwnd: int, lparam: int) -> bool:
        if not _is_window_visible(hwnd):
            return True
        title = _get_window_text(hwnd).strip()
        if not title:
            return True
        if title_substring not in title.lower():
            return True

        try:
            cls = _get_class_name(int(hwnd)).lower().strip()
        except Exception:
            cls = ""
        if cls in bad_classes:
            return True

        pid = _get_window_pid(int(hwnd))
        exe = _process_basename(int(pid)).lower().strip()
        if exe in bad_exes:
            return True

        try:
            area = int(_client_area(int(hwnd)))
        except Exception:
            area = 0

        iconic = 0
        try:
            iconic = 1 if bool(user32.IsIconic(int(hwnd))) else 0
        except Exception:
            iconic = 0

        candidates.append((int(iconic), -int(area), int(hwnd)))
        return True

    try:
        user32.EnumWindows(EnumWindowsProc(_cb), 0)
    except Exception:
        return None

    if not candidates:
        print(f"[DEBUG] find_window_by_title_substring('{title_substring}'): No candidates found.", file=sys.stderr)
        return None

    candidates.sort()
    # Debug logging for window selection
    try:
        print(f"[DEBUG] find_window_by_title_substring('{title_substring}') candidates:", file=sys.stderr)
        for _, neg_area, h in candidates:
            t = _get_window_text(h)
            p = _process_basename(_get_window_pid(h))
            print(f"  - hwnd={h} area={-neg_area} exe={p} title='{t}'", file=sys.stderr)
    except Exception:
        pass

    return int(candidates[0][2])


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
    def _capture_window_dc() -> Image.Image:
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

    def _capture_desktop_dc() -> Image.Image:
        r = get_client_rect_on_screen(hwnd)
        w = int(r.right - r.left)
        h = int(r.bottom - r.top)
        if w <= 0 or h <= 0:
            raise ValueError("window client rect is empty")

        hdc_screen = user32.GetDC(0)
        if not hdc_screen:
            raise OSError(ctypes.get_last_error())

        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        if not hdc_mem:
            user32.ReleaseDC(0, hdc_screen)
            raise OSError(ctypes.get_last_error())

        hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        if not hbmp:
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(0, hdc_screen)
            raise OSError(ctypes.get_last_error())

        old = gdi32.SelectObject(hdc_mem, hbmp)
        SRCCOPY = 0x00CC0020

        if not gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, int(r.left), int(r.top), SRCCOPY):
            gdi32.SelectObject(hdc_mem, old)
            gdi32.DeleteObject(hbmp)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(0, hdc_screen)
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
            user32.ReleaseDC(0, hdc_screen)
            raise OSError(ctypes.get_last_error())

        img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")

        gdi32.SelectObject(hdc_mem, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)

        return img

    def _is_nearly_black(im: Image.Image) -> bool:
        try:
            sim = im.resize((32, 32), resample=Image.BILINEAR).convert("RGB")
            px = sim.getdata()
            tot = 0
            n = 0
            for r, g, b in px:
                tot += int(r) + int(g) + int(b)
                n += 1
            if n <= 0:
                return False
            avg = tot / float(n * 3)
            return avg <= 8.0
        except Exception:
            return False

    def _sig(im: Image.Image) -> bytes:
        try:
            return im.resize((32, 32), resample=Image.BILINEAR).convert("RGB").tobytes()
        except Exception:
            return b""

    try:
        mode = (os.environ.get("WIN_CAPTURE_MODE") or "auto").strip().lower()
    except Exception:
        mode = "auto"

    if mode == "desktop":
        return _capture_desktop_dc()
    if mode == "window":
        return _capture_window_dc()

    img0 = _capture_window_dc()
    try:
        img1 = _capture_desktop_dc()
        if not _is_nearly_black(img1):
            if _is_nearly_black(img0):
                return img1
            s0 = _sig(img0)
            s1 = _sig(img1)
            if s0 and s1 and s0 != s1:
                return img1
    except Exception:
        pass
    return img0
