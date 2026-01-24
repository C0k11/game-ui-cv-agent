import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scripts.win_capture import capture_client, find_largest_visible_child, find_window_by_title_substring, get_client_rect_on_screen


user32 = ctypes.WinDLL("user32", use_last_error=True)


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


ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_ushort), ("wParamH", ctypes.c_ushort)]


class INPUT_I(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", INPUT_I)]


user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = ctypes.c_uint
user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
user32.GetSystemMetrics.restype = ctypes.c_int

user32.IsWindow.argtypes = (ctypes.c_void_p,)
user32.IsWindow.restype = ctypes.c_bool


INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

KEYEVENTF_KEYUP = 0x0002


def _send_inputs(inputs: list[INPUT]) -> None:
    if not inputs:
        return
    arr = (INPUT * len(inputs))(*inputs)
    sent = int(user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT)))
    if sent != len(inputs):
        raise ctypes.WinError(ctypes.get_last_error())


def _to_abs(sx: int, sy: int) -> tuple[int, int]:
    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77
    SM_CXVIRTUALSCREEN = 78
    SM_CYVIRTUALSCREEN = 79
    vx = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
    vy = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
    vw = int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
    vh = int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    if vw <= 0 or vh <= 0:
        vw = int(user32.GetSystemMetrics(0))
        vh = int(user32.GetSystemMetrics(1))
        vx, vy = 0, 0
    vw = max(1, vw)
    vh = max(1, vh)
    x = int(max(vx, min(vx + vw - 1, int(sx))))
    y = int(max(vy, min(vy + vh - 1, int(sy))))
    dx = int((x - vx) * 65535 / max(1, vw - 1))
    dy = int((y - vy) * 65535 / max(1, vh - 1))
    return dx, dy


@dataclass
class WindowTarget:
    title_substring: str
    hwnd: int
    render_hwnd: int


class WindowsInput:
    def __init__(self, *, title_substring: str):
        self.title_substring = (title_substring or "").strip()
        self._target: Optional[WindowTarget] = None

    def _resolve(self) -> WindowTarget:
        if self._target is not None:
            try:
                if user32.IsWindow(ctypes.c_void_p(int(self._target.hwnd))) and user32.IsWindow(ctypes.c_void_p(int(self._target.render_hwnd))):
                    return self._target
            except Exception:
                pass
            self._target = None
        hwnd = find_window_by_title_substring(self.title_substring)
        if hwnd is None:
            raise RuntimeError(f"window not found: {self.title_substring}")
        rh = None
        try:
            rh = find_largest_visible_child(int(hwnd))
        except Exception:
            rh = None
        render_hwnd = int(rh) if rh else int(hwnd)
        self._target = WindowTarget(title_substring=self.title_substring, hwnd=int(hwnd), render_hwnd=render_hwnd)
        return self._target

    def _focus(self) -> int:
        t = self._resolve()
        try:
            user32.SetForegroundWindow(ctypes.c_void_p(t.hwnd))
        except Exception:
            pass
        return t.hwnd

    def screenshot_client(self, out_path: str) -> None:
        last = None
        for _ in range(2):
            t = self._resolve()
            try:
                img = capture_client(t.render_hwnd)
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path)
                return
            except OSError as e:
                last = e
                try:
                    if getattr(e, "winerror", None) == 1400:
                        self._target = None
                        continue
                except Exception:
                    pass
                raise
            except Exception as e:
                last = e
                self._target = None
        if last is not None:
            raise last

    def client_size(self) -> tuple[int, int]:
        last = None
        for _ in range(2):
            t = self._resolve()
            try:
                r = get_client_rect_on_screen(t.render_hwnd)
                w = max(0, int(r.right - r.left))
                h = max(0, int(r.bottom - r.top))
                return int(w), int(h)
            except OSError as e:
                last = e
                try:
                    if getattr(e, "winerror", None) == 1400:
                        self._target = None
                        continue
                except Exception:
                    pass
                raise
            except Exception as e:
                last = e
                self._target = None
        if last is not None:
            raise last
        return 0, 0

    def _client_to_screen(self, x: int, y: int) -> tuple[int, int]:
        last = None
        for _ in range(2):
            t = self._resolve()
            try:
                r = get_client_rect_on_screen(t.render_hwnd)
                w = max(0, int(r.right - r.left))
                h = max(0, int(r.bottom - r.top))
                if w > 0:
                    x = int(max(0, min(w - 1, int(x))))
                if h > 0:
                    y = int(max(0, min(h - 1, int(y))))
                return (int(r.left + x), int(r.top + y))
            except OSError as e:
                last = e
                try:
                    if getattr(e, "winerror", None) == 1400:
                        self._target = None
                        continue
                except Exception:
                    pass
                raise
            except Exception as e:
                last = e
                self._target = None
        if last is not None:
            raise last
        return (int(x), int(y))

    def client_to_screen(self, x: int, y: int) -> tuple[int, int]:
        return self._client_to_screen(int(x), int(y))

    def click_client(self, x: int, y: int) -> None:
        self._focus()
        sx, sy = self._client_to_screen(int(x), int(y))
        try:
            user32.SetCursorPos(int(sx), int(sy))
        except Exception:
            dx, dy = _to_abs(int(sx), int(sy))
            _send_inputs(
                [
                    INPUT(
                        type=INPUT_MOUSE,
                        ii=INPUT_I(
                            mi=MOUSEINPUT(
                                dx=dx,
                                dy=dy,
                                mouseData=0,
                                dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                                time=0,
                                dwExtraInfo=0,
                            )
                        ),
                    ),
                ]
            )
        _send_inputs(
            [
                INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=0))),
            ]
        )

    def swipe_client(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        self._focus()
        sx1, sy1 = self._client_to_screen(int(x1), int(y1))
        sx2, sy2 = self._client_to_screen(int(x2), int(y2))

        try:
            user32.SetCursorPos(int(sx1), int(sy1))
        except Exception:
            dx1, dy1 = _to_abs(int(sx1), int(sy1))
            _send_inputs(
                [
                    INPUT(
                        type=INPUT_MOUSE,
                        ii=INPUT_I(
                            mi=MOUSEINPUT(
                                dx=dx1,
                                dy=dy1,
                                mouseData=0,
                                dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                                time=0,
                                dwExtraInfo=0,
                            )
                        ),
                    ),
                ]
            )

        _send_inputs(
            [
                INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=0))),
            ]
        )

        steps = 20
        dur = max(1, int(duration_ms))
        for i in range(1, steps + 1):
            t = i / steps
            sx = int(sx1 + (sx2 - sx1) * t)
            sy = int(sy1 + (sy2 - sy1) * t)
            try:
                user32.SetCursorPos(int(sx), int(sy))
            except Exception:
                dx, dy = _to_abs(int(sx), int(sy))
                _send_inputs(
                    [
                        INPUT(
                            type=INPUT_MOUSE,
                            ii=INPUT_I(
                                mi=MOUSEINPUT(
                                    dx=dx,
                                    dy=dy,
                                    mouseData=0,
                                    dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                                    time=0,
                                    dwExtraInfo=0,
                                )
                            ),
                        ),
                    ]
                )
            time.sleep(dur / 1000.0 / steps)

        try:
            user32.SetCursorPos(int(sx2), int(sy2))
        except Exception:
            pass
        _send_inputs(
            [
                INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=0))),
            ]
        )

    def press_escape(self) -> None:
        self._focus()
        VK_ESCAPE = 0x1B
        _send_inputs(
            [
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=VK_ESCAPE, wScan=0, dwFlags=0, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=VK_ESCAPE, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0))),
            ]
        )
