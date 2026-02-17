import ctypes
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scripts.win_capture import capture_client, find_largest_visible_child, find_window_by_title_substring, get_client_rect_on_screen

user32 = ctypes.WinDLL("user32", use_last_error=True)

def _ensure_dpi_awareness() -> None:
    try:
        user32.SetProcessDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
        user32.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            print("[INFO] DPI Awareness: Set to Per-Monitor V2 (-4)", file=sys.stderr)
            return
    except Exception:
        pass
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.SetProcessDpiAwareness.argtypes = (ctypes.c_int,)
        shcore.SetProcessDpiAwareness.restype = ctypes.c_int
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        shcore.SetProcessDpiAwareness(2)
        print("[INFO] DPI Awareness: Set to Per-Monitor (ShCore)", file=sys.stderr)
        return
    except Exception:
        pass
    try:
        user32.SetProcessDPIAware.argtypes = ()
        user32.SetProcessDPIAware.restype = ctypes.c_bool
        user32.SetProcessDPIAware()
        print("[INFO] DPI Awareness: Set to System-Aware", file=sys.stderr)
    except Exception:
        print("[WARNING] DPI Awareness: Failed to set awareness", file=sys.stderr)
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

user32.GetForegroundWindow.argtypes = ()
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
user32.AttachThreadInput.restype = wintypes.BOOL
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
user32.ShowWindow.restype = wintypes.BOOL
user32.IsIconic.argtypes = (wintypes.HWND,)
user32.IsIconic.restype = wintypes.BOOL
user32.BringWindowToTop.argtypes = (wintypes.HWND,)
user32.BringWindowToTop.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.SetFocus.argtypes = (wintypes.HWND,)
user32.SetFocus.restype = wintypes.HWND
user32.PostMessageW.argtypes = (wintypes.HWND, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t)
user32.PostMessageW.restype = wintypes.BOOL

user32.ClipCursor.argtypes = (ctypes.POINTER(wintypes.RECT),)
user32.ClipCursor.restype = wintypes.BOOL
user32.GetClipCursor.argtypes = (ctypes.POINTER(wintypes.RECT),)
user32.GetClipCursor.restype = wintypes.BOOL
user32.ReleaseCapture.argtypes = ()
user32.ReleaseCapture.restype = wintypes.BOOL

user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
user32.GetCursorPos.restype = wintypes.BOOL

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

def _send_inputs(inputs: list[INPUT]) -> None:
    if not inputs:
        return
    arr = (INPUT * len(inputs))(*inputs)
    sent = int(user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT)))
    if sent != len(inputs):
        raise ctypes.WinError(ctypes.get_last_error())

def _get_virtual_screen_bounds() -> tuple[int, int, int, int]:
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
    print(f"[DEBUG] Virtual Screen Bounds: x={vx} y={vy} w={vw} h={vh}", file=sys.stderr)
    return vx, vy, max(1, vw), max(1, vh)

def _clamp_to_screen(sx: int, sy: int, *, warn: bool = True) -> tuple[int, int]:
    vx, vy, vw, vh = _get_virtual_screen_bounds()
    x = int(max(vx, min(vx + vw - 1, int(sx))))
    y = int(max(vy, min(vy + vh - 1, int(sy))))
    if warn and (x != int(sx) or y != int(sy)):
        print(
            f"[WARNING] Click target ({sx}, {sy}) is outside visible screen bounds "
            f"[{vx}, {vy}, {vx+vw}, {vy+vh}]. Clamped to ({x}, {y}). "
            f"Please resize/reposition the game window so the UI is fully visible!",
            file=sys.stderr,
        )
    return x, y

def _cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT(0, 0)
    try:
        if bool(user32.GetCursorPos(ctypes.byref(pt))):
            return int(pt.x), int(pt.y)
    except Exception:
        pass
    return 0, 0

def _to_abs(sx: int, sy: int) -> tuple[int, int]:
    vx, vy, vw, vh = _get_virtual_screen_bounds()
    x = int(max(vx, min(vx + vw - 1, int(sx))))
    y = int(max(vy, min(vy + vh - 1, int(sy))))
    dx = int((x - vx) * 65535 / max(1, vw - 1))
    dy = int((y - vy) * 65535 / max(1, vh - 1))
    print(f"[DEBUG] _to_abs({sx}, {sy}) -> ({dx}, {dy}) [virtual: {vx},{vy} {vw}x{vh}]", file=sys.stderr)
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
        hwnd = int(t.hwnd)
        try:
            try:
                if bool(user32.IsIconic(wintypes.HWND(hwnd))):
                    user32.ShowWindow(wintypes.HWND(hwnd), 9)
                else:
                    user32.ShowWindow(wintypes.HWND(hwnd), 5)
            except Exception:
                pass
            try:
                user32.BringWindowToTop(wintypes.HWND(hwnd))
            except Exception:
                pass

            fg = 0
            try:
                fg = int(user32.GetForegroundWindow() or 0)
            except Exception:
                fg = 0

            if fg and fg != hwnd:
                tid_fg = 0
                tid_hwnd = 0
                try:
                    pid1 = wintypes.DWORD(0)
                    pid2 = wintypes.DWORD(0)
                    tid_fg = int(user32.GetWindowThreadProcessId(wintypes.HWND(fg), ctypes.byref(pid1)) or 0)
                    tid_hwnd = int(user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid2)) or 0)
                except Exception:
                    tid_fg = 0
                    tid_hwnd = 0

                if tid_fg and tid_hwnd:
                    try:
                        user32.AttachThreadInput(wintypes.DWORD(tid_fg), wintypes.DWORD(tid_hwnd), True)
                    except Exception:
                        pass
                    try:
                        user32.SetForegroundWindow(wintypes.HWND(hwnd))
                    except Exception:
                        pass
                    try:
                        user32.SetFocus(wintypes.HWND(hwnd))
                    except Exception:
                        pass
                    try:
                        user32.AttachThreadInput(wintypes.DWORD(tid_fg), wintypes.DWORD(tid_hwnd), False)
                    except Exception:
                        pass
                else:
                    try:
                        user32.SetForegroundWindow(wintypes.HWND(hwnd))
                    except Exception:
                        pass
                    try:
                        user32.SetFocus(wintypes.HWND(hwnd))
                    except Exception:
                        pass
            else:
                try:
                    user32.SetForegroundWindow(wintypes.HWND(hwnd))
                except Exception:
                    pass
                try:
                    user32.SetFocus(wintypes.HWND(hwnd))
                except Exception:
                    pass
        except Exception:
            pass
        return hwnd

    def screenshot_client(self, out_path: str) -> None:
        last = None
        try:
            self._focus()
            time.sleep(0.05)
        except Exception:
            pass
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
                sx, sy = int(r.left + x), int(r.top + y)
                sx, sy = _clamp_to_screen(sx, sy, warn=False)
                return (sx, sy)
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

    def _click_via_message(self, x: int, y: int) -> bool:
        try:
            t = self._resolve()
            hwnds = [int(t.render_hwnd), int(t.hwnd)]
            hwnds = [h for h in hwnds if int(h) > 0]
            WM_MOUSEMOVE = 0x0200
            WM_LBUTTONDOWN = 0x0201
            WM_LBUTTONUP = 0x0202
            MK_LBUTTON = 0x0001
            lparam = (int(y) << 16) | (int(x) & 0xFFFF)
            ok_any = False
            seen = set()
            for hwnd in hwnds:
                if not hwnd or hwnd in seen:
                    continue
                seen.add(hwnd)
                try:
                    ok1 = bool(user32.PostMessageW(hwnd, WM_MOUSEMOVE, 0, lparam))
                    time.sleep(0.01)
                    ok2 = bool(user32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam))
                    time.sleep(0.05)
                    ok3 = bool(user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lparam))
                    ok_any = ok_any or (ok1 and ok2 and ok3)
                except Exception:
                    continue
            return bool(ok_any)
        except Exception:
            return False

    def click_client_message(self, x: int, y: int) -> bool:
        self._focus()
        try:
            user32.ClipCursor(None)
        except Exception:
            pass
        try:
            user32.ReleaseCapture()
        except Exception:
            pass
        try:
            return bool(self._click_via_message(int(x), int(y)))
        except Exception:
            return False

    def click_client(self, x: int, y: int) -> bool:
        self._focus()
        try:
            user32.ClipCursor(None)
        except Exception:
            pass
        try:
            user32.ReleaseCapture()
        except Exception:
            pass
        t = self._resolve()
        r = get_client_rect_on_screen(t.render_hwnd)
        raw_sx, raw_sy = int(r.left + x), int(r.top + y)
        sx, sy = _clamp_to_screen(raw_sx, raw_sy, warn=False)
        try:
            print(
                f"[DEBUG] click_client client=({x},{y}) screen=({sx},{sy}) "
                f"rect=[{r.left},{r.top},{r.right},{r.bottom}] "
                f"hwnd={t.hwnd} render={t.render_hwnd}",
                file=sys.stderr,
            )
        except Exception:
            pass
        if (raw_sx != sx) or (raw_sy != sy):
            _clamp_to_screen(raw_sx, raw_sy, warn=True)

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
                )
            ]
        )
        time.sleep(0.01)

        cx, cy = _cursor_pos()
        if abs(int(cx) - int(sx)) + abs(int(cy) - int(sy)) > 12:
            ok_msg = False
            try:
                ok_msg = bool(self._click_via_message(int(x), int(y)))
            except Exception:
                ok_msg = False

            try:
                in_win = (int(r.left) <= int(cx) < int(r.right)) and (int(r.top) <= int(cy) < int(r.bottom))
            except Exception:
                in_win = False

            try:
                print(
                    f"[INFO] Cursor could not reach target ({sx}, {sy}) (at {cx}, {cy}); "
                    f"PostMessage_ok={int(bool(ok_msg))} in_win={int(bool(in_win))} client=({x}, {y}).",
                    file=sys.stderr,
                )
            except Exception:
                pass

            if in_win:
                _send_inputs(
                    [
                        INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=0))),
                    ]
                )
                time.sleep(0.02)
                _send_inputs(
                    [
                        INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=0))),
                    ]
                )
            if not ok_msg:
                try:
                    time.sleep(0.02)
                    self._click_via_message(int(x), int(y))
                except Exception:
                    pass
            return False

        _send_inputs(
            [
                INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=0))),
            ]
        )
        time.sleep(0.02)
        _send_inputs(
            [
                INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=0))),
            ]
        )
        return True

    def swipe_client(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        self._focus()
        try:
            user32.ClipCursor(None)
        except Exception:
            pass
        try:
            user32.ReleaseCapture()
        except Exception:
            pass
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
        SC_ESCAPE = 0x01
        _send_inputs(
            [
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=VK_ESCAPE, wScan=0, dwFlags=0, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=VK_ESCAPE, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=0, wScan=SC_ESCAPE, dwFlags=KEYEVENTF_SCANCODE, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=0, wScan=SC_ESCAPE, dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0))),
            ]
        )

    def press_space(self) -> None:
        self._focus()
        VK_SPACE = 0x20
        SC_SPACE = 0x39
        _send_inputs(
            [
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=VK_SPACE, wScan=0, dwFlags=0, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=VK_SPACE, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=0, wScan=SC_SPACE, dwFlags=KEYEVENTF_SCANCODE, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=0, wScan=SC_SPACE, dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0))),
            ]
        )

    def press_enter(self) -> None:
        self._focus()
        VK_RETURN = 0x0D
        SC_RETURN = 0x1C
        _send_inputs(
            [
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=VK_RETURN, wScan=0, dwFlags=0, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=VK_RETURN, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=0, wScan=SC_RETURN, dwFlags=KEYEVENTF_SCANCODE, time=0, dwExtraInfo=0))),
                INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=0, wScan=SC_RETURN, dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0))),
            ]
        )
