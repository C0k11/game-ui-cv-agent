"""AdbInput: taps, swipes, key events and screencap for the MuMu emulator over adb.

server/app.py, brain/scrcpy_feed.py and several scripts import it from here. The standalone
runner that used to live in this file (window capture, overlay, DailyPipeline loop) was archived
on 2026-09-28 (archive/2026-09-28/_orig/mumu_runner.py); daily runs go through routing_v2.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

#  Repo setup
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


#  ADB Input

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

    #  ADB I/O serialization (live 2026-06-15: bounty/jfd swept 0 tickets).
    # Root cause: the clean-flywheel worker's `exec-out screencap` (streaming a
    # multi-MB 3840x2160 PNG ~200-300ms) and the main-tick `input tap` hit the
    # same adbd transport concurrently with NO lock - the tap's MotionEvent got
    # dropped/timed-out, so the game never saw the 入場 press (manual same-pos
    # tap DID open the popup -> not a coordinate bug). One class-wide lock makes
    # every adb subprocess (capture/tap/swipe/back) mutually exclusive: a tap
    # now waits ≤1 in-flight screencap instead of racing it. The old comment
    # "each capture is thread-safe vs input taps" was a wrong assumption.
    _IO_LOCK = threading.Lock()

    def __init__(self, host: str | None = None, port: int | None = None):
        # 禁端口不写死(2026-07-28 事故: MuMu 实例重启后端口 7555->16384,
        # 模拟器好好跑着 bot 却连不上)。见 brain/mumu_port 的长注释。
        if host is None or port is None:
            try:
                from brain.mumu_port import mumu_host_port
                _h, _p = mumu_host_port()
            except Exception:
                _h, _p = "127.0.0.1", 7555
            host = host or _h
            port = port or _p
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
            with AdbInput._IO_LOCK:
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

    def swipe_tap(self, x1: int, y1: int, x2: int, y2: int, dur_ms: int,
                  tx: int, ty: int) -> bool:
        # 原子连发(一条 adb shell 内 swipe->tap, 间隔≈input进程启动~0.3s):
        # 对自动轮播类 UI, 手动 swipe 会把轮播拉停数秒(2026-07-09 hub banner
        # 实锤), tap 在静止期内落点无时序竞争 - 分两次 adb 调用则间隔 >1s
        # 会耗尽暂停期(0709 败因)。
        return self._shell(
            f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(dur_ms)}"
            f" && input tap {int(tx)} {int(ty)}",
            timeout=max(6.0, dur_ms / 1000.0 + 4.0),
        )

    def back(self) -> bool:
        return self._shell("input keyevent 4")  # KEYCODE_BACK

    def capture_frame(self) -> Optional[np.ndarray]:
        # *RAW screencap 优先 (2026-07-11 实测: PNG 1.50s vs RAW 0.77s @4K -
        # 设备端 PNG 编码是大头, localhost 传 33MB 反而快)。RAW 头=w,h,format
        # (+colorspace) uint32 LE, 后跟 RGBA8888。解析失败回退 PNG。
        try:
            with AdbInput._IO_LOCK:
                r = subprocess.run(
                    [self._adb, "-s", self.addr, "exec-out", "screencap"],
                    capture_output=True, timeout=8,
                )
            data = bytes(r.stdout or b"")
            if len(data) > 16:
                import struct
                w0, h0 = struct.unpack_from("<II", data, 0)
                if 100 < w0 < 10000 and 100 < h0 < 10000:
                    expect = w0 * h0 * 4
                    hdr = len(data) - expect
                    if hdr in (12, 16):
                        arr = np.frombuffer(data, np.uint8, count=expect,
                                            offset=hdr).reshape(h0, w0, 4)
                        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        except Exception:
            pass
        try:
            with AdbInput._IO_LOCK:
                r = subprocess.run(
                    [self._adb, "-s", self.addr, "exec-out", "screencap", "-p"],
                    capture_output=True,
                    timeout=8,
                )
            data = bytes(r.stdout or b"")
            if not data:
                return None
            if data.startswith(b"\x89PNG"):
                png_bytes = data
            else:
                png_bytes = data.replace(b"\r\n", b"\n")
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            if arr.size == 0:
                return None
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                return None
            return frame
        except Exception as e:
            print(f"[ADB] capture_frame error: {e}")
            return None

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
