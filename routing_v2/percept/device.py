# -*- coding: utf-8 -*-
"""设备层 —— ADB 只干两件事：**点** 和 **救命**。绝不在热路径抓帧。

为什么这么切（README §A4 / §C）:
  ADB screencap+decode 实测 **1588ms**(4K), 而 scrcpy latest() 是 **0ms**。
  老代码把抓帧和点击混在一条 ADB 链上, 于是每次决策都背 1.6s 延迟 ——
  活动入口轮播周期才 3.00s, 必然点在翻过去那一页上。这一层从结构上堵死:
  `screencap()` 只在**冷路径**(存活探针 / 冻结检测)可用, 且带显式告警。

内建的三个环境坑（都是实锤过的，不是防御性编程）:
  1. **端口会漂**: MuMu 实例重启 7555  16384。唯一权威 = MuMuManager info。
  2. **adbd 会卡死**: 三级恢复阶梯, setprop 那级实测无效（保留只为记录）。
  3. **分辨率不能写死**: `wm size` 问设备, 不假设 3840x2160。
     (语料里实测出 19 种分辨率 —— 写死过一次就够了。)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from typing import Optional, Tuple

def _find_exe(env_key: str, cands) -> str:
    """MuMu 可执行文件定位: 环境变量 > 逐候选探测 > 第一候选保底。

    保底不 raise: 让"文件不存在"的报错发生在用它的那一步, 错误信息里自带
    路径, 比 import 时炸好定位。路径别写死一处 -- 用户 08-26: MuMu 数据已
    搬 D 盘, 本体哪天跟着搬, 这里不用改码。
    """
    import os as _os
    env = str(_os.environ.get(env_key) or "").strip()
    if env and _os.path.isfile(env):
        return env
    for c in cands:
        if _os.path.isfile(c):
            return c
    return cands[0]


_ADB = _find_exe("MUMU_ADB", (
    r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe",
    r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe",
    r"D:\MuMu\nx_device\12.0\shell\adb.exe",
))
_MANAGER = _find_exe("MUMU_MANAGER", (
    r"C:\Program Files\Netease\MuMu\nx_main\MuMuManager.exe",
    r"D:\Program Files\Netease\MuMu\nx_main\MuMuManager.exe",
    r"D:\MuMu\nx_main\MuMuManager.exe",
))
_FALLBACK_SERIAL = "127.0.0.1:7555"
_PKG = "com.nexon.bluearchive"

# 全进程唯一的 ADB IO 锁。2026-06-15 实锤: 同 transport 上并发跑多 MB 的
# screencap 会**丢 MotionEvent** —— 表现成"点了没反应", 是最难查的形态。
# feed 的 watchdog / 存活探针 / tap 全部走这把锁。
IO_LOCK = threading.RLock()


def _run(args, timeout=15) -> str:
    """跑一条 adb 子命令, 返回 stdout。

    必须显式 encoding='utf-8': text=True 在中文 Windows 上默认走 GBK,
    MuMu 实例名是中文「MuMu安卓设备」 UnicodeDecodeError  **静默回落到
    错的端口**, 比不解析更糟（2026-07-28 当场踩到）。
    """
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout).stdout or ""
    except Exception:
        return ""


def resolve_serial(vmindex: int = 0) -> str:
    """ADB_SERIAL 环境变量 > MuMuManager(权威) > adb devices > 7555。"""
    env = str(os.environ.get("ADB_SERIAL") or "").strip()
    if env:
        return env
    out = _run([_MANAGER, "info", "-v", str(vmindex)], timeout=20)
    try:
        d = json.loads(out)
        rec = d.get(str(vmindex), d)
        port = rec.get("adb_port")
        host = rec.get("adb_host_ip") or "127.0.0.1"
        if port:
            return f"{host}:{int(port)}"
    except Exception:
        pass
    # 兜底: adb devices 里唯一 online 的 127.0.0.1:*
    # 只认 state == "device" —— offline 条目正是事故现场的样子
    #   (7555 offline 与 16384 device 同时列着), 认错就等于没修。
    live = []
    for line in _run([_ADB, "devices"]).splitlines()[1:]:
        p = line.split()
        if len(p) == 2 and p[1] == "device" and p[0].startswith("127.0.0.1:"):
            live.append(p[0])
    return live[0] if len(live) == 1 else _FALLBACK_SERIAL


class Device:
    """一个 MuMu 实例。tap 用归一化坐标，内部换算成设备像素。"""

    def __init__(self, serial: Optional[str] = None, log=None):
        self.serial = serial or resolve_serial()
        self._log = log or (lambda m: print(m, flush=True))
        self._size: Optional[Tuple[int, int]] = None
        self.taps = 0
        self.last_tap_ts = 0.0
        self.last_tap_pt: Optional[Tuple[float, float]] = None

    #  基础
    def sh(self, *args, timeout=15) -> str:
        with IO_LOCK:
            return _run([_ADB, "-s", self.serial, "shell", *args], timeout)

    def _physical(self) -> Tuple[int, int]:
        m = re.search(r"(\d+)x(\d+)", self.sh("wm", "size"))
        return (int(m.group(1)), int(m.group(2))) if m else (2160, 3840)

    def calibrate(self, frame_w: int, frame_h: int) -> Tuple[int, int]:
        """用**scrcpy 首帧的朝向**定 tap 坐标空间。开跑前必须调一次。

        2026-08-08 live 实锤（新架构第一次真机单步就撞上）:
           MuMu 的 `wm size` 在 **display 0/2/3 上一律返回 `2160x3840`（竖屏）**,
           而游戏在 display 2 上是**横屏 3840x2160**, `input tap` 用的也是横屏
           坐标空间。直接信 `wm size`  归一化 y=0.953 换算成 **3662 > 2160**,
           整发点到屏幕外, 表现成"点了没反应"。
           （老代码把 3840x2160 写死反而"碰巧对"—— 这就是为什么写死坐标能活
             那么久：它在唯一被测过的那个配置上是对的。）

        正解：物理面板给出两个边长，**哪个当宽由画面朝向决定**。
        这样竖屏模拟器 / 横屏模拟器 / 换分辨率都自动跟得上。
        """
        a, b = self._physical()
        lo, hi = min(a, b), max(a, b)
        self._size = (hi, lo) if frame_w >= frame_h else (lo, hi)
        self._log(f"[device] tap 坐标空间 {self._size[0]}x{self._size[1]}"
                  f"（物理面板报 {a}x{b}，scrcpy 帧 {frame_w}x{frame_h} "
                  f" 按{'横' if frame_w >= frame_h else '竖'}屏定向）")
        return self._size

    @property
    def size(self) -> Tuple[int, int]:
        """tap 坐标空间。别写死 —— 语料里实测 19 种分辨率。"""
        if self._size is None:
            a, b = self._physical()
            self._size = (a, b)
            self._log(f"[device] 还没 calibrate()，暂用物理面板 {a}x{b}"
                      f" —— 横屏游戏上这几乎肯定是错的，开跑前请先 calibrate")
        return self._size

    #  动作（唯一允许走 ADB 的热路径）
    def tap(self, nx: float, ny: float) -> bool:
        """归一化落点  input tap。实测稳态 32ms。"""
        w, h = self.size
        px, py = int(max(0.0, min(1.0, nx)) * w), int(max(0.0, min(1.0, ny)) * h)
        with IO_LOCK:
            ok = subprocess.run([_ADB, "-s", self.serial, "shell", "input",
                                 "tap", str(px), str(py)],
                                capture_output=True, timeout=15).returncode == 0
        if ok:
            self.taps += 1
            self.last_tap_ts = time.time()
            self.last_tap_pt = (nx, ny)
        return ok

    def swipe(self, x1: float, y1: float, x2: float, y2: float,
              ms: int = 350) -> bool:
        w, h = self.size
        with IO_LOCK:
            return subprocess.run(
                [_ADB, "-s", self.serial, "shell", "input", "swipe",
                 str(int(x1 * w)), str(int(y1 * h)),
                 str(int(x2 * w)), str(int(y2 * h)), str(ms)],
                capture_output=True, timeout=20).returncode == 0

    def key(self, keycode: str = "KEYCODE_BACK") -> bool:
        """剧情过场**不吃** KEYCODE_BACK（2026-08-07 实测连按 5 次全无响应）。
        剧情的退出只能靠 剧情menu  跳过故事键  确认键 那条链，见
        `flow/story.py`。这里保留按键能力，但没有任何 flow 用它做剧情逃生。"""
        with IO_LOCK:
            return subprocess.run(
                [_ADB, "-s", self.serial, "shell", "input", "keyevent", keycode],
                capture_output=True, timeout=15).returncode == 0

    #  冷路径（1588ms，绝不进决策循环）
    def screencap(self, _reason: str = ""):
        """ADB 抓一张。**只允许**存活探针/冻结检测/离线标定用。

        每次调用都打日志——如果你在热路径日志里看到它，那就是 bug 回来了。
        """
        t0 = time.time()
        try:
            import cv2
            import numpy as np
            with IO_LOCK:
                raw = subprocess.run(
                    [_ADB, "-s", self.serial, "exec-out", "screencap", "-p"],
                    capture_output=True, timeout=25).stdout
            fr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            fr = None
        self._log(f"    [device] ADB screencap({_reason}) "
                  f"{(time.time()-t0)*1000:.0f}ms — 冷路径专用")
        return fr

    #  存活 / 恢复
    def game_running(self) -> bool:
        """游戏进程在不在。

        `dumpsys mCurrentFocus` **不可信**（2026-08-07 实锤）: 它报前台是
        `app.lawnchair`（Android 桌面），而游戏好好地停在大厅 —— MuMu + Unity
        全屏不注册 window focus。所以这里只数**进程**，页面身份交给 YOLO。
        """
        return _PKG in self.sh("pidof", _PKG) or bool(
            re.search(rf"\b{re.escape(_PKG)}\b", self.sh("ps", "-A")))

    def launch_game(self) -> bool:
        self._log("[device] monkey 拉起游戏")
        self.sh("monkey", "-p", _PKG, "-c", "android.intent.category.LAUNCHER", "1",
                timeout=30)
        return True

    def alive(self) -> bool:
        """adb 通不通（一条极轻的 shell）。"""
        return "ok" in self.sh("echo", "ok", timeout=8)

    def recover(self) -> bool:
        """adbd 卡死三级阶梯（2026-08-02 / 08-07 各救活一次）。

        L1  disconnect + kill-server + start-server + connect    通常够
        L2  setprop ctl.restart adbd                             **实测无效**,
            保留只为记录"这条路走不通", 别再浪费时间试
        L3  MuMuManager control -v 0 restart                     核弹, 要重拉游戏
        """
        self._log("[device] adb 不通  L1 重连")
        _run([_ADB, "disconnect", self.serial])
        _run([_ADB, "kill-server"], 30)
        _run([_ADB, "start-server"], 30)
        _run([_ADB, "connect", self.serial], 30)
        if self.alive():
            self._log("[device] L1 救回")
            return True
        self._log("[device] L1 失败  L2 setprop（实测无效，走个过场）")
        self.sh("setprop", "ctl.restart", "adbd")
        time.sleep(2.0)
        if self.alive():
            return True
        self._log("[device] L3 MuMuManager restart —— 实例重启，端口会漂")
        _run([_MANAGER, "control", "-v", "0", "restart"], 120)
        for _ in range(40):
            time.sleep(3.0)
            self.serial = resolve_serial()      # 端口会变，必须重解析
            self._size = None
            if self.alive():
                self._log(f"[device] 实例回来了 serial={self.serial}")
                time.sleep(5.0)
                self.launch_game()
                return True
        return False

    def frozen(self) -> bool:
        """游戏画面冻死判据 —— **连续两帧 md5 完全一致**（唯一可靠判据）。

        这条只在"YOLO 认不出任何页面 + 长时间不动"时才值得跑（两次 ADB 抓帧
        = 3s+）。日常靠 scrcpy 的 seq 变化就够了。
        """
        import hashlib
        a = self.screencap("frozen-probe-1")
        time.sleep(1.2)
        b = self.screencap("frozen-probe-2")
        if a is None or b is None:
            return False
        return (hashlib.md5(a.tobytes()).hexdigest()
                == hashlib.md5(b.tobytes()).hexdigest())
