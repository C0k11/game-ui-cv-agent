# -*- coding: utf-8 -*-
"""MuMu ADB 端口解析 —— 别再把 7555 写死。

为什么要有这个模块(2026-07-28 live 事故):
跑到一半 adbd 卡死(dumpsys 15s 超时  scrcpy 起不来  管线掉到 DXcam 上
**零检出**), 从容器里 `setprop ctl.restart adbd` 没救回来, 只能
`MuMuManager control -v 0 restart` 重启实例。实例回来后**adb 端口从 7555
变成了 16384**, 而全仓 16 个文件把 `127.0.0.1:7555` 写死  模拟器好好地
跑着, bot 却连不上, 表现成"游戏没反应"这种最难查的形态。

MuMu12 的实例端口本来就不保证固定(BAAS 也专门判过 16384/7555 两种)。
唯一权威来源 = `MuMuManager.exe info -v <idx>` 的 `adb_port` 字段。

用法:
    from brain.mumu_port import mumu_serial
    serial = mumu_serial()            # "127.0.0.1:16384"
环境变量 `ADB_SERIAL` 永远优先(手动指定 / 多开 / 真机时用)。
"""
from __future__ import annotations

import json
import os
import subprocess

_MANAGER = r"C:\Program Files\Netease\MuMu\nx_main\MuMuManager.exe"
_ADB = r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
_FALLBACK = "127.0.0.1:7555"

_cache: dict = {}


def _from_manager(vmindex: int = 0) -> str | None:
    """问 MuMuManager 要这个实例当前的 adb 端口(权威)。"""
    try:
        # 必须显式 utf-8(实例名是中文「MuMu安卓设备」): text=True 在中文
        # Windows 上默认走 GBK  UnicodeDecodeError  **静默回落到错的端口**,
        # 比不解析更糟(2026-07-28 当场踩到)。
        out = subprocess.run([_MANAGER, "info", "-v", str(vmindex)],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=20).stdout
        d = json.loads(out)
        # 单实例时顶层就是那个实例; -v all 时是 {"0": {...}}
        rec = d.get(str(vmindex), d)
        port = rec.get("adb_port")
        host = rec.get("adb_host_ip") or "127.0.0.1"
        if port:
            return f"{host}:{int(port)}"
    except Exception:
        pass
    return None


def _from_adb_devices() -> str | None:
    """兜底: `adb devices` 里唯一 online 的 127.0.0.1:* 。

    只认 state == "device" —— offline 条目正是事故现场的样子
    (7555 offline 与 16384 device 同时列着), 认错就等于没修。"""
    try:
        out = subprocess.run([_ADB, "devices"], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception:
        return None
    live = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1] == "device" and parts[0].startswith("127.0.0.1:"):
            live.append(parts[0])
    return live[0] if len(live) == 1 else None


def mumu_serial(vmindex: int = 0, refresh: bool = False) -> str:
    """返回 "host:port"。ADB_SERIAL 环境变量 > MuMuManager > adb devices > 7555。

    结果缓存在进程内(每 tick 都去 spawn MuMuManager 会拖慢热路径);
    连接失败的调用方可以 refresh=True 重新解析一次再重试。"""
    env = str(os.environ.get("ADB_SERIAL") or "").strip()
    if env:
        return env
    if not refresh and _cache.get("serial"):
        return _cache["serial"]
    s = _from_manager(vmindex) or _from_adb_devices() or _FALLBACK
    _cache["serial"] = s
    return s


def mumu_host_port(vmindex: int = 0, refresh: bool = False):
    s = mumu_serial(vmindex, refresh)
    host, _, port = s.rpartition(":")
    try:
        return host or "127.0.0.1", int(port)
    except ValueError:
        return "127.0.0.1", 7555


if __name__ == "__main__":
    print(mumu_serial(refresh=True))
