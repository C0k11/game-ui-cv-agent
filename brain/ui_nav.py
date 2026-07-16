# -*- coding: utf-8 -*-
"""UiNav: 日常导航用战斗同款 scrcpy 高频感知 (2026-07-15 用户拍板).

用户原话: "日常管线也要用上战斗一样的配置, 锁定到直接瞬发点击不要
拖泥带水" — 检测→点击全链压到最短:
  ScrcpyFeed(17.9fps, 帧龄0.02s) → ui 模型每帧推理 → 黑板
  → wait_cls(20ms 轮询, cls 出现即返回) → tap 瞬发。
零固定 sleep: 所有等待都是"等目标 cls 出现/消失", 不等墙钟。

坐标: scrcpy 1440p 帧 → 归一化 → tap 乘 4K(3840x2160)。
"""
import time

from mumu_runner import AdbInput
from brain.scrcpy_feed import ScrcpyFeed

TAP_W, TAP_H = 3840, 2160


class UiNav:
    def __init__(self, ui_model, adb: AdbInput = None, log=print):
        self.ui = ui_model
        self.adb = adb or AdbInput()
        if adb is None:
            self.adb.connect()
        self.log = log
        self.feed = ScrcpyFeed(log=log)
        if not self.feed.start():
            raise RuntimeError("scrcpy feed 起不来")
        self._last_seq = 0

    def stop(self):
        self.feed.stop()

    # ── 感知(黑板) ─────────────────────────────────────────────
    def snap(self, conf=0.4, wait_new=False, timeout=2.0):
        """当前帧检出 {cls: [(cx,cy) 归一化]}. wait_new=等新帧(seq 前进)."""
        t0 = time.time()
        while True:
            fr, age, seq = self.feed.latest()
            if fr is not None and (not wait_new or seq != self._last_seq):
                self._last_seq = seq
                break
            if time.time() - t0 > timeout:
                if fr is None:
                    return {}, None
                break
            time.sleep(0.02)
        r = self.ui.predict(fr, conf=conf, imgsz=960, verbose=False)[0]
        out = {}
        for b in (r.boxes or []):
            n = self.ui.names[int(b.cls[0])]
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            H, W = fr.shape[:2]
            out.setdefault(n, []).append(
                ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H))
        return out, fr

    def wait_cls(self, names, timeout=12.0, conf=0.4):
        """事件驱动等待: names 中任一 cls 出现 → 立即返回 (cls, box).
        黑板 20ms 轮询, 静止画面不重推(seq 门)。超时 → (None, None)."""
        if isinstance(names, str):
            names = [names]
        t0 = time.time()
        d_last = {}
        while time.time() - t0 < timeout:
            d, _ = self.snap(conf=conf, wait_new=True, timeout=0.5)
            d_last = d or d_last
            for n in names:
                if d and d.get(n):
                    return n, d[n][0]
            time.sleep(0.02)
        return None, None

    # ── 动作(瞬发) ─────────────────────────────────────────────
    def tap_norm(self, p):
        self.adb._shell(f"input tap {int(p[0] * TAP_W)} {int(p[1] * TAP_H)}")

    def tap_cls(self, names, timeout=12.0):
        """目标 cls 出现 → 瞬发点击。返回命中的 cls 名或 None."""
        n, p = self.wait_cls(names, timeout)
        if n is None:
            return None
        self.tap_norm(p)
        return n

    def swipe_tap_norm(self, p, dx_norm=0.04):
        """轮播拉停原子点击(微滑一格+tap 一条 shell 连发)."""
        cx, cy = int(p[0] * TAP_W), int(p[1] * TAP_H)
        dx = int(dx_norm * TAP_W)
        self.adb.swipe_tap(cx + dx, cy, cx - dx, cy, 250, cx, cy)
