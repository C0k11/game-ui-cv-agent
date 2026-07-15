# -*- coding: utf-8 -*-
"""活动 Quest 参数化扫荡 (2026-07-15, 从 event_bonus_rerun B 段抽出).

用法: py -u scripts/sweep_event.py <关号> [次数|max]
  py -u scripts/sweep_event.py 10 9     # Q10 扫 9 次
  py -u scripts/sweep_event.py 12 max   # Q12 MAX 吃剩余 AP

铁律链(全 cls 驱动 fail-closed):
  扫前正向读 AP(体力cls锚定, <20 收工绝不试探) → Q 行=滑到底倒数第
  (13-q) 行入场键(结构化定位) → MAX_可点击 / 加号 cls 调次数 →
  扫荡开始 cls → 确认框结构白名单闸(body 无 stepper/体力才确认)。
"""
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\Project\ai game secretary")
from mumu_runner import AdbInput  # noqa: E402
from brain.pipeline import run_digit_ocr  # noqa: E402

ROOT = Path(r"D:\Project\ai game secretary")
TOTAL_Q = 12
SWEEP_COST = 20


def main():
    q = int(sys.argv[1])
    times = sys.argv[2] if len(sys.argv) > 2 else "max"
    reg = json.loads((ROOT / "data" / "model_registry.json")
                     .read_text(encoding="utf-8"))
    from ultralytics import YOLO
    ui = YOLO(reg["ui"]["versions"][reg["ui"]["active"]]["path"])
    adb = AdbInput()
    adb.connect()
    tap = lambda x, y: adb._shell(f"input tap {x} {y}")  # noqa: E731

    def dets(fr, conf=0.5):
        r = ui.predict(fr, conf=conf, imgsz=960, verbose=False)[0]
        return [(ui.names[int(b.cls[0])], float(b.conf[0]),
                 *(float(v) for v in b.xyxy[0]),
                 fr.shape[1], fr.shape[0]) for b in (r.boxes or [])]

    def box_center(d, name):
        x = next((b for b in d if b[0] == name), None)
        return (int((x[2] + x[4]) / 2), int((x[3] + x[5]) / 2)) if x else None

    def ensure_quest_tab():
        for _ in range(6):
            fr = adb.capture_frame()
            if fr is None:
                continue
            d = dets(fr, 0.35)
            names = {n for n, *_ in d}
            if "活动quest_已选择" in names or \
                    any(n.startswith("关卡得星") for n in names):
                return fr
            p = box_center(d, "活动quest")
            if p:
                tap(*p)
                print("  tab≠Quest → 点「活动quest」切换", flush=True)
            time.sleep(3)
        return None

    # 0. 扫前读 AP(fail-closed)
    fr = ensure_quest_tab()
    if fr is None:
        print("⛔Quest tab 正锚拿不到 — 停")
        return
    ap = None
    for n, c, x1, y1, x2, y2, W, H in dets(fr, 0.5):
        if n == "体力" and (y1 + y2) / 2 / H < 0.12:
            raw = run_digit_ocr(fr, (x2 / W + 0.002, max(0.0, y1 / H - 0.012),
                                     x2 / W + 0.062, y2 / H + 0.012))
            num = ""
            for ch in (raw or ""):
                if ch.isdigit():
                    num += ch
                elif num:
                    break
            ap = int(num) if num else None
            break
    print(f"扫前 AP={ap}", flush=True)
    if ap is None or ap < SWEEP_COST:
        print("AP 读不出/不足 — 收工(绝不试探)")
        return
    if times != "max" and int(times) * SWEEP_COST > ap:
        print(f"⛔ {times}次需 {int(times) * SWEEP_COST}AP > {ap} — 停")
        return

    # 1. Q 行 = 滑到底倒数第 (13-q) 行
    prev_sig = None
    for _ in range(10):
        fr = ensure_quest_tab()
        if fr is None:
            print("⛔Quest tab 正锚丢失 — 停")
            return
        d = dets(fr, 0.5)
        entries = sorted(
            ((y1 + y2) / 2 / H, (int((x1 + x2) / 2), int((y1 + y2) / 2)))
            for n, c, x1, y1, x2, y2, W, H in d if n == "入场键")
        sig = tuple(round(cy, 2) for cy, _ in entries)
        if entries and sig == prev_sig:
            k = TOTAL_Q + 1 - q
            if len(entries) < k:
                print(f"⛔到底但视野{len(entries)}行 < 倒数{k} — 停")
                return
            tap(*entries[-k][1])
            print(f"Q{q}=倒数第{k}行 → 入场", flush=True)
            time.sleep(7)
            break
        prev_sig = sig
        adb._shell("input swipe 2760 1500 2760 700 500")
        time.sleep(2.0)
    else:
        print("⛔找不到行 — 停")
        return

    # 2. 扫荡面板(全 cls, fail-closed)
    fr = adb.capture_frame()
    d = dets(fr, 0.5)
    if times == "max":
        mx = next((b for b in d if b[0] == "MAX_可点击"), None)
        if mx is None:
            print("⛔MAX_可点击 检不出 — 不扫")
            return
        tap(int((mx[2] + mx[4]) / 2), int((mx[3] + mx[5]) / 2))
        time.sleep(2.0)
    else:
        plus = next((b for b in d if b[0] == "加号"
                     and (b[3] + b[5]) / 2 / b[7] > 0.12), None)
        if plus is None:
            print("⛔加号(body) 检不出 — 不扫")
            return
        px, py = int((plus[2] + plus[4]) / 2), int((plus[3] + plus[5]) / 2)
        for _ in range(int(times) - 1):      # 起始=1, 点 N-1 次
            tap(px, py)
            time.sleep(0.5)
        time.sleep(1.5)
    fr = adb.capture_frame()
    d = dets(fr, 0.20)
    # 弹框前 body 危险 cls 的"位置"基线 — 扫荡面板自带 stepper(右上
    # x~0.8/y~0.42), 确认框弹出后会透出来(2026-07-15 实锤 Q10 手动次数
    # 时 MAX_可点击 仍亮被误拦)。⚠不能按类名差分(购买AP框的 stepper
    # 类名与面板相同会被排除→裸奔), 按位置差分: 同 cls 同位≈面板透出
    # (放行), 新位置(对话框中央)=购买框特征(拦)。
    _DANGER = ("加号", "MAX_可点击", "MIN_灰色", "体力")
    pre_pos = [(n, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H)
               for n, c, x1, y1, x2, y2, W, H in d
               if y1 / H > 0.12 and n in _DANGER]
    sw = next((b for b in d if b[0] == "扫荡开始" and b[1] >= 0.5), None)
    if sw is None:
        print("⛔扫荡开始 检不出 — 不扫")
        return
    tap(int((sw[2] + sw[4]) / 2), int((sw[3] + sw[5]) / 2))
    time.sleep(5)

    # 3. 确认框结构白名单闸(位置差分: 新位置的危险 cls 才拦)
    fr = adb.capture_frame()
    d = dets(fr, 0.20)
    names = {x[0] for x in d}

    def _is_new(n, cx, cy):
        return not any(pn == n and abs(px - cx) < 0.04 and abs(py - cy) < 0.04
                       for pn, px, py in pre_pos)

    body_bad = [n for n, c, x1, y1, x2, y2, W, H in d
                if y1 / H > 0.12 and n in _DANGER
                and _is_new(n, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H)]
    if {"取消键", "确认键"} <= names and not body_bad:
        ck = next(b for b in d if b[0] == "确认键")
        tap(int((ck[2] + ck[4]) / 2), int((ck[3] + ck[5]) / 2))
        print("扫荡确认(纯AP闸过) ✓", flush=True)
        time.sleep(6)
        # 结果窗: 確認 cls → 关卡面板叉
        fr = adb.capture_frame()
        d = dets(fr, 0.5)
        p = box_center(d, "确认键")
        if p:
            tap(*p)
        time.sleep(3)
        fr = adb.capture_frame()
        p = box_center(dets(fr, 0.5), "叉叉")
        if p:
            tap(*p)
        time.sleep(2)
        print("done", flush=True)
    else:
        if "取消键" in names:
            ck = next(b for b in d if b[0] == "取消键")
            tap(int((ck[2] + ck[4]) / 2), int((ck[3] + ck[5]) / 2))
        print(f"⛔确认框结构闸拦截(body={body_bad}) — 不扫, 人工看")


if __name__ == "__main__":
    main()
