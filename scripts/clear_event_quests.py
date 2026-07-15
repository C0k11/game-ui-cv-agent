# -*- coding: utf-8 -*-
"""活动 Quest 首通连清 v2 (2026-07-15, 用户三连纠偏后的状态感知版).

替代盲坐标链(v1 的 Q06/Q08 重打实锤):
  列表页   ui v13 检出「入场键」(cls 0.96+) + 每行星数(金星 HSV 像素判定)
           → 选**0星**最上行入场; 全有星 → 首通完成退出。「入场键没解锁」不碰。
  资讯页   任務開始(坐标+后验)
  编队页   1部隊 tab(用户规则: 首通=速推主力队) → 出擊
  战斗     battle v9 检出战斗HUD → tap AUTO; 「战斗胜利」cls 出现 → 立即
           结算三连(事件驱动, 不再盲 sleep 干等 — 用户点破的浪费)
用法: py scripts/clear_event_quests.py [max_quests=8] [--skill-test]
  --skill-test: 战斗中周期性 试点技能卡→点敌方框中心释放(combat 2.0 操作
  链首验: 卡选中→战场目标两步 tap; buff/自动型卡第二击落空=无害)。
"""
import json
import re
import sys
import time
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\Project\ai game secretary")
from mumu_runner import AdbInput  # noqa: E402

ROOT = Path(r"D:\Project\ai game secretary")
STAR_X = (0.535, 0.615)          # 星星区(入场键同行左侧)
STAR_YELLOW_FRAC = 0.015         # 金星像素占比阈值(0星≈0)


def load_models():
    reg = json.loads((ROOT / "data" / "model_registry.json")
                     .read_text(encoding="utf-8"))
    from ultralytics import YOLO
    ui = YOLO(reg["ui"]["versions"][reg["ui"]["active"]]["path"])
    vers = reg["battle_heads"]["versions"]
    vn = max((v for v in vers if re.fullmatch(r"v\d+", v)),
             key=lambda x: int(x[1:]))
    battle = YOLO(vers[vn]["path"])
    return ui, battle


def dets(model, fr, conf=0.5):
    H, W = fr.shape[:2]
    r = model.predict(fr, conf=conf, imgsz=960, verbose=False)[0]
    out = []
    for b in (r.boxes or []):
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        out.append((model.names[int(b.cls[0])], float(b.conf[0]),
                    (x1 + x2) / 2 / W, (y1 + y2) / 2 / H))
    return out


def stars_present(fr, cy: float) -> bool:
    """入场键行左侧星星区有金星? (金=高饱和黄)"""
    H, W = fr.shape[:2]
    crop = fr[int((cy - 0.04) * H):int((cy + 0.04) * H),
              int(STAR_X[0] * W):int(STAR_X[1] * W)]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow = ((hsv[..., 0] > 15) & (hsv[..., 0] < 40) &
              (hsv[..., 1] > 120) & (hsv[..., 2] > 150)).mean()
    return yellow > STAR_YELLOW_FRAC


def main():
    max_q = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    skill_test = "--skill-test" in sys.argv
    ui, battle = load_models()
    avatar = None
    if skill_test:
        reg = json.loads((ROOT / "data" / "model_registry.json")
                         .read_text(encoding="utf-8"))
        from ultralytics import YOLO
        fav = reg["fused_avatar"]
        avatar = YOLO(fav["versions"][fav["active"]]["path"])
    adb = AdbInput()
    adb.connect()
    tap = lambda x, y: adb._shell(f"input tap {x} {y}")  # noqa: E731
    cleared = 0

    for round_i in range(max_q):
        # ── 列表页: 找 0 星可入场行 ──
        fr = adb.capture_frame()
        d = dets(ui, fr, 0.5)
        enters = sorted([(cy, cx) for n, c, cx, cy in d if n == "入场键"])
        if not enters:
            print(f"[r{round_i}] 列表页无「入场键」检出 — 停(人工看)")
            break
        target = None
        for cy, cx in enters:
            if not stars_present(fr, cy):
                target = (cx, cy)
                break
        if target is None:
            print(f"[r{round_i}] 可入场行全有星 → 首通完成")
            break
        print(f"[r{round_i}] 0星行 cy={target[1]:.3f} → 入场")
        tap(int(target[0] * 3840), int(target[1] * 2160))
        time.sleep(8)

        tap(2803, 1620)                     # 任務開始
        time.sleep(9)
        tap(215, 561)                       # 1部隊(用户规则: 首通主力队)
        time.sleep(3)
        tap(3533, 1976)                     # 出擊
        print("  出击, 等战斗HUD...")

        # ── 战斗: AUTO 状态门(⚠盲 tap=toggle, AUTO 已开再点会关掉 —
        # 用户实锤 bug) → 检出「自动战斗关闭」cls 才点; 战斗胜利 → 结算 ──
        t0 = time.time()
        frame_i = 0
        while time.time() - t0 < 240:
            fr = adb.capture_frame()
            if fr is None:
                continue
            frame_i += 1
            bd = dets(battle, fr, 0.5)
            names = {n for n, c, cx, cy in bd}
            if "自动战斗关闭" in names:
                tap(3621, 2023)
                print(f"  AUTO=关({time.time()-t0:.0f}s) → 点开")
            if "战斗胜利" in names:
                print(f"  战斗胜利({time.time()-t0:.0f}s) → 结算链")
                break
            # ── 技能卡操作试验: 点亮卡 → 点敌方框中心释放 ──
            if skill_test and frame_i % 8 == 0 and "我方" in names:
                cards = [(c, cx, cy) for n, c, cx, cy in dets(avatar, fr, 0.5)
                         if cy > 0.78]
                if cards:
                    _, kx, ky = max(cards)
                    tap(int(kx * 3840), int(ky * 2160))
                    time.sleep(1.2)
                    fr2 = adb.capture_frame()
                    foes = [(c, cx, cy) for n, c, cx, cy in
                            dets(battle, fr2, 0.5) if n == "敌方"]
                    tx, ty = (foes and max(foes)[1:]) or (0.5, 0.5)
                    tap(int(tx * 3840), int(ty * 2160))
                    print(f"  ⚡技能试验: 卡({kx:.2f},{ky:.2f}) → "
                          f"目标({tx:.2f},{ty:.2f}) {'敌方' if foes else '中央'}")
        else:
            print("  战斗 240s 超时 — 停(人工看)")
            break
        for x, y in [(3437, 1998), (1920, 2005), (2318, 1998)]:
            time.sleep(6)
            tap(x, y)
        time.sleep(8)
        cleared += 1
        print(f"  ✓ 第 {cleared} 关清完")

    fr = adb.capture_frame()
    cv2.imencode(".jpg", cv2.resize(fr, (960, 540)))[1].tofile(
        str(ROOT / "data" / "_probe_now.jpg"))
    print(f"done, cleared={cleared}")


if __name__ == "__main__":
    main()
