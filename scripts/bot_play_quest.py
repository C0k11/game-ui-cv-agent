# -*- coding: utf-8 -*-
"""bot 控牌打活动 Quest (2026-07-15, combat 2.0 首个实战任务).

背景: 加成满配队(自动编队 100/105)含 Lv1-36 低级角色, AUTO 打 Q10(Lv23)
超时 DEFEAT — 用户拍板: AUTO 不可靠, bot 自己放牌; 胜利才记加成; farm
只刷最后三关(Q10/11/12)。

打法: AUTO 保持关闭(角色自动普攻, EX 全由 bot 控) —
  每 ~3s: fused_avatar 检出手牌(y>0.78) → 优先 Lv90 输出卡(優香體育服/
  陽葵/花凛) → tap 卡 → battle v9 检出敌方 → tap 最大敌方框(boss) 释放。
  cost 不够时点卡无反应(无害), 3s 节奏 ≈ cost 回复速度。
链路: Quest tab 正锚 → 关号 OCR 入场 → 感知编队(快速编辑→自动编辑按钮→
  确认→出击) → 控牌战斗 → 胜利结算+台账 / DEFEAT(battle 域空+确认键)重试上限。

用法: py -u scripts/bot_play_quest.py 10 11 12
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
LEDGER = ROOT / "data" / "event_bonus_state_midnight.json"
PRIORITY = ("体育服", "阳葵", "陽葵", "花凛", "优香", "優香")   # Lv90 输出优先
MAX_RETRY = 2


def main():
    targets = [int(a) for a in sys.argv[1:] if a.isdigit()] or [10, 11, 12]
    reg = json.loads((ROOT / "data" / "model_registry.json")
                     .read_text(encoding="utf-8"))
    from ultralytics import YOLO
    ui = YOLO(reg["ui"]["versions"][reg["ui"]["active"]]["path"])
    vers = reg["battle_heads"]["versions"]
    vn = max((v for v in vers if re.fullmatch(r"v\d+", v)),
             key=lambda x: int(x[1:]))
    battle = YOLO(vers[vn]["path"])
    fav = reg["fused_avatar"]
    avatar = YOLO(fav["versions"][fav["active"]]["path"])
    adb = AdbInput()
    adb.connect()
    tap = lambda x, y: adb._shell(f"input tap {x} {y}")  # noqa: E731

    def dets(model, fr, conf=0.5):
        r = model.predict(fr, conf=conf, imgsz=960, verbose=False)[0]
        H, W = fr.shape[:2]
        return [(model.names[int(b.cls[0])], float(b.conf[0]),
                 *(float(v) for v in b.xyxy[0]), W, H)
                for b in (r.boxes or [])]

    def box_of(d, name):
        x = next((b for b in d if b[0] == name), None)
        return (int((x[2] + x[4]) / 2), int((x[3] + x[5]) / 2)) if x else None

    def ensure_quest_tab():
        for _ in range(6):
            fr = adb.capture_frame()
            if fr is None:
                continue
            d = dets(ui, fr, 0.35)
            names = {n for n, *_ in d}
            if "活动quest_已选择" in names or \
                    any(n.startswith("关卡得星") for n in names):
                return fr
            p = box_of(d, "活动quest")
            if p:
                tap(*p)
            time.sleep(3)
        return None

    def find_and_enter(q: int) -> bool:
        for _ in range(8):
            fr = ensure_quest_tab()
            if fr is None:
                return False
            d = dets(ui, fr, 0.5)
            H = fr.shape[0]
            rows = []
            for n, c, x1, y1, x2, y2, W, Hh in d:
                if n == "入场键":
                    cy = (y1 + y2) / 2 / Hh
                    num = run_digit_ocr(
                        fr, (0.542, cy - 0.028, 0.60, cy + 0.028))
                    rows.append((int(num) if num and num.isdigit() else None,
                                 cy))
            hit = next(((qq, cy) for qq, cy in rows if qq == q), None)
            nums = [qq for qq, _ in rows if qq]
            print(f"  找Q{q} 视野{nums}", flush=True)
            if hit:
                tap(3340, int(hit[1] * 2160))
                time.sleep(7)
                tap(2803, 1620)          # 任務開始
                time.sleep(9)
                return True
            if nums and q < min(nums):
                adb._shell("input swipe 2760 700 2760 1500 500")
            else:
                adb._shell("input swipe 2760 1500 2760 700 500")
            time.sleep(2.5)
        return False

    def formation_and_sortie() -> bool:
        auto_done = False
        for _ in range(14):
            fr = adb.capture_frame()
            if fr is None:
                continue
            d = dets(ui, fr, 0.5)
            names = {n for n, *_ in d}
            if "自动编辑按钮" in names:        # 快速编辑面板开着
                p = box_of(d, "自动编辑按钮")
                tap(*p)
                print("    自动编辑(面板内)", flush=True)
                time.sleep(2.5)
                fr2 = adb.capture_frame()
                d2 = dets(ui, fr2, 0.5)
                p2 = box_of(d2, "确认键")
                if p2:
                    tap(*p2)
                    print("    面板确认", flush=True)
                    auto_done = True
                    time.sleep(2.5)
                continue
            if "2部队高亮" in names:
                if not auto_done:
                    p = box_of(d, "快速编辑")
                    if p:
                        tap(*p)
                        print("    快速编辑", flush=True)
                        time.sleep(3)
                        continue
                p = box_of(d, "出击")
                if p:
                    tap(*p)
                    print("    出击", flush=True)
                    return True
            elif "2部队" in names:
                p = box_of(d, "2部队")
                tap(*p)
                print("    切部队2", flush=True)
                time.sleep(2.5)
            time.sleep(1.2)
        return False

    def play_battle() -> str:
        """控牌战斗: 返回 win/lose/timeout"""
        t0 = time.time()
        last_play = 0.0
        while time.time() - t0 < 300:
            fr = adb.capture_frame()
            if fr is None:
                continue
            bd = dets(battle, fr, 0.5)
            bnames = {b[0] for b in bd}
            if "战斗胜利" in bnames:
                print(f"    ⭐胜利({time.time()-t0:.0f}s)", flush=True)
                return "win"
            if "自动战斗开启" in bnames:     # AUTO 要关(bot 控牌)
                tap(3621, 2023)
                print("    AUTO→关(bot接管)", flush=True)
                time.sleep(1)
                continue
            in_battle = bool(bnames & {"我方", "敌方", "战斗暂停",
                                       "自动战斗关闭"})
            if not in_battle:
                # battle 域空: 可能 DEFEAT/结算(词表无失败cls) → ui 判
                du = dets(ui, fr, 0.5)
                if any(n == "确认键" for n, *_ in du) and \
                        time.time() - t0 > 30:
                    print(f"    battle域空+确认键({time.time()-t0:.0f}s)"
                          f" → 判败/结束", flush=True)
                    return "lose"
                continue
            # ── 控牌: 3s 节奏 ──
            if time.time() - last_play >= 3.0:
                cards = [(n, c, (x1 + x2) / 2, (y1 + y2) / 2)
                         for n, c, x1, y1, x2, y2, W, H in
                         dets(avatar, fr, 0.4)
                         if (y1 + y2) / 2 / H > 0.78]
                if cards:
                    pri = [k for k in cards
                           if any(p in k[0] for p in PRIORITY)]
                    n, c, kx, ky = (pri or cards)[0]
                    tap(int(kx), int(ky))
                    time.sleep(1.0)
                    fr2 = adb.capture_frame()
                    foes = [(bb[1], (bb[2] + bb[4]) / 2, (bb[3] + bb[5]) / 2)
                            for bb in dets(battle, fr2, 0.5)
                            if bb[0] == "敌方"]
                    if foes:
                        _, tx, ty = max(foes)
                        tap(int(tx), int(ty))
                    else:
                        tap(1920, 1080)
                    print(f"    ⚡放牌 {n} → "
                          f"{'敌' if foes else '中央'}", flush=True)
                    last_play = time.time()
        return "timeout"

    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    for q in targets:
        for attempt in range(MAX_RETRY):
            print(f"[Q{q}] 第{attempt+1}次", flush=True)
            if not find_and_enter(q):
                print(f"[Q{q}] 找不到入口 — 停")
                return
            if not formation_and_sortie():
                print(f"[Q{q}] 编队失败 — 停")
                return
            result = play_battle()
            # 结算/败页收尾三连(win 也走同链)
            for x, y in [(3437, 1998), (1920, 2005), (2318, 1998)]:
                time.sleep(6)
                tap(x, y)
            time.sleep(7)
            if result == "win":
                led["bonus_cleared"][str(q)] = "full"   # 满加成标记
                tmp = LEDGER.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(led, ensure_ascii=False, indent=1),
                               encoding="utf-8")
                tmp.replace(LEDGER)
                print(f"[Q{q}] ✓ 满加成入账", flush=True)
                break
            print(f"[Q{q}] {result}, 重试" if attempt + 1 < MAX_RETRY
                  else f"[Q{q}] {result}, 放弃(换配置再议)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
