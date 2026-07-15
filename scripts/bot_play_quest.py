# -*- coding: utf-8 -*-
"""bot 控牌打活动 Quest (combat 2.0 v7, 2026-07-15).

背景: 加成满配队(自动编队 100/105)含 Lv1-36 低级角色, AUTO 打 Q10
超时 DEFEAT — 用户拍板: 70级满加成队人手随便过, 输在放牌逻辑;
AUTO 不可靠 bot 自己放牌; 胜利才记加成; farm 只刷最后三关(Q10/11/12)。

v7 架构(playbook 4.8 定案): 战斗感知走 scrcpy 视频流(~25fps, 帧龄
<150ms) → Perception 线程写黑板(全队HP/亮卡/敌表/boss) → 行为树控牌
(急救>集火BOSS>AOE清群>辅助增益>单体循环>攒费) → 闭环拖拽释放(拖拽
中每步读黑板跟目标)。实现在 brain/combat_brain.py。
导航/编队仍走 ADB 干净帧链(非实时场景, 1fps 够)。

链路: Quest tab 双正锚 → 关号 OCR 入场 → 感知编队(快速编辑→自动编辑
按钮→确认→出击) → v7 控牌 → 胜利结算+台账 / 判败重试上限。

用法: py -u scripts/bot_play_quest.py 10 11 12   (--resume 接管战斗)
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
from brain.scrcpy_feed import ScrcpyFeed  # noqa: E402
from brain.combat_brain import Perception, CombatBrain  # noqa: E402

ROOT = Path(r"D:\Project\ai game secretary")
LEDGER = ROOT / "data" / "event_bonus_state_midnight.json"
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

    def read_row_num(fr, cy_norm):
        """OCR 辅助校验(rec-only 直读, det 在此场景 0.66 边缘抖动不可靠).
        只在清洗出 1-2 位且 ≤12 的数字时参与判断, 否则返回 None 不挡路."""
        try:
            from brain.pipeline import _get_ocr
            H, W = fr.shape[:2]
            crop = fr[int((cy_norm - 0.028) * H):int((cy_norm + 0.028) * H),
                      int(0.542 * W):int(0.60 * W)]
            out = _get_ocr().text_recognizer([crop])
            txts = out[0] if isinstance(out, tuple) else out
            digs = re.findall(r"\d+", "".join(t for t, _ in (txts or [])))
            if digs and len(digs[0]) <= 2 and 1 <= int(digs[0]) <= 12:
                return int(digs[0])
        except Exception:
            pass
        return None

    def find_and_enter(q: int) -> bool:
        """结构化定位(词表铁律: 关号无数字cls, 但列表结构有):
        活动 Quest 固定 12 关全解锁 → 滑到底(入场键 cy 集合两帧不变)
        → 行序=关号, Q_q = 倒数第 (13-q) 行入场键。OCR 只辅助校验。"""
        prev_sig = None
        for _ in range(10):
            fr = ensure_quest_tab()
            if fr is None:
                return False
            d = dets(ui, fr, 0.5)
            rows = sorted(
                ((y1 + y2) / 2 / Hh, (int((x1 + x2) / 2),
                                      int((y1 + y2) / 2)))
                for n, c, x1, y1, x2, y2, W, Hh in d if n == "入场键")
            sig = tuple(round(cy, 2) for cy, _ in rows)
            if rows and sig == prev_sig:      # 滑不动了 = 到底
                k = 13 - q
                if len(rows) < k:
                    print(f"  到底但视野{len(rows)}行 < 倒数{k} — 停",
                          flush=True)
                    return False
                cy, (px, py) = rows[-k]
                num = read_row_num(fr, cy)
                if num is not None and num != q:
                    print(f"  行序倒数{k}=Q{q} 但OCR读{num} — 矛盾停",
                          flush=True)
                    return False
                print(f"  Q{q}=倒数第{k}行 cy={cy:.2f} OCR辅证={num}",
                      flush=True)
                tap(px, py)                   # 入场键 box 中心
                time.sleep(7)
                tap(2803, 1620)               # 任務開始
                time.sleep(9)
                return True
            prev_sig = sig
            adb._shell("input swipe 2760 1500 2760 700 500")
            time.sleep(2.0)
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

    SKILLS = json.loads((ROOT / "data" / "skill_type_map.json")
                        .read_text(encoding="utf-8"))
    fly_dir = ROOT / "data" / "raw_images" / \
        ("run_" + time.strftime("%Y%m%d_%H%M%S") + "_botplay_clean")
    fly_dir.mkdir(parents=True, exist_ok=True)
    feed = ScrcpyFeed(log=lambda m: print(m, flush=True))
    if not feed.start():
        print("scrcpy feed 起不来 — 停(不盲打)", flush=True)
        return
    f0, _, _ = feed.latest()
    print(f"scrcpy feed OK {f0.shape[1]}x{f0.shape[0]}", flush=True)

    def play_battle() -> str:
        """v7: 感知线程(scrcpy→黑板) + 行为树控牌(brain/combat_brain.py).
        飞轮干净帧由感知线程 1fps 直存(scrcpy=Android 内部流, 无烧录)。"""
        per = Perception(feed, battle, avatar, ui, flywheel_dir=fly_dir)
        per.start()
        try:
            brain = CombatBrain(per, adb._shell, SKILLS,
                                log=lambda m: print(m, flush=True))
            return brain.run_battle(timeout_s=300)
        finally:
            per.stop()
            per.join(timeout=5)

    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    if "--resume" in sys.argv:      # 断线重连恢复的战斗: 直接接管
        q = targets[0]
        print(f"[Q{q}] --resume 接管进行中战斗", flush=True)
        result = play_battle()
        for x, y in [(3437, 1998), (1920, 2005), (2318, 1998)]:
            time.sleep(6)
            tap(x, y)
        time.sleep(7)
        if result == "win":
            led["bonus_cleared"][str(q)] = "full"
            tmp = LEDGER.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(led, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(LEDGER)
            print(f"[Q{q}] ✓ 满加成入账", flush=True)
        else:
            print(f"[Q{q}] {result}", flush=True)
        return
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
