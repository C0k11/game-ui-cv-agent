# -*- coding: utf-8 -*-
"""活动加成重打 + Q12 扫荡 + 台账 (2026-07-15, 用户指令三件套).

A. 部队2(加成 farm 队)把 Q1-12 每关重打一遍 → 该关扫荡永久享受加成
   (上期 Q09 弱队烂记录教训); 台账 data/event_bonus_state_midnight.json
   记 bonus_cleared, 打过的跳过(可断点续跑)。
B. 台账 12/12 → Q12 扫荡链吃剩余 AP: MAX→掃蕩開始→确认框**结构白名单闸**
   (检出取消+确认 且 body 无 stepper/体力 → 才点确认; 否则取消+停,
   money fail-closed)。
路由: 入场键 cls + 行关号 digit-OCR(实测 '01'-'03' 全对), 不赌列表位置。
用法: py -u scripts/event_bonus_rerun.py
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
TOTAL_Q = 12


def load_models():
    reg = json.loads((ROOT / "data" / "model_registry.json")
                     .read_text(encoding="utf-8"))
    from ultralytics import YOLO
    ui = YOLO(reg["ui"]["versions"][reg["ui"]["active"]]["path"])
    vers = reg["battle_heads"]["versions"]
    vn = max((v for v in vers if re.fullmatch(r"v\d+", v)),
             key=lambda x: int(x[1:]))
    return ui, YOLO(vers[vn]["path"])


def dets(model, fr, conf=0.5):
    H, W = fr.shape[:2]
    r = model.predict(fr, conf=conf, imgsz=960, verbose=False)[0]
    return [(model.names[int(b.cls[0])], float(b.conf[0]),
             *(float(v) for v in b.xyxy[0]), W, H) for b in (r.boxes or [])]


def ledger_load():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"event": "秘密的午夜派對", "expire": "2026-07-21",
            "bonus_cleared": {}}


def ledger_save(d):
    tmp = LEDGER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(LEDGER)


def main():
    ui, battle = load_models()
    adb = AdbInput()
    adb.connect()
    tap = lambda x, y: adb._shell(f"input tap {x} {y}")  # noqa: E731
    led = ledger_load()

    def ensure_quest_tab():
        """⛔Challenge 误入防线(2026-07-15 实锤: 盲 tab tap 点空停在 Challenge,
        关号 OCR 读的是 Challenge 01-03)。正锚=「活动quest_已选择」(cls 100)
        检出才放行; 检出「活动quest」(94, 未选中)→ 点它的 box 切换再验。
        fail-closed: 6 次不成 → None(绝不在未知 tab 上点入场)。"""
        for _ in range(6):
            fr = adb.capture_frame()
            if fr is None:
                continue
            d = dets(ui, fr, 0.35)
            names = {n for n, *_ in d}
            # 正锚①活动quest_已选择(实测在部分列表状态漏检) ②关卡得星_N:
            # Quest 关已全3星 / Challenge 全0星(用户抄轴前不打) → 视野有
            # 得星徽章=必在 Quest 列表。⚠此假设生命周期=Challenge 未打期间。
            if "活动quest_已选择" in names or \
                    any(n.startswith("关卡得星") for n in names):
                return fr
            tab = next((x for x in d if x[0] == "活动quest"), None)
            if tab is not None:
                _, _, x1, y1, x2, y2, W, H = tab
                tap(int((x1 + x2) / 2), int((y1 + y2) / 2))
                print("    tab≠Quest → 点「活动quest」tab 切换", flush=True)
            time.sleep(3)
        return None

    def list_rows(fr):
        """[(关号int|None, cy_norm)] 按 cy 升序"""
        rows = []
        for n, c, x1, y1, x2, y2, W, Hh in dets(ui, fr, 0.5):
            if n == "入场键":
                cy = (y1 + y2) / 2 / Hh
                num = run_digit_ocr(fr, (0.542, cy - 0.028, 0.60, cy + 0.028))
                q = int(num) if num and num.isdigit() else None
                rows.append((q, cy))
        return sorted(rows, key=lambda r: r[1])

    def formation_and_sortie() -> bool:
        """编队页感知驱动(2026-07-15 用户纠偏: 不许无脑出击):
        逐帧 YOLO → 2部队(cls 126)未选中→点; 2部队高亮(120)→先「快速编辑」
        (121, 自动编队保加成满配)→(确认弹窗有则点)→「出击」(124)。"""
        auto_done = False
        for _ in range(14):
            fr = adb.capture_frame()
            if fr is None:
                continue
            d = dets(ui, fr, 0.5)
            names = {n for n, *_ in d}

            def box(nm):
                x = next((b for b in d if b[0] == nm), None)
                return (int((x[2] + x[4]) / 2), int((x[3] + x[5]) / 2)) \
                    if x else None

            if "2部队高亮" in names:
                if not auto_done:
                    p = box("快速编辑")
                    if p:
                        tap(*p)
                        auto_done = True
                        print("    快速编辑(自动编队)", flush=True)
                        time.sleep(3)
                        continue
                p = box("出击")
                if p:
                    tap(*p)
                    print("    出击(部队2+自动编队)", flush=True)
                    return True
            elif "确认键" in names and auto_done:   # 快速编辑确认弹窗
                p = box("确认键")
                tap(*p)
                time.sleep(2.5)
                continue
            elif "2部队" in names:
                p = box("2部队")
                tap(*p)
                print("    切部队2", flush=True)
                time.sleep(2.5)
            time.sleep(1.2)
        print("    ⛔编队页感知循环 14 轮未完成 — 停")
        return False

    def fight_and_settle() -> bool:
        t0 = time.time()
        while time.time() - t0 < 240:
            fr = adb.capture_frame()
            if fr is None:
                continue
            names = {d[0] for d in dets(battle, fr, 0.5)}
            if "自动战斗关闭" in names:
                tap(3621, 2023)
            if "战斗胜利" in names:
                print(f"    胜利({time.time()-t0:.0f}s)", flush=True)
                for x, y in [(3437, 1998), (1920, 2005), (2318, 1998)]:
                    time.sleep(6)
                    tap(x, y)
                time.sleep(8)
                return True
        return False

    # ── A. 加成重打 ──
    for safety in range(40):
        todo = [q for q in range(1, TOTAL_Q + 1)
                if not led["bonus_cleared"].get(str(q))]
        if not todo:
            break
        target = todo[0]
        fr = ensure_quest_tab()
        if fr is None:
            print("    ⛔Quest tab 正锚(活动quest_已选择)拿不到 — 停")
            return
        rows = list_rows(fr)
        nums = [q for q, _ in rows if q]
        print(f"[A] 目标Q{target} 视野{nums}", flush=True)
        hit = next(((q, cy) for q, cy in rows if q == target), None)
        if hit is None:
            if not nums:
                print("    列表无检出 — 停(人工看)")
                return
            if target < min(nums):
                adb._shell("input swipe 2760 700 2760 1500 500")
            else:
                adb._shell("input swipe 2760 1500 2760 700 500")
            time.sleep(2.5)
            continue
        tap(3340, int(hit[1] * 2160))
        time.sleep(7)
        tap(2803, 1620)          # 任務開始
        time.sleep(9)
        if not formation_and_sortie():
            return
        print(f"    Q{target} 已出击", flush=True)
        if fight_and_settle():
            led["bonus_cleared"][str(target)] = True
            ledger_save(led)
            print(f"  ✓ Q{target} 加成记录入账", flush=True)
        else:
            print(f"  ✗ Q{target} 战斗超时 — 停")
            return

    print("[A] 12关加成全记录 ✓", flush=True)

    # ── B. Q12 扫荡吃剩余 AP ──
    for _ in range(6):               # 找到 Q12 行
        fr = ensure_quest_tab()
        if fr is None:
            print("[B] ⛔Quest tab 正锚拿不到 — 停")
            return
        rows = list_rows(fr)
        hit = next(((q, cy) for q, cy in rows if q == TOTAL_Q), None)
        if hit:
            tap(3340, int(hit[1] * 2160))
            time.sleep(7)
            break
        adb._shell("input swipe 2760 1500 2760 700 500")
        time.sleep(2.5)
    else:
        print("[B] 找不到Q12 — 停")
        return
    tap(3245, 900)               # 扫荡 MAX
    time.sleep(3)
    tap(2803, 1220)              # 掃蕩開始
    time.sleep(5)
    fr = adb.capture_frame()
    d = dets(ui, fr, 0.20)       # 守卫地板 conf
    names = {x[0] for x in d}
    body_bad = [n for n, c, x1, y1, x2, y2, W, H in d
                if y1 / H > 0.12 and n in
                ("加号", "MAX_可点击", "MIN_灰色", "体力")]
    if {"取消键", "确认键"} <= names and not body_bad:
        tap(2300, 1570)          # 純AP確認 (白名单过闸)
        print("[B] 扫荡确认(纯AP闸过) ✓", flush=True)
        time.sleep(6)
        tap(1920, 1900)          # 结果窗 TOUCH
        time.sleep(3)
        tap(3552, 465)           # 叉(兜底)
    else:
        if "取消键" in names:
            tap(1540, 1570)
        print(f"[B] ⛔确认框结构闸拦截(body={body_bad}) — 不扫, 人工看",
              flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
