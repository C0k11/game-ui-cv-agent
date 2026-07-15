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

    SKILLS = json.loads((ROOT / "data" / "skill_type_map.json")
                        .read_text(encoding="utf-8"))
    BOSS_CLS = {"Boss", "主教", "球", "黑白", "大蛇", "塞特的愤怒"}
    fly_dir = ROOT / "data" / "raw_images" / \
        ("run_" + time.strftime("%Y%m%d_%H%M%S") + "_botplay_clean")
    fly_dir.mkdir(parents=True, exist_ok=True)

    def play_battle() -> str:
        """控牌战斗 v2 (用户四条反馈 2026-07-15):
        ①1.2s 节奏(cost 不顶满) ②知识库轮换选卡(不再только3号位)
        ③点卡+点目标 adb 原子连发(一条 shell, 免中间抓帧 ~2s)
        ④Boss 类在场=集火; aim 路由: auto点卡即放/ally点我方/enemy点敌。
        全程 1fps 存干净帧进飞轮(补 DEFEAT 等新 cls 素材)。"""
        import cv2
        t0 = time.time()
        last_play = 0.0
        last_card = None
        self_state = {}          # 目标粘滞状态(last_tgt)
        fi = 0
        while time.time() - t0 < 300:
            fr = adb.capture_frame()
            if fr is None:
                continue
            cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1] \
                .tofile(str(fly_dir / f"frame_{fi:05d}.jpg"))
            fi += 1
            bd = dets(battle, fr, 0.5)
            bnames = {b[0] for b in bd}
            if "战斗胜利" in bnames:
                print(f"    ⭐胜利({time.time()-t0:.0f}s)", flush=True)
                return "win"
            if "自动战斗开启" in bnames:
                tap(3621, 2023)
                print("    AUTO→关(bot接管)", flush=True)
                time.sleep(1)
                continue
            in_battle = bool(bnames & {"我方", "敌方", "战斗暂停",
                                       "自动战斗关闭"})
            if not in_battle:
                du = dets(ui, fr, 0.5)
                if any(n == "确认键" for n, *_ in du) and \
                        time.time() - t0 > 30:
                    print(f"    battle域空+确认键({time.time()-t0:.0f}s)"
                          f" → 判败/结束", flush=True)
                    return "lose"
                continue
            # ── 选卡 v5: 亮卡=可放(游戏自己渲染的信号, 替代 cost OCR —
            # 灰卡去饱和, HSV sat 判) → 有亮卡立即放, cost 永不顶满 ──
            import cv2 as _c
            hsv_full = _c.cvtColor(fr, _c.COLOR_BGR2HSV)
            cards = []
            for n, c, x1, y1, x2, y2, W, H in dets(avatar, fr, 0.30):
                if (y1 + y2) / 2 / H <= 0.78:
                    continue
                sat = float(hsv_full[int(y1):int(y2),
                                     int(x1):int(x2), 1].mean())
                cards.append((n, c, (x1 + x2) / 2, (y1 + y2) / 2, sat))
            lit = [k for k in cards if k[4] > 70]     # 亮卡阈值(日志标定)
            if not lit:
                if cards and time.time() - last_play > 8:
                    print(f"    (全灰 sat={[round(k[4]) for k in cards]})",
                          flush=True)
                    last_play = time.time() - 6
                continue
            lit.sort(key=lambda k: k[2])
            lit.sort(key=lambda k: (k[0] == last_card,
                                    SKILLS.get(k[0], {}).get("cost") or 9))
            n, cconf, kx, ky, sat = lit[0]
            info = SKILLS.get(n, {})
            tgt_type = info.get("target", "enemy")

            role = info.get("role", "?")
            aoe = info.get("aoe", False)

            def ally_lowest(bdx, frx):
                best, best_hp = None, 2.0
                for bb in bdx:
                    if bb[0] != "我方":
                        continue
                    x1, y1, x2, y2 = map(int, bb[2:6])
                    crop = frx[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    hh = _c.cvtColor(crop, _c.COLOR_BGR2HSV)
                    g = ((hh[..., 0] > 35) & (hh[..., 0] < 60) &
                         (hh[..., 1] > 110) & (hh[..., 2] > 130))
                    cols = g.any(axis=0)
                    hp = cols.mean() if cols.any() else 1.0
                    if hp < best_hp:
                        best_hp = hp
                        best = ((x1 + x2) / 2, (y1 + y2) / 2)
                return best, best_hp

            def pick_enemy(bdx):
                """优先级(用户): BOSS > 框面积大(精英) > 粘滞; AOE=敌群重心"""
                bosses = [bb for bb in bdx if bb[0] in BOSS_CLS]
                foes = [bb for bb in bdx if bb[0] == "敌方"]
                if bosses:
                    bb = max(bosses,
                             key=lambda b: (b[4] - b[2]) * (b[5] - b[3]))
                    return (((bb[2] + bb[4]) / 2, (bb[3] + bb[5]) / 2),
                            "BOSS")
                if not foes:
                    return None, "?"
                if aoe and len(foes) >= 3:          # AOE→敌群重心
                    cx = sum((b[2] + b[4]) / 2 for b in foes) / len(foes)
                    cy = sum((b[3] + b[5]) / 2 for b in foes) / len(foes)
                    return (cx, cy), f"敌群x{len(foes)}"
                foes.sort(key=lambda b: -(b[4] - b[2]) * (b[5] - b[3]))
                if self_state.get("last_tgt"):      # 粘滞: 前2大里挑近的
                    lx, ly = self_state["last_tgt"]
                    for bb in foes[:2]:
                        px, py = (bb[2] + bb[4]) / 2, (bb[3] + bb[5]) / 2
                        if abs(px - lx) + abs(py - ly) < 300:
                            return (px, py), "敌(粘滞)"
                bb = foes[0]
                return (((bb[2] + bb[4]) / 2, (bb[3] + bb[5]) / 2), "敌(最大)")

            # ── 角色策略路由(用户: 按角色特性干活) ──
            if role == "Healer" or (tgt_type == "ally" and
                                    info.get("heal")):
                _, low_hp = ally_lowest(bd, fr)
                if low_hp > 0.55:
                    continue        # 没人残血, 奶卡留手
            base_lum = float(fr.mean())
            tap(int(kx), int(ky))
            time.sleep(0.7)
            fr2 = adb.capture_frame()
            if fr2 is None:
                continue
            aimed = float(fr2.mean()) < base_lum * 0.80
            bd2 = dets(battle, fr2, 0.4)
            if tgt_type == "ally":
                tgt, hp = ally_lowest(bd2, fr2)
                tgt_s = f"我方HP{hp:.0%}" if tgt else "我方?"
            else:
                tgt, tgt_s = pick_enemy(bd2)
            if tgt is None:
                tgt = (1920, 1080)
            tx, ty = int(tgt[0]), int(tgt[1])
            # ── 闭环拖拽释放: DOWN → 抓帧修正目标 → MOVE → UP(松手前圈
            # 一直跟到目标最新位置 — 治"松手吸到别人") ──
            adb._shell(f"input motionevent DOWN {tx} {ty}")
            fr3 = adb.capture_frame()
            if fr3 is not None:
                bd3 = dets(battle, fr3, 0.4)
                if tgt_type == "ally":
                    t2, _ = ally_lowest(bd3, fr3)
                else:
                    t2, _ = pick_enemy(bd3)
                if t2 is not None and abs(t2[0] - tx) + abs(t2[1] - ty) < 500:
                    tx, ty = int(t2[0]), int(t2[1])   # 跟到最新位置
            adb._shell(f"input motionevent MOVE {tx} {ty} && "
                       f"input motionevent UP {tx} {ty}")
            self_state["last_tgt"] = (tx, ty)
            time.sleep(0.9)
            fr4 = adb.capture_frame()
            still = fr4 is not None and any(
                m == n for m, cc, x1, y1, x2, y2, W, H in
                dets(avatar, fr4, 0.30) if (y1 + y2) / 2 / H > 0.78)
            print(f"    ⚡{n}[{role}{'|AOE' if aoe else ''}](sat{sat:.0f})"
                  f"→{tgt_s}@({tx},{ty}) 瞄准={'Y' if aimed else 'N'} "
                  f"释放={'✓' if not still else '✗'}", flush=True)
            last_card = n
            last_play = time.time()
        return "timeout"

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
