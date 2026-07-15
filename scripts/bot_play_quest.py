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
            if time.time() - last_play < 1.5:
                continue
            # ── cost 读数(digit-OCR, 读不出退化节奏模式 fail-safe) ──
            cost_raw = run_digit_ocr(fr, (0.585, 0.895, 0.655, 0.968))
            cost = int(cost_raw) if cost_raw and cost_raw.isdigit() and \
                int(cost_raw) <= 10 else None
            # ── 选卡: conf 0.3(1号位小卡漏检根治)+cx 排序保卡位 ──
            cards = [(n, c, (x1 + x2) / 2, (y1 + y2) / 2)
                     for n, c, x1, y1, x2, y2, W, H in dets(avatar, fr, 0.30)
                     if (y1 + y2) / 2 / H > 0.78]
            if not cards:
                continue
            cards.sort(key=lambda k: k[2])
            cards.sort(key=lambda k: (k[0] == last_card,
                                      SKILLS.get(k[0], {}).get("cost") or 9))
            if cost is not None and cost >= 9:      # 满费防溢出: 强制最便宜
                cards.sort(key=lambda k:
                           SKILLS.get(k[0], {}).get("cost") or 9)
            n, cconf, kx, ky = cards[0]
            info = SKILLS.get(n, {})
            need = info.get("cost")
            if cost is not None and need is not None and cost < need:
                continue                            # 费不够不空点
            tgt_type = info.get("target", "enemy")
            # ── 释放: tap卡 → 0.8s → fr2(瞄准态判定 + ⭐最新目标坐标 —
            # 用旧帧坐标=放歪根因) → tap目标 → 卡消失=释放实锤 ──
            base_lum = float(fr.mean())
            tap(int(kx), int(ky))
            time.sleep(0.8)
            fr2 = adb.capture_frame()
            if fr2 is None:
                continue
            aimed = float(fr2.mean()) < base_lum * 0.80
            bd2 = dets(battle, fr2, 0.4)
            if tgt_type == "ally":
                # 血量: 我方框内 HSV 绿条宽/框宽 ≈ HP%(split_stacked 设施)
                import cv2 as _c
                best, best_hp = None, 2.0
                for bb in bd2:
                    if bb[0] != "我方":
                        continue
                    x1, y1, x2, y2 = map(int, bb[2:6])
                    crop = fr2[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    hsv = _c.cvtColor(crop, _c.COLOR_BGR2HSV)
                    g = ((hsv[..., 0] > 35) & (hsv[..., 0] < 60) &
                         (hsv[..., 1] > 110) & (hsv[..., 2] > 130))
                    cols = g.any(axis=0)
                    hp = cols.mean() if cols.any() else 1.0
                    if hp < best_hp:
                        best_hp, best = hp, ((x1 + x2) // 2, (y1 + y2) // 2)
                pool = [best] if best else []
                tgt_s = f"我方(HP{best_hp:.0%})" if best else "我方?"
            else:
                bosses = [((bb[2] + bb[4]) / 2, (bb[3] + bb[5]) / 2)
                          for bb in bd2 if bb[0] in BOSS_CLS]
                foes = [((bb[2] + bb[4]) / 2, (bb[3] + bb[5]) / 2)
                        for bb in bd2 if bb[0] == "敌方"]
                pool = bosses or foes
                tgt_s = "BOSS" if bosses else "敌"
            tx, ty = pool[0] if pool else (1920, 1080)
            tap(int(tx), int(ty))
            time.sleep(1.0)
            fr3 = adb.capture_frame()
            still = fr3 is not None and any(
                m == n for m, cc, x1, y1, x2, y2, W, H in
                dets(avatar, fr3, 0.30) if (y1 + y2) / 2 / H > 0.78)
            released = not still
            print(f"    ⚡{n}(需{need},费{cost if cost is not None else '?'})"
                  f"→{tgt_s} 瞄准={'Y' if aimed else 'N'} "
                  f"释放={'✓' if released else '✗'}", flush=True)
            if not released and aimed:
                adb._shell(f"input swipe {int(kx)} {int(ky)} "
                           f"{int(tx)} {int(ty)} 600")
                print("    ↪拖拽兜底", flush=True)
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
