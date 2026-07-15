# -*- coding: utf-8 -*-
"""combat 2.0 v7 战斗大脑: scrcpy 感知线程 + 黑板 + 行为树控牌.

架构(playbook 4.8 定案, 不再打补丁):
  ScrcpyFeed(~25fps) → Perception 线程(battle+avatar 每帧, ui 低频)
  → Blackboard(全队HP/亮卡表/敌表/boss/亮度) → 行为树 tick(决策只读黑板)
  → 动作(ADB tap / motionevent 闭环拖拽, 拖拽中每步读黑板跟目标).

v6 的根性缺陷: ADB 1fps 链, 闭环拖拽中途 capture_frame 等 ~1s,
目标早跑了; v7 黑板帧龄 <150ms, 圈真正贴住目标.

行为树优先级(用户可直接改 RULES 顺序):
  急救(残血+奶卡) > 集火BOSS > AOE清群(敌≥3) > 辅助增益 > 单体循环 > 攒费.
坐标系: 黑板全归一化(0-1); 动作层乘 4K(3840x2160) 出 tap 坐标.
"""
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

BOSS_CLS = {"Boss", "主教", "球", "黑白", "大蛇", "塞特的愤怒"}
BATTLE_HUD = {"我方", "敌方", "战斗暂停", "自动战斗关闭", "自动战斗开启"}
CARD_Y = 0.78          # 手牌区: 归一化 cy > 0.78
LIT_SAT = 70           # 亮卡 HSV 饱和度阈值(v5 标定)
TAP_W, TAP_H = 3840, 2160   # input tap 的 4K 坐标系


@dataclass
class Snapshot:
    ts: float = 0.0
    seq: int = 0
    age: float = 9.9
    lum: float = 0.0
    in_battle: bool = False
    victory: bool = False
    auto_on: bool = False
    allies: list = field(default_factory=list)    # (cx,cy,hp) 归一化
    enemies: list = field(default_factory=list)   # (cls,cx,cy,area,is_boss)
    cards: list = field(default_factory=list)     # (name,cx,cy,sat,conf)
    ui_names: set = field(default_factory=set)    # 低频 ui 模型检出

    @property
    def boss(self):
        bs = [e for e in self.enemies if e[4]]
        return max(bs, key=lambda e: e[3]) if bs else None

    @property
    def lit(self):
        return [c for c in self.cards if c[3] > LIT_SAT]


def _hp_of(crop_bgr) -> float:
    """我方框内绿血条宽比 ≈ HP%. 绿=HSV(35-60, S>110, V>130)."""
    if crop_bgr.size == 0:
        return 1.0
    h = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    g = ((h[..., 0] > 35) & (h[..., 0] < 60) &
         (h[..., 1] > 110) & (h[..., 2] > 130))
    cols = g.any(axis=0)
    return float(cols.mean()) if cols.any() else 1.0


class Perception(threading.Thread):
    """scrcpy 帧 → YOLO → 黑板. battle+avatar 每帧, ui 每 20 帧(判败兜底)."""

    def __init__(self, feed, battle, avatar, ui, flywheel_dir=None):
        super().__init__(daemon=True)
        self.feed = feed
        self.battle, self.avatar, self.ui = battle, avatar, ui
        self.fly_dir = flywheel_dir
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self.fps = 0.0

    def snapshot(self) -> Snapshot:
        with self._lock:
            s = self._snap
            return Snapshot(s.ts, s.seq, time.time() - s.ts if s.ts else 9.9,
                            s.lum, s.in_battle, s.victory, s.auto_on,
                            list(s.allies), list(s.enemies), list(s.cards),
                            set(s.ui_names))

    def stop(self):
        self._stop.set()

    def _dets(self, model, fr, conf):
        r = model.predict(fr, conf=conf, imgsz=960, verbose=False)[0]
        H, W = fr.shape[:2]
        return [(model.names[int(b.cls[0])], float(b.conf[0]),
                 float(b.xyxy[0][0]) / W, float(b.xyxy[0][1]) / H,
                 float(b.xyxy[0][2]) / W, float(b.xyxy[0][3]) / H)
                for b in (r.boxes or [])]

    def run(self):
        n_frames, t_fps, last_seq, last_fly = 0, time.time(), 0, 0.0
        ui_names, fly_i = set(), 0
        while not self._stop.is_set():
            fr, age, seq = self.feed.latest()
            if fr is None or seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq
            H, W = fr.shape[:2]
            hsv = None
            bd = self._dets(self.battle, fr, 0.5)
            bnames = {b[0] for b in bd}
            allies, enemies = [], []
            for n, c, x1, y1, x2, y2 in bd:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if n == "我方":
                    crop = fr[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)]
                    allies.append((cx, cy, _hp_of(crop)))
                elif n == "敌方" or n in BOSS_CLS:
                    enemies.append((n, cx, cy, (x2 - x1) * (y2 - y1),
                                    n in BOSS_CLS))
            cards = []
            for n, c, x1, y1, x2, y2 in self._dets(self.avatar, fr, 0.30):
                cy = (y1 + y2) / 2
                if cy <= CARD_Y:
                    continue
                if hsv is None:
                    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
                sat = float(hsv[int(y1 * H):int(y2 * H),
                                int(x1 * W):int(x2 * W), 1].mean())
                cards.append((n, (x1 + x2) / 2, cy, sat, c))
            cards.sort(key=lambda k: k[1])
            n_frames += 1
            if n_frames % 20 == 0:       # ui 低频: 判败/结算兜底
                ui_names = {d[0] for d in self._dets(self.ui, fr, 0.5)}
                self.fps = 20 / max(time.time() - t_fps, 1e-6)
                t_fps = time.time()
            if self.fly_dir and time.time() - last_fly > 1.0:  # 飞轮 1fps
                cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1] \
                    .tofile(str(self.fly_dir / f"frame_{fly_i:05d}.jpg"))
                fly_i += 1
                last_fly = time.time()
            with self._lock:
                s = self._snap
                s.ts, s.seq = time.time(), seq
                s.lum = float(fr[::8, ::8].mean())
                s.victory = s.victory or "战斗胜利" in bnames
                s.auto_on = "自动战斗开启" in bnames
                s.in_battle = bool(bnames & BATTLE_HUD)
                s.allies, s.enemies, s.cards = allies, enemies, cards
                s.ui_names = ui_names


class CombatBrain:
    """行为树控牌. shell = AdbInput._shell 注入; skills = skill_type_map."""

    def __init__(self, perception: Perception, shell, skills: dict,
                 log=print, adb_capture=None):
        self.p = perception
        self.shell = shell
        self.skills = skills
        self.log = log
        self.adb_capture = adb_capture   # ADB 抓帧(判败二次确认用)
        self.last_tgt = None        # 粘滞(归一化)
        self.last_card = None
        self.fail_n = {}            # 卡名→连续释放失败次数
        self.cooldown = {}          # 卡名→冷却截止时刻(防误判亮卡死循环)

    # ── 目标函数(闭环拖拽中反复调用, 每次读最新黑板) ──────────────
    def tgt_boss(self, s: Snapshot):
        b = s.boss
        if b:
            return (b[1], b[2]), "BOSS:" + b[0]
        return self.tgt_enemy(s)      # boss 没了(击杀/漏检)退单体

    def tgt_enemy(self, s: Snapshot):
        foes = sorted(s.enemies, key=lambda e: -e[3])
        if not foes:
            return None, "?"
        if self.last_tgt:             # 粘滞: 前2大里挑 300px(≈0.08) 内的
            lx, ly = self.last_tgt
            for e in foes[:2]:
                if abs(e[1] - lx) + abs(e[2] - ly) < 0.08:
                    return (e[1], e[2]), "敌(粘滞)"
        e = foes[0]
        return (e[1], e[2]), "敌(最大)"

    def tgt_pack(self, s: Snapshot):
        foes = [e for e in s.enemies if not e[4]]
        if len(foes) < 2:
            return self.tgt_enemy(s)
        cx = sum(e[1] for e in foes) / len(foes)
        cy = sum(e[2] for e in foes) / len(foes)
        return (cx, cy), f"敌群x{len(foes)}"

    def tgt_ally_low(self, s: Snapshot):
        if not s.allies:
            return None, "我方?"
        a = min(s.allies, key=lambda x: x[2])
        return (a[0], a[1]), f"我方HP{a[2]:.0%}"

    def tgt_ally_center(self, s: Snapshot):
        if not s.allies:
            return None, "我方?"
        cx = sum(a[0] for a in s.allies) / len(s.allies)
        cy = sum(a[1] for a in s.allies) / len(s.allies)
        return (cx, cy), f"我方群x{len(s.allies)}"

    # ── 行为树: 从上往下第一个命中的规则执行(顺序=优先级, 直接改) ──
    def decide(self, s: Snapshot):
        """返回 (规则名, card, target_fn) 或 None(攒费).
        ⚠sat 判亮对部分卡失真(优香卡面天然高饱和, 亮灰 sat 重叠,
        2026-07-15 Q11 实锤) → 释放失败退避冷却是主防线, 不追求判准."""
        now = time.time()
        lit = [c for c in s.lit if self.cooldown.get(c[0], 0) < now]
        if not lit:
            return None

        def info(c):
            return self.skills.get(c[0], {})

        heal = [c for c in lit if info(c).get("heal") or
                info(c).get("role") == "Healer"]
        low = min((a[2] for a in s.allies), default=1.0)
        if heal and low < 0.55:
            return "急救", heal[0], self.tgt_ally_low
        # enemy 卡只在有敌检出时选(空场放 EX = 纯浪费, 屏心兜底只留给
        # 闭环中目标瞬时丢失的场景)
        ec = [c for c in lit if info(c).get("target") == "enemy"] \
            if s.enemies else []
        if s.boss and ec:                     # 大招砸 boss: 高费优先
            ec.sort(key=lambda c: -(info(c).get("cost") or 0))
            return "集火BOSS", ec[0], self.tgt_boss
        aoe_e = [c for c in ec if info(c).get("aoe")]
        if aoe_e and len([e for e in s.enemies if not e[4]]) >= 3:
            return "AOE清群", aoe_e[0], self.tgt_pack
        if ec:                                # 输出优先于 buff(cost 先喂主C)
            ec.sort(key=lambda c: ((info(c).get("cost") or 9),
                                   c[0] == self.last_card))
            return "单体循环", ec[0], self.tgt_enemy
        sup = [c for c in lit if info(c).get("target") in ("ally", "self")
               and c not in heal]
        if sup and s.enemies:                 # buff 别浪费在空场
            c = sup[0]
            fn = self.tgt_ally_center if info(c).get("aoe") \
                else self.tgt_ally_low
            return "辅助增益", c, fn
        return None

    # ── 释放(闭环拖拽: DOWN → 每步读黑板 MOVE 跟目标 → UP) ──────────
    def play(self, rule: str, card, target_fn) -> bool:
        name = card[0]
        inf = self.skills.get(name, {})
        kx, ky = int(card[1] * TAP_W), int(card[2] * TAP_H)
        base_lum = self.p.snapshot().lum
        self.shell(f"input tap {kx} {ky}")
        if inf.get("aim") == "auto":          # 点卡即放(阳葵类)
            time.sleep(0.6)
            ok = self._card_gone(name)
            self.log(f"    ⚡[{rule}]{name}(auto, sat{card[3]:.0f}) "
                     f"释放={'✓' if ok else '✗'}")
            self.last_card = name
            self._book(name, ok)
            return ok
        time.sleep(0.55)                      # 等瞄准态起
        s = self.p.snapshot()
        aimed = s.lum < base_lum * 0.85
        tgt, desc = target_fn(s)
        if tgt is None:
            tgt, desc = (0.5, 0.5), "屏心兜底"
        tx, ty = int(tgt[0] * TAP_W), int(tgt[1] * TAP_H)
        self.shell(f"input motionevent DOWN {tx} {ty}")
        for _ in range(3):                    # 闭环: 圈全程贴目标
            time.sleep(0.12)
            t2, d2 = target_fn(self.p.snapshot())
            if t2 and abs(t2[0] * TAP_W - tx) + abs(t2[1] * TAP_H - ty) < 500:
                tx, ty = int(t2[0] * TAP_W), int(t2[1] * TAP_H)
                desc = d2
            self.shell(f"input motionevent MOVE {tx} {ty}")
        self.shell(f"input motionevent UP {tx} {ty}")
        self.last_tgt = (tx / TAP_W, ty / TAP_H)
        time.sleep(0.5)
        ok = self._card_gone(name)
        self.log(f"    ⚡[{rule}]{name}[{inf.get('role', '?')}"
                 f"{'|AOE' if inf.get('aoe') else ''}](sat{card[3]:.0f})"
                 f"→{desc}@({tx},{ty}) 瞄准={'Y' if aimed else 'N'} "
                 f"释放={'✓' if ok else '✗'}")
        self.last_card = name
        self._book(name, ok)
        return ok

    def _book(self, name: str, ok: bool):
        """失败退避: 连续 2 次释放✗ → 冷却 12s(等 cost 回复).
        sat 判亮对部分卡失真(亮灰重叠), 退避让误判无法卡死决策循环."""
        if ok:
            self.fail_n[name] = 0
            self.cooldown.pop(name, None)
            return
        self.fail_n[name] = self.fail_n.get(name, 0) + 1
        if self.fail_n[name] >= 2:
            self.cooldown[name] = time.time() + 12.0
            self.fail_n[name] = 0

    def _card_gone(self, name: str) -> bool:
        """释放实锤 = 卡从手牌消失(等黑板出新帧再判)."""
        seq0 = self.p.snapshot().seq
        for _ in range(12):
            s = self.p.snapshot()
            if s.seq > seq0:
                return all(c[0] != name for c in s.cards)
            time.sleep(0.05)
        return False

    # ── 主战斗循环 ──────────────────────────────────────────────
    def run_battle(self, timeout_s: float = 300.0,
                   auto_xy=(3621, 2023)) -> str:
        t0 = time.time()
        last_status, empty_since = 0.0, None
        while time.time() - t0 < timeout_s:
            s = self.p.snapshot()
            if s.age > 5.0:               # feed 假死: 不拿旧状态做决策
                if time.time() - last_status > 5:
                    self.log(f"    ⚠帧龄{s.age:.1f}s — 感知断流, 等恢复"
                             f"(feed重启x{getattr(self.p.feed, 'restarts', '?')})")
                    last_status = time.time()
                time.sleep(1.0)
                continue
            if s.victory:
                self.log(f"    ⭐胜利({time.time() - t0:.0f}s)")
                return "win"
            if s.auto_on:                     # 状态门: 检出开启才点(接管)
                self.shell(f"input tap {auto_xy[0]} {auto_xy[1]}")
                self.log("    AUTO→关(bot接管)")
                time.sleep(1.0)
                continue
            if not s.in_battle:
                empty_since = empty_since or time.time()
                if ("确认键" in s.ui_names and time.time() - t0 > 30
                        and time.time() - empty_since > 6):
                    # ADB 干净帧二次确认(独立于 scrcpy 断流的第二链路):
                    # Q12 实锤胜利帧落在断流窗口 → 胜局被误判败
                    if self.adb_capture is not None:
                        fr = self.adb_capture()
                        if fr is not None:
                            bn = {d[0] for d in self.p._dets(
                                self.p.battle, fr, 0.5)}
                            if "战斗胜利" in bn:
                                self.log(f"    ⭐胜利(ADB 二次确认, "
                                         f"{time.time() - t0:.0f}s)")
                                return "win"
                            if bn & BATTLE_HUD:
                                self.log("    (ADB 帧仍在战斗 — scrcpy"
                                         "断流误报, 继续)")
                                empty_since = None
                                continue
                    self.log(f"    battle域空+确认键({time.time() - t0:.0f}s)"
                             f" → 判败/结束")
                    return "lose"
                time.sleep(0.4)
                continue
            empty_since = None
            d = self.decide(s)
            if d is None:
                if time.time() - last_status > 8:
                    sats = [round(c[3]) for c in s.cards]
                    self.log(f"    (攒费 手牌sat={sats} 敌x{len(s.enemies)}"
                             f" 感知{self.p.fps:.0f}fps 帧龄{s.age:.2f}s)")
                    last_status = time.time()
                time.sleep(0.25)
                continue
            self.play(*d)
        return "timeout"
