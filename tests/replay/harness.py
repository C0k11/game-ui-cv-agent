# -*- coding: utf-8 -*-
"""Trajectory replay harness — 把落盘的 tick json 还原成 ScreenState 喂回 skill。

为什么要它(2026-07-25):
  这个项目所有 skill 逻辑验证都靠 live 跑游戏 —— 一次 daily 两小时、消耗真实
  AP/票/货币, 而且只覆盖当天恰好走到的那条路径。同时盘上躺着 754 run / 90,396
  tick 的全量决策录像(每 tick 落 yolo_boxes + ocr_boxes + 干净帧)。用它做离线
  回归, 改一行代码 5 分钟就能知道"这一改动会改变历史上哪些 tick 的决策"。

⛔**做 diff replay, 不做 golden replay**(用户 2026-07-24 定):
  录像里的 `action` 字段是**当时的 bot 实际干的事**, 里面含 bot 自己犯的错
  (Challenge 假阳性 / 假账 / 误入活动)。拿它当期望值 = 把 bug 固化成测试。
  正确用法: 同一批 tick 在 **改动前的代码** 和 **改动后的代码** 上各跑一遍,
  只看**决策发生变化的 tick**, 人去判断每处变化是修好了还是弄坏了。录制的
  action 只作为"当时发生了什么"的旁注, 不是断言目标。

  副作用: 垃圾 tick(空检出/老架构废类名, 实测占 65%)对 diff 天然免疫 ——
  两边代码在同一张垃圾帧上都返回同样的 wait, diff 为空, 不产生噪声。

重建保真度(逐字段核对 brain/skills/base.py 的 ScreenState):
  ocr_boxes / yolo_boxes / image_w / image_h  ✓ 原样落盘
  timestamp                                    ✓ 由 ts 字符串解析
  screenshot_path / frame                      ✓ 同名 tick_NNNN.jpg 在盘上(懒加载)
  YoloBox.model_tag                            ~ 未落盘, 按 cls id 分段还原
                                                 (0-142/395-475=ui, 143-394=avatar,
                                                  476-484=battle) —— 与 pipeline
                                                 的真实来源一致
  fresh_boxes/fresh_frame/fresh_ts             ✗ 未落盘 → None/0.0。用到它们的
                                                 skill(event_quest 原子 banner)
                                                 在 replay 下会走 null-check 分支,
                                                 这类判定不能靠 replay 验证。
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAJ_DIR = os.path.join(ROOT, "data", "trajectories")
_MASTER_YAML = r"D:/Project/ml_cache/models/yolo/dataset/ui_v2/data.yaml"

# master 类表分段 → 产出该 cls 的模型(见 memory val_set_crisis: 485 是三模型共用表)
_SEG = ((0, 142, "ui"), (143, 394, "avatar"), (395, 475, "ui"), (476, 484, "battle"))

_NAME2TAG: Optional[Dict[str, str]] = None
_NAME2ID: Dict[str, int] = {}


def _name_to_tag() -> Dict[str, str]:
    """cls_name → model_tag, 从 master 类表按 id 分段推。顺带建 cls_name → cls_id
    (tick json 只落了类名, YoloBox 要 id)。表读不到就返回空 dict —— model_tag
    全空只影响 cafe/schedule 的头像过滤, cls_id 退化成 -1 只影响按 id 取类的
    代码(全仓用的是 cls_name)。"""
    global _NAME2TAG
    if _NAME2TAG is not None:
        return _NAME2TAG
    out: Dict[str, str] = {}
    try:
        with open(_MASTER_YAML, encoding="utf-8") as f:
            in_names = False
            for line in f:
                if line.startswith("names:"):
                    in_names = True
                    continue
                if not in_names:
                    continue
                s = line.strip()
                if not s or ":" not in s:
                    continue
                idx, _, nm = s.partition(":")
                try:
                    i = int(idx.strip())
                except ValueError:
                    break
                nm = nm.strip().strip("'\"")
                _NAME2ID[nm] = i
                for a, b, tag in _SEG:
                    if a <= i <= b:
                        out[nm] = tag
                        break
    except Exception:
        pass
    _NAME2TAG = out
    return out


@dataclass
class ReplayTick:
    """一条录像 tick 的原始记录 + 重建出的 ScreenState。"""
    run: str
    tick: int
    path: str
    skill: str
    sub_state: str
    skill_ticks: int
    recorded_action: Dict[str, Any]
    raw: Dict[str, Any]

    @property
    def jpg(self) -> str:
        return self.path[:-5] + ".jpg"


def screen_from_tick(raw: Dict[str, Any], *, jpg_path: str = "",
                     load_frame: bool = False):
    """tick json → ScreenState。load_frame=True 时把同名 jpg 读成 BGR 挂上去
    (数字 OCR 类逻辑需要; 纯 cls 判定不需要, 省掉解码开销)。"""
    from brain.skills.base import OcrBox, ScreenState, YoloBox

    tag = _name_to_tag()
    yb: List[YoloBox] = []
    for b in raw.get("yolo_boxes") or []:
        nm = b.get("cls", "")
        yb.append(YoloBox(
            cls_id=_NAME2ID.get(nm, -1),
            cls_name=nm,
            confidence=float(b.get("conf", 0.0)),
            x1=float(b.get("x1", 0.0)), y1=float(b.get("y1", 0.0)),
            x2=float(b.get("x2", 0.0)), y2=float(b.get("y2", 0.0)),
            model_tag=tag.get(nm, ""),
        ))
    ob: List[OcrBox] = []
    for b in raw.get("ocr_boxes") or []:
        ob.append(OcrBox(
            text=b.get("text", ""),
            confidence=float(b.get("conf", b.get("confidence", 0.0))),
            x1=float(b.get("x1", 0.0)), y1=float(b.get("y1", 0.0)),
            x2=float(b.get("x2", 0.0)), y2=float(b.get("y2", 0.0)),
        ))

    ts = 0.0
    try:
        ts = datetime.strptime(str(raw.get("ts", "")),
                               "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        ts = time.time()

    frame = None
    if load_frame and jpg_path and os.path.exists(jpg_path):
        import cv2
        frame = cv2.imread(jpg_path)

    return ScreenState(
        ocr_boxes=ob,
        yolo_boxes=yb,
        image_w=int(raw.get("image_w", 0) or 0),
        image_h=int(raw.get("image_h", 0) or 0),
        screenshot_path=jpg_path,
        timestamp=ts,
        frame=frame,
    )


def iter_run(run_dir: str, *, skill: Optional[str] = None) -> Iterator[ReplayTick]:
    """按 tick 序遍历一个 run。skill 给定时只产出该 skill 的 tick。"""
    run = os.path.basename(run_dir)
    for p in sorted(glob.glob(os.path.join(run_dir, "tick_*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue
        sk = (raw.get("skill") or "").strip()
        if skill is not None and sk != skill:
            continue
        yield ReplayTick(
            run=run, tick=int(raw.get("tick", 0) or 0), path=p, skill=sk,
            sub_state=(raw.get("sub_state") or "").strip(),
            skill_ticks=int(raw.get("skill_ticks", 0) or 0),
            recorded_action=raw.get("action") or {},
            raw=raw,
        )


def find_runs(*, since: str = "", skill: Optional[str] = None,
              limit: int = 0) -> List[str]:
    """列出 run 目录。since='20260720' 按目录名日期过滤; skill 给定时只留
    真正含该 skill tick 的 run(先扫首尾几个 tick, 便宜)。"""
    runs = sorted(glob.glob(os.path.join(TRAJ_DIR, "run_*")))
    runs = [r for r in runs if os.path.isdir(r)]
    if since:
        runs = [r for r in runs if os.path.basename(r)[4:12] >= since]
    if skill:
        keep = []
        for r in runs:
            for t in iter_run(r, skill=skill):
                keep.append(r)
                break
        runs = keep
    if limit:
        runs = runs[-limit:]
    return runs


# ── 决策 trace ────────────────────────────────────────────────────────────

_ACTION_KEYS = ("action", "x", "y", "x2", "y2", "duration_ms", "key")


def action_signature(act: Dict[str, Any]) -> Tuple:
    """动作的可比较指纹。**不含 reason** —— reason 是给人看的说明文字, 改文案
    不该算决策变化; 坐标/类型/时长才是真正落到屏幕上的东西。"""
    if not isinstance(act, dict):
        return ("<non-dict>",)
    out = []
    for k in _ACTION_KEYS:
        v = act.get(k)
        if isinstance(v, float):
            v = round(v, 4)
        out.append(v)
    return tuple(out)


def replay_skill(skill_factory, ticks: List[ReplayTick], *,
                 mode: str = "stateless", load_frame: bool = False
                 ) -> List[Dict[str, Any]]:
    """把一串 tick 喂给 skill, 返回每 tick 的决策记录。

    mode:
      'stateless' (默认, 推荐) — 每个 tick 用**全新 skill 实例**, 并把录像里的
          sub_state / skill_ticks 灌回去再 tick()。每个 tick 相互独立, 不会因为
          某一步决策变了就让后面全部漂移。做 A/B diff 时两边状态构造完全一致,
          所以比较是公平的。代价: skill 的私有 latch(_bought/_stage/...)是默认
          初值, 与真实历史不同 —— 所以**不能**拿它跟录制 action 对齐, 只能
          A/B 互比(这正是 diff replay 的用法)。
      'sequential' — 单个实例顺序吃完整段, 状态自然演化。更接近真实, 但一旦
          决策与历史分叉, 后面的帧就不再对应了 → 只看"第一处分叉在哪"。
    """
    out: List[Dict[str, Any]] = []
    seq_skill = skill_factory() if mode == "sequential" else None
    if seq_skill is not None:
        seq_skill.reset()

    for rt in ticks:
        if mode == "sequential":
            sk = seq_skill
        else:
            sk = skill_factory()
            sk.reset()
            sk.sub_state = rt.sub_state
            sk.ticks = rt.skill_ticks
        screen = screen_from_tick(rt.raw, jpg_path=rt.jpg, load_frame=load_frame)
        err = ""
        try:
            act = sk.tick(screen)
        except Exception as e:                                   # noqa: BLE001
            act = {"action": "<exception>", "reason": repr(e)}
            err = f"{type(e).__name__}: {e}"
        out.append({
            "run": rt.run, "tick": rt.tick,
            "in_sub_state": rt.sub_state, "in_ticks": rt.skill_ticks,
            "out_sub_state": getattr(sk, "sub_state", ""),
            "sig": list(action_signature(act)),
            "reason": str(act.get("reason", ""))[:120],
            "recorded_sig": list(action_signature(rt.recorded_action)),
            "recorded_reason": str(rt.recorded_action.get("reason", ""))[:120],
            "error": err,
        })
    return out
