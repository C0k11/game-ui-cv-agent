# -*- coding: utf-8 -*-
"""YOLO 检测层 —— 实测 **23ms**（ui, imgsz=960, RTX4090）。

三条硬规矩:
  1. **模型只从 registry 解析，解析不出就 raise**（绝不静默回落老权重 ——
     "改了 registry 没效果、旧模型还在跑"就是这么来的）。
  2. **`_废弃` 前缀的类在这一层直接丢弃**（README §A11）。`_classes.txt` 是
     按行号索引的，废案类不能删行，只能改名；那就在检测出口处统一过滤，
     上层永远看不见废案，也就不会再有人拿废案当"漏训"去补数据。
  3. **imgsz 必须按模型给**（ui_v2+ 训练在 960；实测 @1920  0 检出）。

别对 ui 权重跑 `strip_optimizer`: YOLO26 是 NMS-free，去重靠 o2o 头学出来，
   FP16 量化会改变它 —— 同一轮权重 FP32 双框 206  FP16 550（2026-08-07 实锤）。
   registry 里 v15 指的是**未 strip** 的 `best_real.pt`，别"顺手优化"。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from routing_v2.percept.observe import Box

_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = _ROOT / "data" / "model_registry.json"

# 每个模型的推理尺寸。ui 必须 960 —— 训练就在 960，@1920 实测零检出。
_IMGSZ = {"ui": 960, "avatar": 960, "battle": 960, "emoticon": 640}
# 每个模型的检出下限。ui 取 0.20：弱类（免费 live ~0.18-0.30）在这个下限才出得来，
# 上层判据自己再按 cls 收紧（金钱链一律 ≥0.45）。
_CONF = {"ui": 0.20, "avatar": 0.35, "battle": 0.45, "emoticon": 0.15}

_models: Dict[str, object] = {}
_names: Dict[str, Dict[int, str]] = {}
_lock = threading.Lock()
_dropped_deprecated = 0

_CLASSES_TXT = _ROOT / "data" / "raw_images" / "_classes.txt"


def _name_table(model, tag: str) -> Dict[int, str]:
    """类名表。**ui 优先用 `_classes.txt`（按行号索引）而不是权重里烧进去的名字。**

    为什么: 权重里的 names 是**训练那一刻**的快照。之后我们又把 9 个废案类
       改名加了 `_废弃N_` 前缀（`_classes.txt` 是按行号索引的，废案只能改名不能
       删行）。只信权重的话，这些废案还会照常吐出来，然后又有人拿它们当
       "有 cls 但检不到 = 漏训"去补数据 —— §A11 那个坑我已经踩过一次。
       nc 对得上就用文本表，对不上就退回权重表并告警（不静默）。
    """
    wnames = {int(k): str(v) for k, v in (getattr(model, "names", {}) or {}).items()}
    if tag != "ui" or not _CLASSES_TXT.is_file():
        return wnames
    lines = [l.strip() for l in
             _CLASSES_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
    nc = len(wnames)
    # 2026-08-11 放宽「行数必须相等」：**master 比权重多是正常状态** ——
    #    加了新 cls 还没重训时就是这样，而且前端（server/app.py `_master_append`）
    #    在标注中心自己就会 append。新类**永远 append 在末尾**，idx 递增、
    #    前 nc 行的对应关系不变  取前 nc 行仍然正确。
    #    **只放宽「多于」**：少于 nc 说明表被删过行 = idx 已错位，仍整表退回。
    #    再加一道安全闸：截断后 diff 若暴增（>40），说明不是单纯 append 而是
    #      顺序真乱了（memory 里那种「按另一套 idx 打的标」），照样退回权重表。
    if len(lines) < nc:
        print(f"[detect] _classes.txt {len(lines)} 行 < 权重 nc={nc}"
              f" — 表被删过行, idx 已错位, 退回权重名字表", flush=True)
        return wnames
    extra = len(lines) - nc
    head = lines[:nc]
    diff = [(i, wnames[i], head[i]) for i in range(nc) if wnames.get(i) != head[i]]
    if len(diff) > 40:
        print(f"[detect] 截断到前 {nc} 行后仍有 {len(diff)} 处不同 —"
              f" 疑似顺序错乱而非 append, 退回权重名字表", flush=True)
        return wnames
    if extra:
        print(f"[detect] _classes.txt 比权重多 {extra} 行（新 cls 待重训）"
              f" — 取前 {nc} 行, 废案过滤保持有效", flush=True)
    if diff:
        print(f"[detect] 类名表以 _classes.txt 为准，{len(diff)} 个与权重不同"
              f"（改名/标废弃）", flush=True)
    return {i: head[i] for i in range(nc)}


def _registry() -> dict:
    if not _REGISTRY.is_file():
        raise RuntimeError(f"model registry 缺失: {_REGISTRY}")
    return json.loads(_REGISTRY.read_text(encoding="utf-8"))


def resolve(model_key: str) -> Path:
    """registry[key].active  versions[active].path。**fail-closed**。"""
    reg = _registry()
    sec = reg.get(model_key)
    if not sec:
        raise RuntimeError(f"registry 无 '{model_key}' 节 — 修 registry，不回落老模型")
    active = sec.get("active")
    info = (sec.get("versions") or {}).get(active) or {}
    p = info.get("path")
    if not p:
        raise RuntimeError(f"registry {model_key}.active='{active}' 无 path")
    path = Path(p)
    if not path.is_absolute():
        path = Path("D:/Project") / str(path).lstrip("/\\")
    if not path.is_file():
        raise RuntimeError(f"registry {model_key}.active='{active}' 路径无效: {path}")
    return path


def load(tag: str = "ui"):
    """按 tag 载模型（进程内单例）。tag: ui / avatar / battle / emoticon。"""
    key = {"ui": "ui", "avatar": "fused_avatar",
           "battle": "battle_heads", "emoticon": "emoticon"}[tag]
    with _lock:
        if tag not in _models:
            from ultralytics import YOLO
            path = resolve(key)
            m = YOLO(str(path))
            _models[tag] = m
            _names[tag] = _name_table(m, tag)
            n = len(_names[tag])
            dep = sum(1 for v in _names[tag].values() if str(v).startswith("_废弃"))
            print(f"[detect] {tag}  {path.name} (nc={n}, 其中废案 {dep} 个已屏蔽)",
                  flush=True)
        return _models[tag]


def warm(tags=("ui",)):
    """预热 —— 主循环起跑前调用，别让第一帧背模型加载的几秒。

    起跑必须过预热闸（2026-08-01 事故）：冷启动的加载帧会把收菜 skill 的
    预算烧光。这里同步加载并跑一张空图，返回才算就绪。
    """
    import numpy as np
    for t in tags:
        m = load(t)
        try:
            m(np.zeros((640, 640, 3), dtype=np.uint8),
              conf=0.5, imgsz=_IMGSZ.get(t, 960), verbose=False)
        except Exception:
            pass


def infer(frame, tags=("ui",), conf_override: Optional[float] = None) -> List[Box]:
    """一帧  Box 列表（归一化坐标，已过滤废案类）。"""
    global _dropped_deprecated
    if frame is None:
        return []
    h, w = frame.shape[:2]
    out: List[Box] = []
    for tag in tags:
        m = load(tag)
        c = conf_override if conf_override is not None else _CONF.get(tag, 0.25)
        try:
            res = m(frame, conf=c, imgsz=_IMGSZ.get(tag, 960), verbose=False)
        except Exception as e:
            print(f"[detect] {tag} 推理失败: {e}", flush=True)
            continue
        table = _names.get(tag) or {}
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                idx = int(b.cls[0])
                name = str(table.get(idx, m.names.get(idx, idx)))
                if name.startswith("_废弃"):
                    _dropped_deprecated += 1        # A11：废案永不出这一层
                    continue
                out.append(Box(cls=name, conf=float(b.conf[0]),
                               x1=x1 / w, y1=y1 / h, x2=x2 / w, y2=y2 / h,
                               model=tag))
    return out


def stats() -> dict:
    return {"loaded": sorted(_models), "dropped_deprecated": _dropped_deprecated}
