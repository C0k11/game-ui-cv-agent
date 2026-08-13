# -*- coding: utf-8 -*-
"""按游戏日对齐的本地台账。

用户 2026-08-13 拍板的模式（先用在课程表房间上，这里做成公共件）:
   「我认为的解决办法是本地记录，然后根据游戏每日刷新对齐，这样就很简单了」。
屏上判据缺位的"今天做过没"（免費包领没领、房间上没上过课）一律走这里 --
   红点/绿勾只是副产品，**自己的账本才是权威**。
游戏日 = UTC+8 减 3 小时（繁中服 JST04:00 刷新）。绝不用裸 `datetime.now()`
   -- [[game_day_timezone]] 那次 12h 错窗就是这么来的。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

_FILE = Path("data/routing_v2/daybook.json")


def game_day() -> str:
    return (datetime.now(timezone(timedelta(hours=8)))
            - timedelta(hours=3)).strftime("%Y%m%d")


def _load() -> dict:
    try:
        d = json.loads(_FILE.read_text(encoding="utf-8"))
        if d.get("day") == game_day():
            return d
    except Exception:
        pass
    return {"day": game_day()}


def done(key: str) -> bool:
    """这件事本游戏日做过没。"""
    return bool(_load().get(key))


def mark(key: str) -> None:
    """记账（写穿 -- 掉线/重启不丢今天的账）。"""
    d = _load()
    d[key] = True
    _FILE.parent.mkdir(parents=True, exist_ok=True)
    _FILE.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
