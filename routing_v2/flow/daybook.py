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


def _file(cfg: dict) -> Path:
    """账号桶里的 daybook（08-15 分桶: 键只有游戏日, 大小号同一天会把
    对方的「免費包领过了」当成自己的账）。cfg 必须是**全量配置**。"""
    from routing_v2.config import data_dir
    return data_dir(cfg) / "daybook.json"


def game_day() -> str:
    return (datetime.now(timezone(timedelta(hours=8)))
            - timedelta(hours=3)).strftime("%Y%m%d")


def _load(cfg: dict) -> dict:
    try:
        d = json.loads(_file(cfg).read_text(encoding="utf-8"))
        if d.get("day") == game_day():
            return d
    except Exception:
        pass
    return {"day": game_day()}


def done(key: str, cfg: dict) -> bool:
    """这件事本游戏日做过没。"""
    return bool(_load(cfg).get(key))


def mark(key: str, cfg: dict) -> None:
    """记账（写穿 -- 掉线/重启不丢今天的账）。

    ⛔「读-改-整份写回」的台账, **读失败时必须放弃写**(event_topped 2026-08-12
    数据丢失同型: 读失败吞成空字典, 加一条写回去把当天已有的账全抹了)。
    只有三种情况允许写: 文件不存在 / 正常读到今天的账 / 真的换游戏日了。
    """
    f = _file(cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                raise ValueError("daybook 不是 dict")
        except Exception:
            # 文件在但读不出 = 损坏/写一半 -- 宁可少记这一条, 绝不用
            #    空表覆盖真实台账
            print(f"[daybook] 读失败, 放弃记账 {key}（不覆盖现有文件）")
            return
        if d.get("day") != game_day():
            d = {"day": game_day()}      # 真换日: 开新账本
    else:
        d = {"day": game_day()}
    d[key] = True
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
