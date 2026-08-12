# -*- coding: utf-8 -*-
"""前端 API（FastAPI router）—— 挂到 `server/app.py` 上，或独立起。

    # server/app.py 里加一行
    from routing_v2.app.api import router as v2_router
    app.include_router(v2_router)

    # 或者独立跑
    py -m uvicorn routing_v2.app.api:app --port 8788

前端不用手写表单：`GET /v2/schema` 吐出每个配置项的类型/标签/取值来源，
`GET /v2/scan/{name}` 在对应页面上**动态扫 cls** 拿真实可选项（悬赏分支 /
JFD 学院 / 活动关卡）—— **不写死清单**（§A8/§A9）。

金钱: `safety.*` 是只读的（`config.LOCKED` 强制覆盖，写进去也不生效）。
   `POST /v2/run` 的 `money_ok` 只在**这一次运行**里有效，不落盘 ——
   金钱步必须每天重新人审。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from routing_v2.config import LOCKED, SCHEMA, load, merged, save
from routing_v2.flow.registry import ALL, COMPOSITE
from routing_v2.state import vocab as V

router = APIRouter(prefix="/v2", tags=["routing_v2"])

_state: Dict[str, Any] = {
    "running": False, "thread": None, "runner": None,
    "log": [], "started": 0.0, "pending": None, "money_ok": False,
}
_LOCK = threading.Lock()
_LOG_CAP = 4000


def _log(m: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    with _LOCK:
        _state["log"].append(line)
        if len(_state["log"]) > _LOG_CAP:
            del _state["log"][:len(_state["log"]) - _LOG_CAP]


# ══ 配置 ═══════════════════════════════════════════════════════════════
@router.get("/schema")
def get_schema():
    """前端渲染表单用。`dynamic` 字段的取值要调 /v2/scan/{name}。"""
    return {"schema": SCHEMA, "locked": list(LOCKED),
            "flows": sorted(ALL), "composite": COMPOSITE}


@router.get("/config")
def get_config():
    return load()


class ConfigIn(BaseModel):
    config: Dict[str, Any]


@router.post("/config")
def post_config(body: ConfigIn):
    cfg = merged(body.config)          # LOCKED 在这里被强制盖回来
    save(cfg)
    return {"ok": True, "config": cfg,
            "note": "safety.* / shop.refresh_times / run.frame_source 已被锁死覆盖"}


# ══ 动态选项（进页扫 cls，不写死清单）═════════════════════════════════
_SCANS = {
    "bounty_branches": (V.BOUNTY_BRANCHES, "在悬赏分支选择页扫"),
    "jfd_academies": (V.JFD_ACADEMIES, "在学院交流会学院选择页扫"),
    "event_shop_currencies": ([V.CURRENCY, V.CURRENCY_SEL], "在活动商店页扫"),
    "event_stages": ([V.STAGE_ENTER, V.STAGE_ENTER_LOCKED], "在活动关卡列表页扫"),
    "cafe_students": ([], "邀请学生（头像 cls，需要 avatar 模型）"),
}


@router.get("/scan/{name}")
def scan(name: str):
    """抓一帧，看这一页上**真的有哪些**可选项。

    前端必须先把游戏切到对应页面再调 —— 这就是"动态"的代价，也是它比
       写死清单强的地方：换活动/换版面/游戏更新都自动跟得上。
    """
    if name not in _SCANS:
        raise HTTPException(404, f"未知扫描项 {name}，可选: {sorted(_SCANS)}")
    classes, where = _SCANS[name]
    if not classes:
        return {"name": name, "options": [], "note": where}
    obs = _grab()
    if obs is None:
        raise HTTPException(503, "取不到帧（scrcpy 没起来？）")
    from routing_v2.state.pages import classify
    m = classify(obs)
    found = [{"cls": b.cls, "conf": round(b.conf, 3),
              "cx": round(b.cx, 3), "cy": round(b.cy, 3)}
             for b in obs.rows(list(classes), 0.35)]
    return {"name": name, "page": m.page, "where": where,
            "options": [f["cls"] for f in found], "detail": found}


def _grab():
    """临时抓一帧（只给 API 的 probe/scan 用；主循环有自己的常驻 feed）。"""
    from routing_v2.percept import detect
    from routing_v2.percept.device import Device
    from routing_v2.percept.feed import Feed
    from routing_v2.percept.observe import Observation
    r = _state.get("runner")
    if r is not None and getattr(r, "feed", None) is not None:
        fr, age, seq = r.feed.latest()
        if fr is not None:
            h, w = fr.shape[:2]
            return Observation(boxes=detect.infer(fr, ("ui",)), frame=fr,
                               seq=seq, age=age, ts=time.time(), w=w, h=h)
    dev = Device(log=_log)
    detect.warm(("ui",))
    feed = Feed(serial=dev.serial, log=_log)
    if not feed.start():
        return None
    fr, age, seq = feed.wait_new(-1, timeout=5.0)
    feed.stop()
    if fr is None:
        return None
    h, w = fr.shape[:2]
    return Observation(boxes=detect.infer(fr, ("ui",)), frame=fr, seq=seq,
                       age=age, ts=time.time(), w=w, h=h)


@router.get("/probe")
def probe():
    obs = _grab()
    if obs is None:
        raise HTTPException(503, "取不到帧")
    from routing_v2.state.pages import classify, describe
    m = classify(obs)
    return {"page": m.page, "note": describe(m.page), "interrupt": m.interrupt,
            "age_ms": round(obs.age * 1000), "seq": obs.seq,
            "boxes": [{"cls": b.cls, "conf": round(b.conf, 3),
                       "cx": round(b.cx, 3), "cy": round(b.cy, 3)}
                      for b in sorted(obs.boxes, key=lambda x: -x.conf)[:60]]}


# ══ 运行 ═══════════════════════════════════════════════════════════════
class RunIn(BaseModel):
    flows: Optional[List[str]] = None
    step_mode: Optional[bool] = None
    money_ok: bool = False           # 只对这一次运行有效，不落盘


@router.post("/run")
def run(body: RunIn):
    with _LOCK:
        if _state["running"]:
            raise HTTPException(409, "已经在跑了")
        _state.update(running=True, log=[], started=time.time(),
                      money_ok=bool(body.money_ok), pending=None)

    cfg = load()
    if body.flows:
        cfg["order"] = list(body.flows)
        for k in cfg["modules"]:
            cfg["modules"][k] = k in body.flows
        for k in ("club", "craft", "shop"):
            if k in body.flows:
                cfg["modules"]["daily_routine"] = True
    if body.step_mode is not None:
        cfg["run"]["step_mode"] = bool(body.step_mode)

    def approver(act, obs) -> bool:
        """逐帧门控 / 金钱人审 —— 挂起等前端 /v2/approve。"""
        if act.money and not _state["money_ok"]:
            _log(f"金钱步「{act.reason}」本次未授权（money_ok=False） 拒绝")
            return False
        with _LOCK:
            _state["pending"] = {"action": repr(act), "reason": act.reason,
                                 "money": act.money, "spend": act.spend,
                                 "x": act.x, "y": act.y,
                                 "cls": act.target_cls, "answer": None,
                                 "ts": time.time()}
        _log(f"⏸ 等前端放行: {act}")
        t0 = time.time()
        while time.time() - t0 < 600:
            with _LOCK:
                ans = (_state["pending"] or {}).get("answer")
            if ans is not None:
                with _LOCK:
                    _state["pending"] = None
                return bool(ans)
            time.sleep(0.05)
        _log("⏸ 人审超时 10 分钟  当作不放行")
        with _LOCK:
            _state["pending"] = None
        return False

    def worker():
        from routing_v2.app.runner import Runner
        try:
            r = Runner(cfg, log=_log,
                       approver=approver if cfg["run"].get("step_mode", True)
                       or body.money_ok else None)
            _state["runner"] = r
            r.run_all()
        except Exception as e:
            import traceback
            _log(f"运行异常: {e}\n{traceback.format_exc()}")
        finally:
            _state["running"] = False

    t = threading.Thread(target=worker, daemon=True)
    _state["thread"] = t
    t.start()
    return {"ok": True, "flows": cfg["order"],
            "step_mode": cfg["run"].get("step_mode", True),
            "money_ok": bool(body.money_ok)}


class ApproveIn(BaseModel):
    approve: bool


@router.post("/approve")
def approve(body: ApproveIn):
    with _LOCK:
        if not _state["pending"]:
            raise HTTPException(409, "现在没有等待放行的动作")
        _state["pending"]["answer"] = bool(body.approve)
    return {"ok": True}


@router.get("/status")
def status(tail: int = 200):
    r = _state.get("runner")
    st = getattr(r, "stats", None)
    return {
        "running": _state["running"],
        "elapsed_s": round(time.time() - _state["started"], 1)
        if _state["started"] else 0,
        "pending": _state["pending"],
        "ticks": getattr(st, "ticks", 0),
        "taps": getattr(st, "taps", 0),
        "halted": getattr(st, "halted", ""),
        "reports": getattr(st, "reports", []),
        "balances": getattr(getattr(r, "ledger", None), "confirmed", {}),
        "log": _state["log"][-tail:],
    }


@router.post("/stop")
def stop():
    r = _state.get("runner")
    if r is not None:
        r.stats.halted = "前端停止"
        with _LOCK:
            if _state["pending"]:
                _state["pending"]["answer"] = False
    return {"ok": True}


@router.get("/health")
def health():
    from routing_v2.state.vocab import DEAD, WEAK, health_report
    return {"dead": sorted(DEAD), "weak": sorted(WEAK), "report": health_report()}


@router.get("/", response_class=HTMLResponse)
def console():
    """控制台单页。所有表单由 `/v2/schema` 自省生成 —— 加配置项前端自动跟，
    不用改这个 HTML（老 dashboard 把悬赏三个分支写成硬编码 checkbox 的教训）。"""
    p = Path(__file__).with_name("console.html")
    if not p.is_file():
        raise HTTPException(404, f"找不到 {p}")
    return HTMLResponse(p.read_text(encoding="utf-8"))


# 独立运行时的 app
try:
    from fastapi import FastAPI
    app = FastAPI(title="routing_v2")
    app.include_router(router)
except Exception:      # pragma: no cover
    app = None
