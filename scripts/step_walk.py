# -*- coding: utf-8 -*-
"""守卫式步进 —— 逐帧门控的可扩展版本(2026-07-25 live 建)。

## 为什么需要它
用户要求"门控每一帧"; 但一次 daily 全链上百步, 纯人工一步一轮跑不完。
execution_doctrine #13 的例外: **同构且已逐帧验证过的重复步骤**可批量。
本工具把"批量"做成**带闸的**: 每一步照样抓干净帧 + 跑 YOLO + 比对落点,
只有**全部守卫通过**才自动放行, 任何一条不满足就**停下来交给人**。

## 守卫(任一触发即停, fail-closed)
 ① **碰钱**: reason 含 購買/购买/PURCHASE/確認/确认/青辉石/免費/pyroxene …
    → 一律停。金钱路径永远人审, 不管见过多少次。
 ② **新 reason**: 没在本次 walk 里被人批准过的 reason 模式 → 停。
    (同构重复步的 reason 字符串相同, 第一次人批, 之后才自动)
 ③ **落点无 cls 支撑**: click 落点 0.06 半径内没有任何 conf>=0.30 的检出框
    → 停。这是"盲拍"的直接判据 —— 感知层没锁定就不许动手。
 ④ **连点同一目标**: 同一 (reason, 落点) 连续 N 次 → 停(空转/被吞的指纹)。
 ⑤ **步数上限**: 防跑飞。

## 每步都留痕
逐帧写 jsonl: tick / skill / sub / reason / 落点 / 该帧全部检出 / 守卫判定。
事后可复盘"第几步的感知是什么样"—— 日志是意图, 帧才是事实。

用法:
  py scripts/step_walk.py --max 40                    # 走最多 40 步
  py scripts/step_walk.py --max 40 --allow "dismiss"  # 预批准含该串的 reason
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

ADB = r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
DEV = "127.0.0.1:7555"
API = "http://127.0.0.1:8000/api/v1"
TMP = Path(os.environ.get("TEMP", ".")) / "_walk.png"

# ⛔硬碰钱词 —— 命中一律停。只留**真会掏钱**的那几个。
# 早先把 確認/扫荡/票 也塞进来了, 结果掃蕩確認一天几十次全卡住 = 跑不完。
MONEY = ("購買", "购买", "PURCHASE", "青辉石", "青輝石", "pyroxene",
         "免費", "免费", "buy", "BUY")

# ⭐结构判据(比字符串可靠, 与 skill 内 _dialog_is_purchase 同源):
# **确认框在屏 且 框体内有数量步进器/体力/青辉石** = 购买框 → 停。
# 纯 AP 扫荡确认框体内只有 取消/确认/叉, 不会命中 → 不卡流程。
_DLG_BTN = ("确认键", "取消键")
_DLG_BODY = ("加号", "加号灰色", "减号", "减号灰色", "MAX_可点击", "MAX_灰色",
             "MIN_可点击", "MIN_灰色", "体力", "青辉石")


def purchase_dialog_structure(boxes) -> str:
    """返回非空字符串 = 这一帧有购买框结构(该停)。"""
    names = {b.cls_name for b in boxes}
    if not all(n in names for n in _DLG_BTN):
        return ""
    # 框体 = 确认/取消按钮上方的区域
    ys = [(b.y1 + b.y2) / 2 for b in boxes if b.cls_name in _DLG_BTN]
    btn_y = min(ys) if ys else 1.0
    body = [b for b in boxes
            if b.cls_name in _DLG_BODY and 0.12 < (b.y1 + b.y2) / 2 < btn_y]
    return ("购买框结构: 确认+取消 且体内有 "
            + ",".join(sorted({b.cls_name for b in body}))) if body else ""
CLS_RADIUS = 0.06      # 落点多远内要有检出框才算"有 cls 支撑"
REPEAT_CAP = 4         # 同一 (reason,落点) 连续几次算空转


def api(path: str, method: str = "GET", body=None):
    import urllib.request
    req = urllib.request.Request(
        f"{API}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())
    except Exception as e:                                    # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def grab():
    subprocess.run([ADB, "-s", DEV, "shell", "screencap", "-p",
                    "/sdcard/_walk.png"], capture_output=True, timeout=60)
    subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/_walk.png", str(TMP)],
                   capture_output=True, timeout=60)
    import cv2
    return cv2.imread(str(TMP))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--allow", action="append", default=[],
                    help="预批准的 reason 子串(可多次); 碰钱词仍然一律停")
    # ⛔与 --allow **分开**是故意的: 金钱步必须逐条显式点名才放行, 不能被
    # 普通白名单顺带批过去。写进命令行 = 我确实逐帧看过那一步的落点与检出。
    ap.add_argument("--money-ok", action="append", default=[],
                    help="已人审过的**金钱**步 reason 子串, 精确点名才放行")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--log", default="")
    a = ap.parse_args()

    from brain.pipeline import (BASE_DETECTORS, SKILL_YOLO_MAP,
                                _run_yolo_on_image)
    from brain.nav.page_graph import identify
    from brain.skills.base import ScreenState

    def ctx_for(skill_name: str) -> str:
        """⛔必须跟 pipeline 用**同一套检测器**(2026-07-25 当场踩): 探针写死
        context="ui" 时, Schedule/Cafe 的**学生头像框(avatar 域 143-394)**
        我根本看不见 → 把有 cls 支撑的落点判成"盲拍" = 假警报, 差点据此报 bug。
        SKILL_YOLO_MAP: Schedule=ui+avatar / DailyRoutine=ui+cafe+avatar /
        Bounty·Arena·JFD=ui+battle。"""
        return SKILL_YOLO_MAP.get(skill_name or "", BASE_DETECTORS)

    logp = Path(a.log) if a.log else (
        _ROOT / "data" / f"walk_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    # money_ok 同时算作"已人审过的 reason" —— 否则两道闸串成死结:
    # 金钱闸放行了, 新-reason 闸又拦下来, 而 --allow 里加它又会削弱金钱闸的语义。
    approved: set = set(a.allow) | set(a.money_ok)
    last_sig = None
    repeats = 0
    n = 0

    while n < a.max:
        raw = api("/step/pending")
        pend = (raw.get("pending") or {}) if not raw.get("_error") else {}
        if raw.get("_error"):
            print(f"⛔ API: {raw['_error']}")
            return 1
        if not pend:
            time.sleep(0.8)
            continue

        reason = str(pend.get("reason", ""))
        act = str(pend.get("action", ""))
        tgt = pend.get("target")
        img = grab()
        if img is None:
            print("⛔ 抓帧失败 — 停")
            return 1
        h, w = img.shape[:2]
        _ctx = ctx_for(str(pend.get("skill") or ""))
        boxes = [b for b in _run_yolo_on_image(img, w, h, context=_ctx)
                 if b.confidence >= a.conf]
        boxes.sort(key=lambda b: -b.confidence)
        try:
            page, why = identify(ScreenState(ocr_boxes=[], image_w=w,
                                             image_h=h, frame=img,
                                             yolo_boxes=boxes))
        except Exception:                                     # noqa: BLE001
            page, why = "?", "?"

        # ── 落点的 cls 支撑 ──
        near = []
        if act == "click" and isinstance(tgt, (list, tuple)) and len(tgt) == 2:
            tx, ty = float(tgt[0]), float(tgt[1])
            for b in boxes:
                cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
                if abs(cx - tx) <= CLS_RADIUS and abs(cy - ty) <= CLS_RADIUS:
                    near.append((b.cls_name, round(b.confidence, 2)))

        sig = (reason, tuple(tgt) if isinstance(tgt, (list, tuple)) else tgt)
        repeats = repeats + 1 if sig == last_sig else 0
        last_sig = sig

        rec = {"i": n, "tick": pend.get("tick"), "skill": pend.get("skill"),
               "sub": pend.get("sub_state"), "action": act, "reason": reason,
               "target": tgt, "page": page, "near_cls": near,
               "boxes": [(b.cls_name, round(b.confidence, 2),
                          round((b.x1 + b.x2) / 2, 3),
                          round((b.y1 + b.y2) / 2, 3)) for b in boxes[:14]]}

        # ── 守卫 ──
        stop = None
        _money = any(k in reason for k in MONEY)
        _money_ok = any(k in reason for k in a.money_ok) if a.money_ok else False
        _struct = purchase_dialog_structure(boxes)
        rec["purchase_struct"] = _struct
        if _struct and act == "click":
            # ⛔结构闸不吃白名单: 屏上真有购买框还去点, 一律人审
            stop = f"⛔{_struct} —— 屏上有购买框还要点, 人审"
        elif _money and not _money_ok:
            stop = f"碰钱词 → 人审(reason={reason!r})"
        elif repeats >= REPEAT_CAP:
            stop = f"同一目标连发 {repeats + 1} 次 = 空转/被吞"
        elif act == "click" and not near:
            # ⚠盲拍按危害分级(2026-07-25 live 定): 关弹窗/確認 这类**尾发**是已知
            # 良性形态 —— skill 在它自己那帧上确实看得见按钮(两次落点坐标不同=新检出),
            # 只是弹窗在关闭动画里, 到我探针时已消失(铁律#17 探针帧永远比 pending 新)。
            # 这类放行但**大声记账 + 连发上限**; **导航/进入类**盲拍一律停(点歪=进错页,
            # schedule 那次 popout 尾发就误开过设施)。
            _dismissy = any(k in reason for k in (
                "close", "dismiss", "確認", "确认", "取消", "叉叉", "reward",
                "continue", "領取", "领取"))
            if _dismissy and repeats < 2:
                rec["note"] = "弹窗尾发(落点已无cls, 良性形态)"
                print("      ⚠尾发: 落点已无 cls(弹窗关闭动画) — 放行但记账")
            else:
                stop = (f"落点 {tgt} 半径{CLS_RADIUS} 内**无任何 cls 支撑** = 盲拍"
                        + ("(尾发但已连发)" if _dismissy else "(导航类, 危险)"))
        elif not any(k in reason for k in approved):
            stop = "新 reason, 没被人批准过"

        rec["verdict"] = stop or "auto-go"
        with open(logp, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[{n:>3}] {pend.get('skill')}/{pend.get('sub_state')} "
              f"{act} · {reason[:52]}  [det={_ctx}]")
        print(f"      落点={tgt} 附近cls={near or '无'} | 页面={page}")
        if stop:
            print(f"      ⛔停: {stop}")
            print(f"\n本帧全部检出({len(boxes)}):")
            for b in boxes[:14]:
                print(f"  {b.confidence:.2f} {b.cls_name:<24} "
                      f"cx={(b.x1 + b.x2) / 2:.3f} cy={(b.y1 + b.y2) / 2:.3f}")
            print(f"\n审完继续: py scripts/step_walk.py --max N "
                  + " ".join(f'--allow "{s}"' for s in sorted(approved))
                  + f' --allow "{reason[:28]}"')
            print(f"日志: {logp}")
            return 0
        api("/step/go", "POST", {})
        n += 1
        time.sleep(0.9)

    print(f"\n走满 {n} 步, 全部守卫通过。日志: {logp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
