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
MONEY = ("購買", "购买", "purchase", "青辉石", "青輝石", "pyroxene",
         "免費", "免费", "buy", "free", "cad", "$")


def _hits_money(reason: str) -> bool:
    """⛔大小写不敏感(2026-07-28 审 buy_pyroxene 时发现的**工具自身的洞**):
    旧表写死 `"PURCHASE"` 大写 + `"buy"/"BUY"` 两个变体, 而全 bot **最该拦的
    那一步** reason 是 `confirm FREE purchase (確認键)` —— 小写 `purchase` 不
    匹配大写常量, `FREE` 也不在表里 ⇒ **青辉石商店里点確認这一步能悄悄自动放行**。
    (buy_pyroxene 那个商店里, 免費組合包旁边就是 CAD$ 真钱包。)
    这就是「reason 措辞当控制信号」的又一种形态: 守卫按字符串匹配, 而字符串的
    大小写/措辞是另一个人随手写的。⇒ 统一小写比对, 并补上 free/cad/$。
    """
    r = reason.lower()
    return any(k.lower() in r for k in MONEY)

# ⭐结构判据(比字符串可靠, 与 skill 内 _dialog_is_purchase 同源):
# **确认框在屏 且 框体内有数量步进器/体力/青辉石** = 购买框 → 停。
# 纯 AP 扫荡确认框体内只有 取消/确认/叉, 不会命中 → 不卡流程。
_DLG_BTN = ("确认键", "取消键")
_DLG_BODY = ("加号", "加号灰色", "减号", "减号灰色", "MAX_可点击", "MAX_灰色",
             "MIN_可点击", "MIN_灰色", "体力", "青辉石")


def purchase_dialog_structure(boxes) -> str:
    """返回非空字符串 = 这一帧有购买框结构(该停)。

    ⛔2026-07-27 live 修: 旧版判"体内"**只卡 y 不卡 x**, 于是信用点商店的
    純信用点确认框(總購買價格 3,121,500 C)被判成"体内有青辉石" —— 那个青辉石
    在 cx=0.030, 是**被弹窗遮住的左侧栏「青輝石」tab**, 弹窗左界≈0.09。
    这跟 2026-07-25 全语料实测删掉的「体内体力」误伤(体力在弹窗右界 0.74 外的
    0.82, 是底层扫荡面板图标)是**同一个病**: 缺横向定界。

    横向定界几何(两个已知案例上都验过):
      中心 = (确认.cx + 取消.cx)/2      —— 按钮对称居中于弹窗
      右界 = 弹窗叉叉.cx + 0.02        —— 叉叉在弹窗右上角
      左界 = 中心 - (右界 - 中心)       —— 对称
    本帧实测: 中心 0.4995 / 叉叉 0.883 → 左界 0.096, 与目检的 0.09 吻合;
    青辉石 0.030 < 0.096 被正确排除。旧的体力案例 0.82 > 右界 0.74 也排除。

    ⛔**没有叉叉就定不出界 → fail-closed 照旧全宽判定**(宁可误报停下人审,
    不可漏掉真购买框)。
    """
    names = {b.cls_name for b in boxes}
    if not all(n in names for n in _DLG_BTN):
        return ""
    # 框体 = 确认/取消按钮上方的区域
    ys = [(b.y1 + b.y2) / 2 for b in boxes if b.cls_name in _DLG_BTN]
    btn_y = min(ys) if ys else 1.0
    btn_xs = [(b.x1 + b.x2) / 2 for b in boxes if b.cls_name in _DLG_BTN]
    x_lo, x_hi = 0.0, 1.0
    _xx = [(b.x1 + b.x2) / 2 for b in boxes if b.cls_name == "弹窗叉叉"]
    if btn_xs and _xx:
        _c = sum(btn_xs) / len(btn_xs)
        _r = max(_xx) + 0.02
        if _r > _c:                       # 叉叉必须在中心右侧才是同一个弹窗
            x_lo, x_hi = _c - (_r - _c), _r
    body = [b for b in boxes
            if b.cls_name in _DLG_BODY
            and 0.12 < (b.y1 + b.y2) / 2 < btn_y
            and x_lo <= (b.x1 + b.x2) / 2 <= x_hi]
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
    # 一次 daily 全链有几十种 reason, 每停一次就往命令行再拼一个 --allow,
    # 到后面命令行长到看不清**已批过什么** —— 那本身就是审核失效的开始。
    # 改成一个"人审台账"文件, 一行一条(# 开头是注释)。⛔碰钱词仍然一律停,
    # 金钱步只认 --money-ok, 台账文件放不进去。
    ap.add_argument("--allow-file", default="",
                    help="已人审通过的 reason 子串清单文件(一行一条)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--log", default="")
    # ⛔用户 2026-07-27: "要看和分析 yolo 识别帧" —— 只留 jsonl 是**事后无法目检**的
    # (jsonl 里是 cls 名+conf, 不是画面)。判断"这一帧该不该这么点"必须回到画面本身,
    # 而 grab() 原来只写同一个 TMP, 走一步覆盖一步 = 帧证据当场就没了。
    ap.add_argument("--frames", default="",
                    help="每步的干净帧另存到该目录(缺省=与日志同名的 _frames 目录)")
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
    framedir = Path(a.frames) if a.frames else logp.with_name(
        logp.stem + "_frames")
    framedir.mkdir(parents=True, exist_ok=True)
    # money_ok 同时算作"已人审过的 reason" —— 否则两道闸串成死结:
    # 金钱闸放行了, 新-reason 闸又拦下来, 而 --allow 里加它又会削弱金钱闸的语义。
    approved: set = set(a.allow) | set(a.money_ok)
    if a.allow_file:
        for _ln in Path(a.allow_file).read_text(encoding="utf-8").splitlines():
            _ln = _ln.strip()
            if _ln and not _ln.startswith("#"):
                approved.add(_ln)
        print(f"[allow-file] 已人审台账 {len(approved)} 条 ← {a.allow_file}")
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

        # ⛔落点必须**量化后**再比(2026-07-27 live 实锤): 旧码用浮点精确相等,
        # 而 YOLO 框每帧有亚像素抖动 —— 实测同一个「弹窗叉叉」连点 6 次, 其中
        # 一次落点是 0.6920320749 而其余是 0.6920148213, 差在小数第 5 位就把
        # repeats 清零, REPEAT_CAP=4 永远够不到 ⇒ **连点守卫被抖动架空**,
        # 眼睁睁看着 bot 对着一个无效目标点了 6 次没人喊停。
        # 这就是 memory live_gating_2026_07_25 里记的"关X连发(坐标不同=新检出)
        # 未定性"那条 —— 现在定性了。量化到 0.01(4K 上约 38px, 远小于任何按钮)。
        def _q(v):
            return round(float(v) / 0.01)
        sig = (reason, (_q(tgt[0]), _q(tgt[1]))
               if isinstance(tgt, (list, tuple)) and len(tgt) == 2 else tgt)
        repeats = repeats + 1 if sig == last_sig else 0
        last_sig = sig

        # ── 存帧: 原图 + 标注图(全部检出框 + 落点十字) ──
        # 判断"这一步安排得对不对"必须回到画面: cls 名对了不代表点的是对的东西
        # (Bonus 固定位漂那次, jsonl 里一切正常, 是看图才发现槽位错了)。
        # ⛔文件名必须带墙钟: walk 是"停一次改一次 --allow 再起一次"的用法,
        # n 每次从 0 重来 —— 只用 n 命名会**把上一段的帧证据覆盖掉**。
        _stem = f"{time.strftime('%H%M%S')}_{n:03d}"
        _raw_p = framedir / f"{_stem}_raw.png"
        _ann_p = framedir / f"{_stem}_ann.png"
        import cv2
        cv2.imwrite(str(_raw_p), img)
        _ann = img.copy()
        for b in boxes:
            p1 = (int(b.x1 * w), int(b.y1 * h))
            p2 = (int(b.x2 * w), int(b.y2 * h))
            cv2.rectangle(_ann, p1, p2, (0, 255, 0), 3)
            cv2.putText(_ann, f"{b.cls_name} {b.confidence:.2f}",
                        (p1[0], max(24, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2, cv2.LINE_AA)
        if act == "click" and isinstance(tgt, (list, tuple)) and len(tgt) == 2:
            cx_, cy_ = int(float(tgt[0]) * w), int(float(tgt[1]) * h)
            cv2.drawMarker(_ann, (cx_, cy_), (0, 0, 255),
                           cv2.MARKER_CROSS, 90, 6)
            cv2.circle(_ann, (cx_, cy_), int(CLS_RADIUS * w), (0, 0, 255), 3)
        cv2.imwrite(str(_ann_p), _ann)

        rec = {"i": n, "tick": pend.get("tick"), "skill": pend.get("skill"),
               "sub": pend.get("sub_state"), "action": act, "reason": reason,
               "target": tgt, "page": page, "page_why": why, "near_cls": near,
               "raw": str(_raw_p), "ann": str(_ann_p),
               "n_boxes": len(boxes),
               # ⛔全量存, 不再 [:14] —— 截断过的检出表没法回答"漏检了什么"
               "boxes": [(b.cls_name, round(b.confidence, 2),
                          round((b.x1 + b.x2) / 2, 3),
                          round((b.y1 + b.y2) / 2, 3)) for b in boxes],
               # ⛔只存中心点是不够的(2026-07-27 当场卡住): 想事后复算
               # purchase_dialog_structure 这类**用到 x1/x2/y1/y2 的几何判据**时,
               # 中心点重建不出框 → 只能回去重跑 YOLO。存全 bbox, 一次到位。
               "bboxes": [[b.cls_name, round(b.confidence, 3),
                           round(b.x1, 4), round(b.y1, 4),
                           round(b.x2, 4), round(b.y2, 4)] for b in boxes]}

        # ── 守卫 ──
        stop = None
        _money = _hits_money(reason)
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
            # ⭐zero-det wake 是**定义上**无 cls 支撑的动作, 盲拍闸对它永远误报
            # (2026-07-27 第一步就撞上)。与"危险盲拍"的本质区别:
            #   危险盲拍 = skill **以为**屏上有按钮 → 点歪进错页;
            #   zero-det wake = pipeline **明知**零检出, 故意点一个标定过的空白点。
            # 落点 (0.35,0.12) 经两代 lobby 皮肤实测且特意避开 topbar 的 AP「+」
            # (pipeline.py 那段注释: 旧点 0.5,0.05 压在 + 上, 反复戳开購買AP框)。
            # ⛔放行但大声记账, 且**不吃 repeats 豁免** —— 连发仍由 REPEAT_CAP 拦。
            _zero_wake = ("zero-det" in reason or "放置立绘屏" in reason)
            if _zero_wake and not boxes:
                rec["note"] = "zero-det wake(全帧零检出, 落点为标定空白区)"
                print("      ⚠zero-det wake: 全帧零检出, 落点是标定过的安全空白区 — 放行但记账")
            elif _zero_wake:
                # ⛔帧上其实有 UI 却还在发 wake = pending 已跨画面陈旧, 必须人审:
                # 落点在**新画面**上压着什么, 只能看这一帧才知道。
                stop = (f"zero-det wake 但探针帧有 {len(boxes)} 个检出 = pending 跨画面陈旧, "
                        f"人工确认落点 {tgt} 在当前画面上是空白")
            # ⭐摸头(2026-07-27): 落点锚在 `Emoticon_Action` 上, 而这个 cls
            # **实测就是弱的**(cafe_flow_spec: 650 帧录制, mean conf 0.65,
            # 18/51=35% < 0.55) 且**学生在走动**位置一直变。探针帧永远比 pending
            # 新(铁律#17) → 到我抓帧时 marker 常已闪没/移位 = 必然"无 cls 支撑"。
            # pipeline 那边 headpat 也在稳定门豁免里, 注释写明"走动学生永不稳定,
            # 摸头必须抢最新帧" —— 那个豁免是对的。⛔放行但记账 + 仍受 REPEAT_CAP。
            elif "headpat" in reason and repeats < 3:
                rec["note"] = "摸头(Emoticon_Action 弱cls+学生走动, 探针帧必已变)"
                print("      ⚠摸头: 落点 marker 已闪没/移位 — 放行但记账")
            elif _dismissy and repeats < 2:
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
