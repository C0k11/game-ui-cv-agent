# -*- coding: utf-8 -*-
"""逐帧门控探针 —— step_mode 审核的标准工具(用户 2026-07-25: "门控每一帧,
确保逻辑以及我们的 yolo 没问题")。

每一步做三件事, 缺一不可:
  ① **抓干净帧**(ADB screencap, Android 内部取流 — overlay 物理进不去)
  ② **跑 YOLO 看检出** —— 验感知层: 这一帧模型认出了什么, 置信度多少
  ③ **拉 bot 的 pending 意图** —— 验逻辑层: 它想干什么, 落点在哪个 cls 上
然后人/主对话判断"该不该放行", 对了才 go。

⛔纪律(execution_doctrine #13, 用户三次纠偏): **绝不批量放行**。
只看 reason 字符串就 go = 假审核 —— bounty 弃 6 票 / jfd 假成功 / arena 空转
全是这么放过去的。批量只允许用于**同构且已逐帧验证过**的重复步骤。
⛔纪律(#17): **探针帧永远比 pending 新** —— pending 是几秒前那一帧算出来的。
看到"落点与当前检出对不上"先想这个, 别急着报 bug。

用法:
  py scripts/step_probe.py              # 抓帧+检出+pending, 不放行
  py scripts/step_probe.py --go         # 审完放行一步
  py scripts/step_probe.py --shot out.png   # 另存标注图供目检
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

ADB = r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
# ⛔端口不写死 — MuMu 实例重启会换号(2026-07-28: 7555→16384, 全仓 16 处硬编码)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from brain.mumu_port import mumu_serial
    DEV = mumu_serial()
except Exception:
    DEV = "127.0.0.1:7555"
API = "http://127.0.0.1:8000/api/v1"
TMP = Path(os.environ.get("TEMP", ".")) / "_step_probe.png"


def api(path: str, method: str = "GET", body=None):
    req = urllib.request.Request(
        f"{API}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:                                    # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def grab() -> str:
    """ADB 干净帧。⚠必须走 pull, 不能用 PowerShell 的 `>` 重定向(会破坏二进制)。"""
    subprocess.run([ADB, "-s", DEV, "shell", "screencap", "-p",
                    "/sdcard/_probe.png"], capture_output=True, timeout=60)
    subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/_probe.png", str(TMP)],
                   capture_output=True, timeout=60)
    return str(TMP)


def main() -> int:
    ap = argparse.ArgumentParser()
    # ⛔语义: **先放行(上一轮已审过的那一步), 再展示下一步**。
    # 反过来(先展示后放行)等于没审就批 —— 那是铁律 #13 禁止的自动放行。
    ap.add_argument("--go", action="store_true",
                    help="先放行上一轮审过的那步, 再抓下一步给我审")
    # ⛔硬闸(2026-07-25 live 当场补): 上一轮探针如果显示 pending=无(bot 还在
    # 算/在 wait), 我 --go 就会把**期间新算出来的动作盲批掉** —— 实测发生过一次
    # (批了 BuyPyroxene 的 FREE 购买点击, 事后核对才确认是对的 = 靠运气不是门控)。
    # 加 --expect: 放行前先比对当前 pending 的 reason, 对不上直接拒批。
    ap.add_argument("--expect", default="",
                    help="必须与当前 pending 的 reason 匹配(子串)才放行, 否则拒批")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--shot", default="")
    ap.add_argument("--no-frame", action="store_true", help="只看 pending, 不抓帧")
    a = ap.parse_args()

    if a.go:
        # 放行的是**上一轮已经审过**的那一步
        cur = (api("/step/pending").get("pending") or {})
        if a.expect:
            got = str(cur.get("reason", ""))
            if not cur:
                print(f"⛔拒批: 当前无 pending, 但我期待 {a.expect!r} —— "
                      f"上一轮看到的那步已经不在了, 重新探再审")
                return 2
            if a.expect not in got:
                print(f"⛔拒批: 期待 reason 含 {a.expect!r}, 实际是 {got!r} —— "
                      f"这不是我审过的那一步, 绝不盲批")
                return 2
        elif not cur:
            print("⛔拒批: 当前无 pending —— 加 --expect 或先探针看清再批")
            return 2
        r = api("/step/go", "POST", {})
        print(f"[放行] {r.get('approved', r)}")
        import time as _t
        _t.sleep(1.2)      # 给 bot 执行 + 算出下一步的时间

    # ── ③ bot 的意图(先拉, 免得抓帧耗时后状态又变) ──────────────────────
    # ⚠响应是**包了一层**的 {"step_mode":bool, "pending":{...}|None};
    # pending 里 action 是动作类型**字符串**(click/back/wait/...), target 平铺。
    raw = api("/step/pending")
    pend = (raw.get("pending") or {}) if not raw.get("_error") else {}
    print("=" * 78)
    if raw.get("_error"):
        print(f"pending: {raw['_error']}")
    elif not pend:
        print(f"pending: 无(step_mode={raw.get('step_mode')}) — "
              f"bot 未暂停在动作上(可能在 wait/加载/刚启动还没算出第一步)")
    else:
        print(f"skill={pend.get('skill')}  sub={pend.get('sub_state')}  "
              f"tick={pend.get('tick')}")
        print(f"意图: {pend.get('action')}  →  {pend.get('reason')}")
        t = pend.get("target")
        if t:
            print(f"  落点: {t}")

    if a.no_frame:
        return 0

    # ── ①② 干净帧 + YOLO 检出 ─────────────────────────────────────────
    p = grab()
    import cv2
    img = cv2.imread(p)
    if img is None:
        print("⛔抓帧失败")
        return 1
    h, w = img.shape[:2]
    # ⛔跟 pipeline 用同一套检测器(2026-07-25 踩): 写死 "ui" 会看不见
    # Schedule/Cafe 的**学生头像框(avatar 域)** → 把有支撑的落点误判成盲拍。
    from brain.pipeline import (BASE_DETECTORS, SKILL_YOLO_MAP,
                                _run_yolo_on_image)
    _ctx = SKILL_YOLO_MAP.get(str(pend.get("skill") or ""), BASE_DETECTORS)
    boxes = sorted(_run_yolo_on_image(img, w, h, context=_ctx),
                   key=lambda b: -b.confidence)
    boxes = [b for b in boxes if b.confidence >= a.conf]
    print("-" * 78)
    print(f"帧 {w}x{h}  检测器={_ctx}  YOLO 检出 {len(boxes)} 框 "
          f"(conf>={a.conf}):")
    for b in boxes[:a.top]:
        print(f"  {b.confidence:.2f}  {b.cls_name:<26} "
              f"cx={(b.x1 + b.x2) / 2:.3f} cy={(b.y1 + b.y2) / 2:.3f}")
    if len(boxes) > a.top:
        print(f"  ... 另 {len(boxes) - a.top} 框")

    # 页面图观测(L2, 只观测不接管)
    try:
        from brain.nav.page_graph import identify
        from brain.skills.base import ScreenState, YoloBox
        scr = ScreenState(ocr_boxes=[], image_w=w, image_h=h, frame=img,
                          yolo_boxes=boxes)
        name, why = identify(scr)
        print(f"[PageGraph] {name}  ({why})")
    except Exception as e:                                    # noqa: BLE001
        print(f"[PageGraph] 不可用: {type(e).__name__}: {e}")

    # ── 对照图: 左=原始游戏画面, 右=YOLO 框 + 落点十字 ──────────────────
    # 用户 2026-07-25: "走一步你就看一下游戏画面然后再看一下带有yolo框的画面"。
    # 拼成一张是为了一次 Read 就能同时看到两边(逐帧审时每多一次调用都是成本)。
    out = a.shot or str(Path(os.environ.get("TEMP", ".")) / "_step_view.png")
    SW = 1100                       # 每半边宽度
    sc = SW / w
    raw = cv2.resize(img, (SW, int(h * sc)))
    ann = raw.copy()
    # ⚠cv2.putText 画不了中文(渲染成 ??????, 2026-07-25 当场发现) → 框上只标
    # **编号**, 编号↔类名的对照在上面的文字清单里(已按 conf 降序打印)。
    for i, b in enumerate(boxes):
        x1, y1 = int(b.x1 * w * sc), int(b.y1 * h * sc)
        x2, y2 = int(b.x2 * w * sc), int(b.y2 * h * sc)
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.rectangle(ann, (x1, max(0, y1 - 15)), (x1 + 26, y1),
                      (0, 200, 255), -1)
        cv2.putText(ann, str(i), (x1 + 3, max(11, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    # 落点十字(红) —— 一眼看出 bot 要点哪
    if isinstance(pend.get("target"), (list, tuple)) and len(pend["target"]) == 2:
        tx, ty = pend["target"]
        px, py = int(float(tx) * SW), int(float(ty) * h * sc)
        for canvas in (raw, ann):
            cv2.drawMarker(canvas, (px, py), (0, 0, 255),
                           cv2.MARKER_CROSS, 46, 4)
            cv2.circle(canvas, (px, py), 30, (0, 0, 255), 3)
    hh = raw.shape[0]
    sheet = cv2.hconcat([raw, ann])
    cv2.line(sheet, (SW, 0), (SW, hh), (255, 255, 255), 2)
    cv2.putText(sheet, "RAW", (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)
    cv2.putText(sheet, "YOLO", (SW + 12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2)
    cv2.imwrite(out, sheet)
    print(f"→ 对照图 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
