# -*- coding: utf-8 -*-
"""事故回归库 suite (2026-07-14, 自动驾驶'事故回归库'借鉴落地).

历史事故帧固定集, 新模型/新防线出货前必跑:
  py scripts/regression_suite.py            # 全部用例
  py scripts/regression_suite.py --domain ui
veto 用例 FAIL → exit 2(金钱安全一票否决, 训练收尾流程必须硬停);
非 veto FAIL → exit 1(功能回退警告, 人工定夺)。

check 实现与 brain 内真实防线同判据(独立复刻, 判据变更两边同步):
  purchase_dialog ≈ event_quest._dialog_is_purchase 结构白名单闸:
    取消键+确认键同屏(conf≥0.20 守卫地板) 且 body(y>0.12) 出现
    stepper(加号/MAX_可点击/MIN_灰色) 或 体力 → 购买框。
帧来源见 data/regression/manifest.json 各 case 的 origin。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Project\ai game secretary")
from vision.io_utils import imread_any  # noqa: E402

REG = Path(r"D:\Project\ai game secretary\data\regression")
REGISTRY = Path(r"D:\Project\ai game secretary\data\model_registry.json")
GUARD_CONF = 0.20          # 守卫类地板(money_safety: 危险检测最大灵敏度)
BODY_Y = 0.12


def _model(domain: str):
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if domain == "ui":
        ent = reg["ui"]
        path = ent["versions"][ent["active"]]["path"]
    elif domain == "battle":
        import re
        vers = reg["battle_heads"]["versions"]
        vn = max((v for v in vers if re.fullmatch(r"v\d+", v)),
                 key=lambda x: int(x[1:]))
        path = vers[vn]["path"]
    else:
        raise ValueError(domain)
    from ultralytics import YOLO
    print(f"  [{domain}] {path}")
    return YOLO(path)


def _dets(model, img):
    r = model.predict(img, conf=GUARD_CONF, iou=0.5, imgsz=960,
                      verbose=False)[0]
    out = []
    if r.boxes is not None:
        H, W = r.orig_shape
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            out.append((model.names[int(b.cls[0])], float(b.conf[0]),
                        x1 / W, y1 / H, x2 / W, y2 / H))
    return out


def check_purchase_dialog(dets) -> bool:
    names = {n for n, *_ in dets}
    if not ({"取消键", "确认键"} <= names):
        return False
    body = [(n, c) for n, c, x1, y1, x2, y2 in dets if y1 > BODY_Y]
    stepper = {"加号", "MAX_可点击", "MIN_灰色"}
    return any(n in stepper or n == "体力" for n, _ in body)


CHECKS = {"purchase_dialog": check_purchase_dialog}


def main():
    want = None
    if "--domain" in sys.argv:
        want = sys.argv[sys.argv.index("--domain") + 1]
    manifest = json.loads((REG / "manifest.json").read_text(encoding="utf-8"))
    cases = [c for c in manifest["cases"]
             if want is None or c["domain"] == want]
    models = {}
    veto_fail = warn_fail = 0
    print(f"== 事故回归 {len(cases)} 用例 ==")
    for c in cases:
        dom = c["domain"]
        if dom not in models:
            models[dom] = _model(dom)
        img = imread_any(str(REG / c["frame"]))
        got = CHECKS[c["check"]](_dets(models[dom], img))
        ok = got == c["expect"]
        tag = "PASS" if ok else ("⛔VETO-FAIL" if c.get("veto") else "⚠FAIL")
        if not ok:
            if c.get("veto"):
                veto_fail += 1
            else:
                warn_fail += 1
        print(f"  {tag:<12} {c['id']:<28} expect={c['expect']} got={got}"
              f"{'  (known_overblock)' if c.get('known_overblock') else ''}")
    print(f"\nveto失败 {veto_fail} | 功能失败 {warn_fail} | "
          f"通过 {len(cases) - veto_fail - warn_fail}/{len(cases)}")
    if veto_fail:
        print("⛔ 金钱防线回归失败 — 一票否决, 禁止出货!")
        sys.exit(2)
    if warn_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
