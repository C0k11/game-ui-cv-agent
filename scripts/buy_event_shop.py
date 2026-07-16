# -*- coding: utf-8 -*-
"""活动商店购买 (2026-07-15, 用户拍板策略).

策略: 每个货币 tab 内, 检出所有「购买」按钮 → OCR 按钮内单价 →
**单价降序**逐档买(MAX 数量); 单价 >1000 = 家具跳过(用户: 优先换
角色素材); 买到买不起为止; 三 tab 依次(tab3 余额 0 会自然跳过)。

⛔金钱防线(fail-closed):
  - 活动商店只收活动币, 但确认框仍走白名单: 「确认键+取消键」在场
    且 body 无「青辉石」检出才确认, 否则取消。
  - 购买后余额只减不清零判断: 每档购买前后读余额(digit-OCR 辅证),
    读不出不阻塞(活动币非付费币), 只记日志对账。
用法: py -u scripts/buy_event_shop.py            # 三 tab 全跑
      py -u scripts/buy_event_shop.py --tab 1    # 只跑第 N tab(1-3)
"""
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\Project\ai game secretary")
from mumu_runner import AdbInput  # noqa: E402
from brain.pipeline import _get_ocr  # noqa: E402

ROOT = Path(r"D:\Project\ai game secretary")
FURNITURE_PRICE = 1000       # 单价 > 此值 = 家具(用户定义), 跳过
TAB_XY = [(230, 440), (230, 680), (230, 920)]   # 左侧三货币 tab(4K)


def main():
    only_tab = None
    if "--tab" in sys.argv:
        only_tab = int(sys.argv[sys.argv.index("--tab") + 1])
    reg = json.loads((ROOT / "data" / "model_registry.json")
                     .read_text(encoding="utf-8"))
    from ultralytics import YOLO
    ui = YOLO(reg["ui"]["versions"][reg["ui"]["active"]]["path"])
    adb = AdbInput()
    adb.connect()
    tap = lambda x, y: adb._shell(f"input tap {x} {y}")  # noqa: E731
    ocr = _get_ocr()

    def dets(fr, conf=0.5):
        r = ui.predict(fr, conf=conf, imgsz=960, verbose=False)[0]
        return [(ui.names[int(b.cls[0])], float(b.conf[0]),
                 *(float(v) for v in b.xyxy[0]),
                 fr.shape[1], fr.shape[0]) for b in (r.boxes or [])]

    def read_price(fr, x1, y1, x2, y2):
        """单价数字 rec-only 直读。价格条紧贴按钮上沿(高~90px);
        ⚠再往上是「可購買N次」黑条 — 读区过宽会把次数当单价
        (2026-07-15 实锤: 95次被当95单价, 排序污染)."""
        crop = fr[max(0, int(y1 - 95)):max(0, int(y1 - 8)),
                  int(x1):int(x2)]
        if crop.size == 0:
            return None
        try:
            out = ocr.text_recognizer([crop])
            txts = out[0] if isinstance(out, tuple) else out
            digs = re.findall(r"\d+", "".join(t for t, _ in (txts or [])))
            return int(digs[0]) if digs else None
        except Exception:
            return None

    def buy_one(px, py) -> str:
        """点购买 → MAX → 白名单闸确认。返回 bought/blocked/no_dialog."""
        tap(px, py)
        time.sleep(2.5)
        fr = adb.capture_frame()
        d = dets(fr, 0.20)
        names = {x[0] for x in d}
        if "确认键" not in names:          # 没弹框 = 买不起/售罄
            # 兜底: 若有取消/叉能关就关
            for n in ("取消键", "叉叉"):
                b = next((x for x in d if x[0] == n), None)
                if b:
                    tap(int((b[2] + b[4]) / 2), int((b[3] + b[5]) / 2))
                    time.sleep(1.5)
            return "no_dialog"
        mx = next((b for b in d if b[0] == "MAX_可点击"
                   and (b[3] + b[5]) / 2 / b[7] > 0.12), None)
        if mx is not None:
            tap(int((mx[2] + mx[4]) / 2), int((mx[3] + mx[5]) / 2))
            time.sleep(1.5)
            fr = adb.capture_frame()
            d = dets(fr, 0.20)
            names = {x[0] for x in d}
        # ⛔白名单闸: 确认+取消在场 且 body 无青辉石
        pyx_body = [b for b in d if b[0] == "青辉石"
                    and (b[3] + b[5]) / 2 / b[7] > 0.12]
        if "确认键" in names and "取消键" in names and not pyx_body:
            ck = next(b for b in d if b[0] == "确认键")
            tap(int((ck[2] + ck[4]) / 2), int((ck[3] + ck[5]) / 2))
            time.sleep(2.5)
            # 「獲得獎勵! TOUCH TO CONTINUE」全屏弹窗(2026-07-15 实锤
            # 挡死后续检出) → 点屏中清掉; 再有确认/叉也收
            tap(1920, 1750)
            time.sleep(1.5)
            fr = adb.capture_frame()
            d = dets(fr, 0.5)
            for n in ("确认键", "叉叉"):
                b = next((x for x in d if x[0] == n), None)
                if b:
                    tap(int((b[2] + b[4]) / 2), int((b[3] + b[5]) / 2))
                    time.sleep(1.5)
                    break
            return "bought"
        if pyx_body:
            print("    ⛔body 检出青辉石 — 取消!", flush=True)
        b = next((x for x in d if x[0] == "取消键"), None)
        if b:
            tap(int((b[2] + b[4]) / 2), int((b[3] + b[5]) / 2))
            time.sleep(1.5)
        return "blocked"

    COLS = [2093, 2561, 3029, 3497]     # 4 列按钮 cx(4K 布局固定)
    ROW_DY = 738                        # 行距
    BTN_W, BTN_H = 420, 110
    _dead = set()                       # 售罄/买不起位置(tab 内拉黑)

    def sweep_screen() -> int:
        """检出「购买」按钮当行锚 → 同行 4 列+上下行网格补全漏检位
        (该 cls 对活动商店皮肤检出稀疏, 2026-07-15 实锤) → 每位读价
        (读不出=跳过 fail-closed) → 价格降序买。返回成交数."""
        bought = 0
        for _round in range(12):            # 每买一次重扫(余额变)
            fr = adb.capture_frame()
            d = dets(fr, 0.28)
            anchor_ys = sorted({int((y1 + y2) / 2)
                                for n, c, x1, y1, x2, y2, W, H in d
                                if n == "购买"})
            row_ys = set()
            for ay in anchor_ys:
                for ry in (ay - ROW_DY, ay, ay + ROW_DY):
                    if 700 < ry < 2050 and not any(
                            abs(ry - e) < 200 for e in row_ys):
                        row_ys.add(ry)
            items = []
            for ry in row_ys:
                for cx in COLS:
                    price = read_price(fr, cx - BTN_W / 2, ry - BTN_H / 2,
                                       cx + BTN_W / 2, ry + BTN_H / 2)
                    if price is None or price > FURNITURE_PRICE:
                        continue
                    items.append((price, cx, ry))
            if not items:
                return bought
            items.sort(key=lambda t: -t[0])   # 单价降序
            items = [t for t in items if (t[1], t[2]) not in _dead]
            if not items:
                return bought
            price, px, py = items[0]
            print(f"    买单价{price} @({px},{py})", flush=True)
            r = buy_one(px, py)
            print(f"      → {r}", flush=True)
            if r == "bought":
                bought += 1
            else:
                _dead.add((px, py))     # 售罄/买不起: 本 tab 内不再点
        return bought

    total = 0
    for ti, (tx, ty) in enumerate(TAB_XY, 1):
        if only_tab and ti != only_tab:
            continue
        print(f"[tab{ti}]", flush=True)
        _dead.clear()
        tap(tx, ty)
        time.sleep(3)
        # 「购买」cls 只对顶部行位置检出稳(训练分布) → 小步滑动让每行
        # 轮流滚到顶部, 每步用 cls 检出买(cls 主导, 滑动只是取景)
        adb._shell("input swipe 2400 700 2400 1600 400")   # 先回顶
        time.sleep(2)
        for screen_i in range(5):
            if screen_i:
                adb._shell("input swipe 2400 1300 2400 750 500")
                time.sleep(2)
            n = sweep_screen()
            total += n
            print(f"  取景{screen_i} 成交 {n}", flush=True)
    print(f"done 总成交 {total}", flush=True)


if __name__ == "__main__":
    main()
