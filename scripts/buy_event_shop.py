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
用法: py -u scripts/buy_event_shop.py            # tab1+2(tab3 永跳, 见 VALID_PRICES)
      py -u scripts/buy_event_shop.py --tab 1    # 只跑第 N tab
      py -u scripts/buy_event_shop.py --plan     # ⭐只读不买: 盘店+候选清单(人审入口)
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
TAB_XY = [(220, 410), (220, 653), (220, 883)]   # 左侧三货币 tab(4K, 2026-07-28 帧标定)

# ⭐合法价目集合(2026-07-28 逐帧人工盘店标定, 奇普托斯活动) —— playbook 债
# 「行列位置→已知价目表映射, OCR 降级校验」的落地形态: 读出的单价必须
# ∈ 本 tab 素材价集合, 否则= OCR 串位(上期 540→56 / 95次→95 实害)丢弃该位。
# tab1(卡带): 報告1/3/12/60 + BD女武神/狂獵 10/30/100/300 + 鐵系 5/15/50/200
# tab2(银币): 強化石1/4/15/60 + 筆記女武神/狂獵 5/15/50/200 + 星象盤 5/15/50/200
# ⛔家具行不靠价集挡(tab1 家具 300 与狂獵BD 300 同价): 行内判据见 sweep_screen。
VALID_PRICES = {
    1: {1, 3, 5, 10, 12, 15, 30, 50, 60, 100, 200, 300},
    2: {1, 4, 5, 15, 50, 60, 200},
    3: None,   # tab3 本轮不跑(用户 2026-07-28: 只买前两币; 符咒是盒抽产物)
}


def main():
    only_tab = None
    if "--tab" in sys.argv:
        only_tab = int(sys.argv[sys.argv.index("--tab") + 1])
    # --plan: 只读不买(全店盘点+购买顺序模拟), 逐帧门控的人审入口
    plan_only = "--plan" in sys.argv
    reg = json.loads((ROOT / "data" / "model_registry.json")
                     .read_text(encoding="utf-8"))
    from ultralytics import YOLO
    ui = YOLO(reg["ui"]["versions"][reg["ui"]["active"]]["path"])
    adb = AdbInput()
    adb.connect()
    tap = lambda x, y: adb._shell(f"input tap {x} {y}")  # noqa: E731
    ocr = _get_ocr()

    # 飞轮: 商店帧=购买按钮/价格条/确认框补标素材(task#18), ADB 干净帧
    import cv2, time as _t
    fly_dir = ROOT / "data" / "raw_images" / \
        f"flywheel_event_shop_{_t.strftime('%Y%m%d')}"
    fly_dir.mkdir(parents=True, exist_ok=True)
    _fly_i = [0]
    _raw_cap = adb.capture_frame

    def _cap():
        fr = _raw_cap()
        if fr is not None:
            try:
                cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1] \
                    .tofile(str(fly_dir / f"frame_{_fly_i[0]:04d}.jpg"))
                _fly_i[0] += 1
            except Exception:
                pass
        return fr
    adb.capture_frame = _cap

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
            # 千分位防线: "1,200"经findall拆成["1","200"]取[0]=1 →
            # 1200家具穿过FURNITURE_PRICE过滤被当1块钱买(审计⑧) —
            # 先剥分隔符再取第一段完整数字
            raw = "".join(t for t, _ in (txts or []))
            raw = raw.replace(",", "").replace(".", "").replace(" ", "")
            digs = re.findall(r"\d+", raw)
            return int(digs[0]) if digs else None
        except Exception:
            return None

    def _body_close_btn(d):
        """对话框 body 区(0.15<cy<0.80)内的 取消键/叉叉 — 商店页自身
        右上返回X(cy~0.05)绝不点(点=退出商店整页, 审计⑥)。
        关闭是清理动作非守卫: conf≥0.5 防误检点到商品误开购买框."""
        for n in ("取消键", "叉叉"):
            b = next((x for x in d if x[0] == n and x[1] >= 0.5
                      and 0.15 < (x[3] + x[5]) / 2 / x[7] < 0.80), None)
            if b:
                return b
        return None

    def buy_one(px, py) -> str:
        """点购买 → MAX → 白名单闸确认。返回 bought/blocked/no_dialog."""
        tap(px, py)
        time.sleep(2.5)
        fr = adb.capture_frame()
        d = dets(fr, 0.20)
        names = {x[0] for x in d}
        if "确认键" not in names:          # 没弹框 = 买不起/售罄
            # 兜底: 只关 body 内的残留框(位置过滤, 审计⑥)
            b = _body_close_btn(d)
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
            time.sleep(2.0)
            # 「獲得獎勵! TOUCH TO CONTINUE」全屏弹窗(2026-07-15 实锤
            # 挡死后续检出) → **先 cls 验证横幅在场才点屏中**——
            # 盲点(1920,1750)在横幅缺席时会点进商店网格误开购买框,
            # 后面无闸点确认=白名单闸旁路(审计⑤根治)
            for _ in range(3):
                fr = adb.capture_frame()
                d = dets(fr, 0.30)
                if any(x[0] in ("獲得獎勵", "获得奖励", "点击继续字样")
                       for x in d):
                    tap(1920, 1750)
                    time.sleep(1.5)
                    continue
                break
            # 残留框只走关闭语义(取消/叉, body 位置过滤), 绝不点确认
            b = _body_close_btn(d)
            if b:
                tap(int((b[2] + b[4]) / 2), int((b[3] + b[5]) / 2))
                time.sleep(1.5)
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

    # ⛔read_balance 换 rec-only(2026-07-30 结案, task#35「丢首位」机制查清):
    # 屏上 3,135 被旧路径一致读成 135 — 不是裁剪几何, 是 run_digit_ocr 走
    # **det+rec 全管线**, det 在浅底胶囊上把「3,」段漏检(只框住 135), 任何
    # 窗口组都救不了(四窗 4:0 一致地错, 后缀合票无从救 — pass1 因此提前
    # 收工真损失)。rec-only 全框直读 '3,135' conf .80 完美 — 与 read_price
    # 同法。⚠rec-only 的误差模型与 det 相反: 垃圾出现在**前缀**(图标圆环/
    # 左缘阴影被硬解成 9/2, 实测 [865,865,9865] / 0728 语料 2↔22 交替),
    # 不再有截位短数 → ①窗口全部必含完整数字(剥图标窗 l0.20 会半切首位,
    # 弃用) ②合票方向反转: 后缀关系时票归**短数**(长数=幻影前缀)。
    _BAL_WINDOWS = ((0.0, 1.0), (0.06, 1.0), (0.12, 1.0))

    def read_balance(fr, d):
        """余额 = 「货币数量显示区域」cls 框内 rec-only 直读(det 会漏首段)。

        读不出 → None, 调用方降级(不做买不起过滤, 走 MAX/no_dialog 兜底) —
        余额是增强层, fail-open 到旧行为, 不阻塞购买。"""
        b = next((x for x in d if x[0] == "货币数量显示区域"), None)
        if b is None:
            return None
        _, _, x1, y1, x2, y2, W, H = b
        bw = x2 - x1
        vals = []
        for lcut, rcut in _BAL_WINDOWS:
            crop = fr[int(y1):int(y2), int(x1 + bw * lcut):int(x1 + bw * rcut)]
            if crop.size == 0:
                continue
            try:
                out = ocr.text_recognizer([crop])
                txts = out[0] if isinstance(out, tuple) else out
                raw = "".join(t for t, _ in (txts or []))
            except Exception:
                continue
            digs = re.sub(r"[^\d]", "", raw)
            if digs and 0 <= int(digs) <= 99999:
                vals.append(int(digs))
        # ⛔孤证不采 + ⭐后缀合票**归短数**(2026-07-30 反转): rec-only 下窗口
        # 必含完整数字, 短数不可能是截位; 长数=短数+幻影前缀(图标/阴影硬解)。
        # [865,865,9865] → 票归 865。合票后仍 <2 票 → None 降级(安全方向)。
        votes = {}
        for v in vals:
            votes[v] = votes.get(v, 0) + 1
        for v in sorted(votes, key=lambda x: len(str(x))):
            total = sum(c for u, c in votes.items()
                        if str(u) == str(v) or str(u).endswith(str(v)))
            if total >= 2:
                return v
        return None

    def verify_tab(d, want_i):
        """当前选中币种 = 「货币_已选择」cy 在左侧 tab 列的序位(用户指路)。"""
        sel = next((x for x in d if x[0] == "货币_已选择"), None)
        if sel is None:
            return None
        tabs = sorted([x for x in d if x[0] in ("货币", "货币_已选择")],
                      key=lambda x: (x[3] + x[5]) / 2)
        for i, t in enumerate(tabs, 1):
            if t is sel:
                return i
        return None

    def scan_items(tab_i):
        """当前屏 → [(price, cx, ry)] 候选(价集校验+家具行剔除)。纯读不点。"""
        fr = adb.capture_frame()
        d = dets(fr, 0.28)
        scan_items.last_frame, scan_items.last_dets = fr, d
        # ⛔conf 分界已被对抗审推翻(全量 645 位置+153 位人审): 售罄位 79%
        # conf≥0.5(23 个 ≥0.90), 亮态还有 0.065/0.130 — conf 编码的是训练
        # 熟悉度, 与亮暗态无关, 任何阈值都切不开。我上一版只看一帧 3 个位就
        # 定 0.5 = 单帧标定谬误(与上期 read_price 同病)。
        # ⭐真信号 = 购买按钮 det 框内灰度均值: 暗态(售罄) 75-97 / 亮态
        # 115-145, 双峰零重叠且与人审 100% 吻合。列级过滤(比行锚准: 同行
        # 亮暗混合也能挡)。⚠绝对亮度信号, 只在固定 MuMu 环境成立。
        anchor_ys = sorted({int((y1 + y2) / 2)
                            for n, c, x1, y1, x2, y2, W, H in d
                            if n == "购买"})
        dark_spots = []
        for n, c, x1, y1, x2, y2, W, H in d:
            if n == "购买":
                _g = cv2.cvtColor(fr[int(y1):int(y2), int(x1):int(x2)],
                                  cv2.COLOR_BGR2GRAY).mean()
                if _g < 105:
                    dark_spots.append(((x1 + x2) / 2, (y1 + y2) / 2))
        # ⛔跨行 ±ROW_DY 补全已删(2026-07-28 --plan 实锤): 补出来的**幽灵行**
        # 没有按钮, 读价窗口卡在邻行「可購買N次」黑条上, 30/12 这类次数恰好
        # ∈ 价集 → 校验漏过。v14 购买按钮检出已稳(实测 7/8 conf 0.90-0.95,
        # 漏的恰是售罄位) → 只信检出锚行, 同行 4 列补漏检即可。
        row_ys = set()
        for ay in anchor_ys:
            # 行闸 850: ry<850 时 read_price 读区(ry-150 起)伸进
            # 头部货币栏, 余额数字被当单价(审计⑦)
            if 850 < ay < 2050 and not any(
                    abs(ay - e) < 200 for e in row_ys):
                row_ys.add(ay)
        valid = VALID_PRICES.get(tab_i)
        items = []
        for ry in row_ys:
            row = []
            for cx in COLS:
                price = read_price(fr, cx - BTN_W / 2, ry - BTN_H / 2,
                                   cx + BTN_W / 2, ry + BTN_H / 2)
                if price is None:
                    continue
                row.append((price, cx, ry))
            # ⭐家具行判别(2026-07-28 帧标定): 两 tab 家具行价签都是
            # (200,200,300,2000) — **行内出现重复价** 或 行内 max>1000。
            # 素材行是严格 4 档递增, 绝无重复价。价格过滤挡不住 200/300
            # 档家具(tab1 狂獵BD 也是 300), 只有整行判才干净。
            prices = [p for p, _, _ in row]
            if len(prices) != len(set(prices)) or (
                    prices and max(prices) > FURNITURE_PRICE):
                print(f"    行 y={ry}: 价签 {sorted(prices)} → 家具行, 整行跳过",
                      flush=True)
                continue
            for price, cx, ry2 in row:
                # ⭐价集校验(playbook 债落地): 读出的价 ∉ 本 tab 价目表 =
                # OCR 串位(540→56 / 95次→95 实害) → 该位丢弃 fail-closed。
                if valid is not None and price not in valid:
                    print(f"    ⚠read_price {price} ∉ tab{tab_i} 价目集 "
                          f"@({cx},{ry2}) → 弃(疑 OCR 串位)", flush=True)
                    continue
                # 暗态售罄位不进候选(gray 双峰判定, 见 dark_spots 注释)
                if any(abs(cx - dx) < 100 and abs(ry2 - dy) < 100
                       for dx, dy in dark_spots):
                    print(f"    售罄(暗态按钮) @({cx},{ry2}) → 跳过", flush=True)
                    continue
                items.append((price, cx, ry2))
        return items

    def sweep_screen(tab_i, min_price=0) -> int:
        """检出「购买」按钮当行锚 → 网格补全 → 价集校验 → 价格降序买。

        min_price: 本遍只买 ≥ 该价的位置。⭐两遍策略(2026-07-28 tab1 流水
        实锤): 单遍"取景屏内降序"不是全局降序 — 取景0 的 1/3 币低档先花钱,
        滚到后面高純鋼@200 落空。第一遍全店只吃 ≥100 高价档, 第二遍清低档。"""
        bought = 0
        for _round in range(12):            # 每买一次重扫(余额变)
            items = [t for t in scan_items(tab_i) if t[0] >= min_price]
            if not items:
                return bought
            # ⭐数学闸(用户 2026-07-28: 不许"点了才知道买不起"):
            # 余额可读时, 单价 > 余额的位置直接不点。⛔tab 身份校验同帧做 —
            # 读的是哪个币的余额必须与正在买的 tab 一致, 否则跳过本轮。
            _fr = getattr(scan_items, "last_frame", None)
            _d = getattr(scan_items, "last_dets", [])
            _tab_now = verify_tab(_d, tab_i)
            if _tab_now is not None and _tab_now != tab_i:
                print(f"    ⛔tab 身份不符: 选中={_tab_now} 预期={tab_i} — 重点 tab",
                      flush=True)
                tap(*TAB_XY[tab_i - 1])
                time.sleep(2.5)
                continue
            bal = read_balance(_fr, _d) if _fr is not None else None
            if bal is not None:
                before = len(items)
                items = [t for t in items if t[0] <= bal]
                if before != len(items):
                    print(f"    余额 {bal}: 砍掉 {before - len(items)} 个买不起的位置",
                          flush=True)
                if not items:
                    print(f"    余额 {bal} < 本遍最低价 → 本遍收工", flush=True)
                    return bought
            items.sort(key=lambda t: -t[0])   # 单价降序
            # 拉黑距离匹配: ry 来自检出锚, ±5px 帧间抖动会击穿精确
            # tuple 匹配 → 售罄位被反复点(审计⑨)
            items = [t for t in items
                     if not any(abs(t[1] - dx) < 40 and abs(t[2] - dy) < 40
                                for dx, dy in _dead)]
            if not items:
                return bought
            price, px, py = items[0]
            if plan_only:
                for p2, x2, y2 in items:
                    print(f"    [plan] 单价{p2} @({x2},{y2})", flush=True)
                return 0
            print(f"    买单价{price} @({px},{py})", flush=True)
            r = buy_one(px, py)
            print(f"      → {r}", flush=True)
            if r == "bought":
                bought += 1
                # 买后对账: 新余额落到日志(能读出时), Δ=本次真实花费
                _fr2 = adb.capture_frame()
                _b2 = read_balance(_fr2, dets(_fr2, 0.28))
                if _b2 is not None:
                    _tag = f" (Δ-{bal - _b2})" if bal is not None else ""
                    print(f"      对账: 余额 → {_b2}{_tag}", flush=True)
            else:
                _dead.add((px, py))     # 售罄/买不起: 本 tab 内不再点
        return bought

    def on_shop_page(d) -> bool:
        """活动商店页判据: 左侧货币 tab 列(≥2 个)或余额胶囊在屏。"""
        tabs = [x for x in d if x[0] in ("货币", "货币_已选择") and x[1] >= 0.5]
        cap = any(x[0] == "货币数量显示区域" and x[1] >= 0.5 for x in d)
        return len(tabs) >= 2 or cap

    def ensure_shop_page() -> bool:
        """⛔到达闸(2026-07-30 用户当场抓包): 旧版没有任何"我在商店页吗"的
        判据, 假定启动时人已把游戏放在商店页 — 今天停在任务 hub, tab 坐标
        点空 + verify_tab None 被放行 + scan 扫不到但 20 次盲滑照做(swipe
        不带落点, step_walk 盲拍闸只管 click = 守卫盲区)。
        修: 不在商店页 → cls 锚定逐步导航(每步验帧), 到不了绝不滑动。
        导航链: 大厅[任务大厅入口] → 任务hub[距离结束还剩] → 活动hub
        [活动商店(底栏, conf≥0.85 — 贈品交換页会被同 cls 捡 0.66)] → 商店。"""
        for hop in range(6):
            fr = _cap()
            if fr is None:
                print("⛔抓帧失败 — 停", flush=True)
                return False
            d = dets(fr, 0.5)
            if on_shop_page(d):
                if hop:
                    print(f"  [nav] 已到活动商店页({hop} 跳)", flush=True)
                return True
            # ⚠「活动商店」(底栏)只在 Quest tab 取景下检得出(0.97), Story tab
            # 上零检出(2026-07-30 实测, 训练分布缺) → 活动页内先切 Quest tab。
            for cls_name, min_c in (("活动商店", 0.85),
                                    ("活动quest", 0.60),
                                    ("距离结束还剩", 0.60),
                                    ("任务大厅入口", 0.60)):
                cands = [x for x in d if x[0] == cls_name and x[1] >= min_c]
                if cands:
                    b = max(cands, key=lambda x: x[1])
                    cx, cy = int((b[2] + b[4]) / 2), int((b[3] + b[5]) / 2)
                    print(f"  [nav] {cls_name} {b[1]:.2f} → tap({cx},{cy})",
                          flush=True)
                    tap(cx, cy)
                    time.sleep(3)
                    break
            else:
                print(f"⛔不在商店页且无导航锚点(检出 {len(d)} 框) — "
                      f"fail-closed 停, 绝不盲滑", flush=True)
                return False
        print("⛔6 跳仍未到商店页 — 停", flush=True)
        return False

    if not ensure_shop_page():
        sys.exit(1)

    total = 0
    for ti, (tx, ty) in enumerate(TAB_XY, 1):
        if only_tab and ti != only_tab:
            continue
        # ⛔tab3(符咒)本轮不跑(用户 2026-07-28): 前两币=角色素材;
        # 符咒出自盒抽小活动, 花法(神名碎片)等用户拍板, 绝不自动买。
        if VALID_PRICES.get(ti) is None:
            print(f"[tab{ti}] 跳过(不在本轮购买范围)", flush=True)
            continue
        print(f"[tab{ti}]", flush=True)
        _dead.clear()
        tap(tx, ty)
        time.sleep(3)
        # 「购买」cls 只对顶部行位置检出稳(训练分布) → 小步滑动让每行
        # 轮流滚到顶部, 每步用 cls 检出买(cls 主导, 滑动只是取景)
        # ⭐两遍: PASS1 只买 ≥100(高价档吃满预算), PASS2 清低档。
        for pass_i, floor in ((1, 100), (2, 0)):
            adb._shell("input swipe 2400 700 2400 1600 400")   # 回顶
            time.sleep(2)
            # 售罄拉黑按 (px,py) 记 — 两遍间滚动相位不同, 不跨遍复用
            _dead.clear()
            for screen_i in range(5):
                if screen_i:
                    adb._shell("input swipe 2400 1300 2400 750 500")
                    time.sleep(2)
                n = sweep_screen(ti, min_price=floor)
                total += n
                print(f"  pass{pass_i} 取景{screen_i} 成交 {n}", flush=True)
        # ⭐经济台账(2026-07-28 用户: 让程序自己推断刷哪关): 每 tab 跑完记
        # 「购后余额」— planner 用它判断该币商店还饿不饿(买完清零=还缺,
        # 剩一大笔=饱和停刷)。选关推断的唯一数据源, 零新增 OCR 依赖。
        if not plan_only:
            try:
                _fr = adb.capture_frame()
                _bal = read_balance(_fr, dets(_fr, 0.28))
                _eco_p = ROOT / "data" / "event_economy.json"
                _eco = (json.loads(_eco_p.read_text(encoding="utf-8"))
                        if _eco_p.exists() else {})
                _eco.setdefault("tabs", {})[str(ti)] = {
                    "balance_after_buy": _bal,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                _eco_p.write_text(json.dumps(_eco, ensure_ascii=False,
                                             indent=1), encoding="utf-8")
                print(f"  [economy] tab{ti} 购后余额={_bal} → 台账", flush=True)
            except Exception as _e:
                print(f"  [economy] 台账写入失败(不阻塞): {_e}", flush=True)
    print(f"done 总成交 {total}", flush=True)


if __name__ == "__main__":
    main()
