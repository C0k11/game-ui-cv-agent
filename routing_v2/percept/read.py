# -*- coding: utf-8 -*-
"""数字读取层 —— YOLO 定位图标，OCR 只读图标旁边的数字。

设计铁律（用户 2026-05-28 定，至今未变）:
  **YOLO cls 驱动全部点击；OCR 只干读数字这一件事。**
  YOLO 看不见某个 cls，正确做法是补训练数据，不是加 OCR 兜底把问题藏起来。

两条几何铁律（都是流血换的）:

【一】strip 的单位必须是**图标自身尺寸**，不能是屏幕比例。
  屏幕比例只在标定它的那个分辨率+宽高比上成立。本系统的帧来路不止一条，
  语料里实测出 **19 种分辨率、宽高比 1.4812~1.7927**。UI 是整体等比缩放的，
  所以"数字串相对图标的位置"是**布局常数**，"数字串占屏幕宽度的百分之几"不是。

【二】y 留白必须**逐币标定，且方向相反**，绝不能拿一个常数套所有货币。
  信用点要紧（0.65），AP 要松（1.60）。DB 文本检测器需要行外留白才肯出框，
  但留多了就吃进邻居。

【三】共识 ≠ 真值（README §A10，2026-08-07 差点酿成事故）:
  青辉石这一格当初留着"多参数网格共识"标出来的 (6.0, 1.20)，27 帧实测 20 对
  **7 错**，把 18,036 读成 18,003  kill-switch 报 MONEY BREACH 急停整条
  pipeline，而余额一分没少。按人眼真值重标后 (5.5, 0.45)  27/27 零错读。
  **危害双向：会读低就会读高，读高会掩盖真实掉钱。**
   任何新增的数值判据，标定必须挂人眼真值，禁止用参数共识当真值。
"""
from __future__ import annotations

import re
import threading
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from routing_v2.percept.observe import Box, Observation, Rect

# ── cls 名（避免循环 import，这里直接写字面量；state/vocab.py 是权威表）──
AP = "体力"
CREDIT = "信用点"
PYROXENE = "青辉石"
ARENA_COIN = "战术大赛商店货币"

# 每种货币的数字 strip，单位 = **图标宽 iw / 图标高 bh**：(x_from, x_to, y_pad)
#
#  货币   | 值           | 出处                                   | 实测
#  ------ | ------------ | -------------------------------------- | ---------------
#  体力   | (.10,5.0,1.6)| 2026-07-27 网格标定                     | 88.1% (n=202)
#  信用点 | (.10,5.5,.65)| **人眼真值 23,390,459** 重标            | 10/10
#  青辉石 | (.10,5.5,.45)| **人眼真值 18,036** 重标（27帧×42组）    | 27/27
#
# 信用点的 x_to 用网格推荐的 (5.5,0.50) 会读出 `390459`（吞掉前三位）——
#   8 位长数的**系统性截断可以自己成为共识**，这就是 §A10 的具体形态。
# 青辉石 y_pad 的失效是**几何性**的：≤0.35 纵向切掉数字读成 `36`；
#   ≥0.55 吃进邻居读成 `18003`；0.40~0.50 是安全块，取中心 0.45。
# (x_from, x_to, y_pad_icon, pad_px_min, pad_px_max)
# 后两个是 y 留白的**绝对像素**上下限 —— 见 PAD_FLOOR_PX 那段的实测依据。
# None = 不设限。
STRIP = {
    #  AP 的 1.60 是刻意放很松的（要连 "/240" 一起框住），**不设上限**，
    #  否则会压坏它（1440p 下 1.60×57.4 = 92px，实测 12/12）。
    AP:       (0.10, 5.0, 1.60, 30.0, None),
    #  信用点 8 位长数。2026-08-12 按人眼真值 **56,701,774** 重标
    #  （scrcpy 1440p，**14 帧**扫 x_from × x_to）:
    #      原始 (0.10,5.5)  8 对 / 3 错 / 3 None   错法 `6701774`(切首位)
    #      (0.10,6.5)       9 对 / 2 错 / 3 None   错法 `56701`(切末三位)
    #      (0.26,6.5)       9 对 / 2 错 / 3 None   错法 `5701774`  ⇐ 取它
    #  **这只是边际改进，不是根治**（89 对、32 错就这么多）。别被小样本骗:
    #    我先用 2 帧扫出过"(0.30,6.5) 零错读 "，扩到 14 帧就变成 8 对 4 错 ——
    #    **参数扫描的样本量不够时，最优值本身就是噪声**。
    #  结论：**单帧 OCR 读长数天生不可靠，真正的防线是 ledger 的多帧共识**
    #    （位数变少判截断+基线自愈 / 位数变多等复读一致）。标定只负责
    #    把错读压到少数，不负责杜绝。
    #  为什么宁可多 None 也不要错读：None 会被台账正确拒绝，**错读会成为基线**。
    #    08-12 live 实锤这条恶性循环两次，方向还相反 ——
    #       一帧读成 `56701` 当了基线  之后正确的 `56701774` 全被判
    #         "位数多于基线 = 读大嫌疑"挡掉；
    #       一帧读成 `566701774`(9位) 当了历史最大  正确的 8 位全被判
    #         "位数 < 历史最大 = OCR 截断"挡掉。
    #    [[read_layer_icon_units]]「系统性截断可以自己成为共识」的两个形态。
    CREDIT:   (0.26, 6.5, 0.65, 30.0, 42.0),
    #  青辉石 2026-08-08 按人眼真值 **20,176** 重标（1440p，12 帧 × 120 组参数）:
    #    旧参数 (0.10, 5.5, 0.45)  **0/12 全错**（读成 `176`，前两位被切）
    #    x_from 0.10 太靠左会吃到图标  读成 `200176`
    #    定 (0.25, 5.5, 0.80)，安全块中心，实测 12/12。
    PYROXENE: (0.25, 5.5, 0.80, 30.0, 42.0),
    #  战术大赛商店货币 2026-08-10 按人眼真值 **3,038** 标定（1440p，
    #    5×4×3 = 60 组参数扫）:
    #    **y_pad ≤ 0.45 一律读成 `038`** —— 首位 3 被纵向切掉，而且
    #    x_to 从 5.0 扫到 9.0 全是这个结果  失效是**几何性**的，加宽 x 救不了
    #    （和青辉石那次同形）。0.55 起 5 个 x_to 全部读出 3038。
    #    x_to=9.0 配 pad_max≥42 会过读成 `0038`（吃进右边邻居） x_to 取 6.0、
    #      pad_max 收到 30.0。
    #    这一格读**低**的后果是"钱不够不买"（漏做，无损）；读**高**才危险
    #      （买不起还去点）—— 但 memory 铁律「会读低就会读高」，所以余额闸
    #      仍旧 fail-CLOSED：读不出就不买。
    ARENA_COIN: (0.10, 6.0, 0.55, 5.5, 30.0),
}
# **已知缺口（2026-08-08 实测，别忘了）**：青辉石这组参数是在 **18,036**
#    上标的，余额涨到 **20,176** 后实测被读成 **1176** —— **首位被吃掉**。
#    多帧投票救不了（错读是稳定的，两次都错成同一个值）。
#     唯一挡住它的是 `act/ledger.py` 的**位数收缩闸**（位数变少且掉一大半
#       判为截断不采信）。所以**任何绕开 ledger 直接用 read_topbar 的地方
#      都不可信**，尤其别拿它当金钱判据。
#     待办：按人眼真值重标（要覆盖 5 位数的高低两端，别只标一个值）。
#      这正是 §A10「共识≠真值」那条的续集 —— 单点标定跨不过量级。
STRIP_DEFAULT = (0.10, 4.8, 0.80, 30.0, 42.0)

# 票券计数器专用（悬赏/JFD/大赛/课程表）。
# x 要**给足**：同一个计数器有两种布局，标签字数不同（`持有票券` 4 字 /
#   `懸賞通緝票券` 6 字），窄的右界会把数字整个推出去 —— 旧参数在全语料 4K 上
#   **0/211 帧**读得出，放宽后 **211/211**（2026-07-27）。
# y 留白是 **DB 文本检测器的开关**，不是"多切一点保险"：0.4bh 会让整条 strip
#   返回**空**（不是读错，是 det 根本不出框）。加宽 x / 放大 2-3× / 拉对比度
#   **全部无效**。0.80 是实测能出框的值。
TICKET_STRIP = (0.10, 6.5, 0.80, 30.0, 46.0)

# **长标签票种要更宽的 x_to**（2026-08-08 悬赏 live 实锤）:
#   悬赏列表页写的是「懸賞通緝票券 0/6」—— 六个字的标签把 6.5 个图标宽
#   全吃掉，条带右缘正好切在数字前  read_ticket 永远 None 
#   「票数读不出fail-closed」在有入场键时**根本不拦**，0 票也会去开面板。
#   （课程表票标签短，6.5 就够 —— 所以 schedule 读得出而 bounty 全瞎。）
TICKET_STRIP_LONG = (0.10, 13.0, 0.80, 30.0, 46.0)
_TICKET_STRIPS = {
    "悬赏通缉票": TICKET_STRIP_LONG,
    "学院交流会票": TICKET_STRIP_LONG,      # 「學院交流會票券」标签同样长
}

# 顶栏 region —— **任何顶栏读数/角落按钮锚定必须带 region**。
# 840AP 事故的根因就是 find_cls 全屏 argmax 没带 region，弹窗体内的体力图标
# 抢走了顶栏锚点。顶栏 cy < 0.10。
TOPBAR: Rect = (0.0, 0.0, 1.0, 0.10)

_ocr = None
_ocr_lock = threading.Lock()

# **OCR 用量计数**（用户 2026-08-08 再次强调的分工）:
#   「我们是 YOLO 的识别为主导，而不是 OCR，OCR 只是特定区域看数字
#     而且只有特定的时候开始扫描」
#    全 bot 只有这个文件会调 OCR，而且只在两种时候：
#       台账采样（每 8s 一次，且只在顶栏锚点可见时）
#       flow 主动读票数/AP（进页时读一次）
#   决策链上的**每一次点击都由 cls 决定**，OCR 一次都不参与。
#   这个计数会打进本轮报告 —— 如果它接近 tick 数，说明有人把 OCR 塞进热路径了。
STATS = {"ocr_calls": 0, "ocr_ms": 0.0, "ocr_none": 0}


def _engine():
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            from pathlib import Path
            from rapidocr_onnxruntime import RapidOCR
            root = Path(__file__).resolve().parents[2]
            rec = root / "data" / "ocr_model" / "ba_rec.onnx"
            try:
                # det_model_path / cls_model_path 必须显式给 None —— 不给的话
                #   RapidOCR 在取默认权重时 KeyError('model_path')，然后静默
                #   退回 CPU（2026-08-08 live 日志里就是这条）。
                _ocr = RapidOCR(det_use_cuda=True, det_model_path=None,
                                rec_use_cuda=True, cls_use_cuda=True,
                                cls_model_path=None,
                                rec_model_path=str(rec) if rec.exists() else None)
            except Exception as e:
                print(f"[read] OCR CUDA 起不来({e})，退 CPU", flush=True)
                _ocr = RapidOCR(rec_model_path=str(rec)) if rec.exists() else RapidOCR()
        return _ocr


# **y 留白的像素地板**（2026-08-08 实测定出来的，解决"换分辨率就读错"）
#
# 背景：y 留白是 **DB 文本检测器的开关** —— 留白不够它干脆不出框/只框到半截。
# 而"够不够"是**绝对像素**的事，不是图标倍数的事：
#
#   青辉石图标高    y_pad 0.45 的绝对像素   实测
#   ------------   --------------------   ----
#   4K   ~77px      34.7px                 27/27 全对（当初就是这么标的）
#   1440p 51.5px    **23.2px**             **0/12 全错**（读成 `176`，前两位被切）
#
#  同一组"图标倍数"参数，换个分辨率就从全对变全错。这正是用户要求的
#   「digit OCR 和 cls 的配合得适配各种分辨率下各种窗口大小」那一条。
#  修法 = 只加**地板**不加天花板：不够 30px 就补到 30px。
#   4K 下三种货币的 pad 本来都 >30px，**一个都不受影响**；
#   1440p 下青辉石从 23.2 补到 30，实测回到 12/12。
#   （不加天花板是因为 AP 的 y_pad=1.60 本来就是刻意放很松的，压它会伤 AP。）
PAD_FLOOR_PX = 30.0


def icon_strip(box: Box, x_from: float, x_to: float, y_pad: float,
               frame_h: Optional[int] = None,
               pad_min_px: Optional[float] = PAD_FLOOR_PX,
               pad_max_px: Optional[float] = None) -> Rect:
    """锚点图标右侧的数字 strip。

    x 单位 = 图标宽 iw（从 box.x2 起算）；y 单位 = 图标高 bh，
    再按**绝对像素**上下限夹一次（这一步才是跨分辨率的关键，见上面那段）。
    """
    iw = max(1e-6, box.w)
    bh = max(1e-6, box.h)
    pad = y_pad * bh
    if frame_h:
        if pad_min_px:
            pad = max(pad, pad_min_px / float(frame_h))
        if pad_max_px:
            pad = min(pad, pad_max_px / float(frame_h))
    return (min(1.0, box.x2 + x_from * iw),
            max(0.0, box.y1 - pad),
            min(1.0, box.x2 + x_to * iw),
            min(1.0, box.y2 + pad))


def digits(frame, rect: Rect) -> Optional[str]:
    """裁一块跑 OCR，只留数字/分隔符。返回原始串，解析交 parse_count。

    这是全 bot **唯一**的 OCR 入口。别在别处调 RapidOCR。
    """
    if frame is None:
        return None
    import time as _t
    import cv2
    _t0 = _t.time()
    STATS["ocr_calls"] += 1
    try:
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(rect[0] * w)), max(0, int(rect[1] * h))
        x2, y2 = min(w, int(rect[2] * w)), min(h, int(rect[3] * h))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 40:                                   # 小字放大，OCR 明显更准
            sc = 40.0 / ch
            crop = cv2.resize(crop, (int(cw * sc), 40), interpolation=cv2.INTER_CUBIC)
        result, _ = _engine()(crop)
        if not result:
            return None
        # 片段必须先按 x **从左到右**排序再拼：检测器返回顺序是任意的，
        #   live 2026-06-09 实锤 "179,958,141" 拼成了 '9581179414'。
        try:
            result = sorted(result, key=lambda ln: min(p[0] for p in ln[0]))
        except Exception:
            pass
        raw = "".join(line[1] for line in result)
        # 千位分隔符归一（2026-07-28 实锤）：青辉石 15,426 那条 OCR 返回**两个
        #   片段** '15.' 与 ',426' —— 两边各自把分隔符包了进去，拼成 '15.,426'。
        #   先折叠连续分隔符，再做三位分组匹配。
        #   把 '.' 也当分隔符是安全的：判据是**三位一组** —— 本域真正的小数只有
        #   百分比且都是一位小数（cafe 收益 '58.3'），永远匹配不上 \d{3}。
        raw_n = re.sub(r"[.,]{2,}", ",", raw.replace("，", ","))
        groups = re.findall(r"\d{1,3}(?:[.,]\d{3})+", raw_n)
        if groups:
            longest = max(groups, key=len)
            # 多个互不包含的分组 = 片段重叠错乱（'25,583,379'  '255833379'
            # 是 10 倍过读，商店预算会当场炸） fail-closed 返回 None 让投票重试。
            if [g for g in groups if g != longest and g not in longest]:
                return None
            return longest.replace(",", "").replace(".", "")
        kept = re.sub(r"[^0-9/.]", "", raw_n.replace(",", ""))
        return kept or None
    except Exception as e:
        print(f"[read] digit-OCR 异常: {e}", flush=True)
        return None
    finally:
        STATS["ocr_ms"] += (_t.time() - _t0) * 1000.0


def parse_count(s: Optional[str]) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """"7/8"(7,8)  "25117"(25117,None)  "240/240"(240,240)。不可解析  None。"""
    if not s:
        return None
    try:
        if "/" in s:
            a, _, b = s.partition("/")
            if not a.isdigit():
                return None
            return (int(a), int(b) if b.isdigit() else None)
        if s.isdigit():
            return (int(s), None)
    except Exception:
        pass
    return None


def read_beside(obs: Observation, anchor: Box,
                strip: Optional[Tuple[float, float, float]] = None
                ) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """读某个图标右边的数字。票数/价格/余额通用入口。"""
    spec = strip or STRIP.get(anchor.cls, STRIP_DEFAULT)
    x_from, x_to, y_pad = spec[0], spec[1], spec[2]
    pmin = spec[3] if len(spec) > 3 else PAD_FLOOR_PX
    pmax = spec[4] if len(spec) > 4 else None
    fh = obs.h or (obs.frame.shape[0] if obs.frame is not None else None)
    return parse_count(digits(obs.frame, icon_strip(
        anchor, x_from, x_to, y_pad, fh, pmin, pmax)))


def read_ticket(obs: Observation, anchor: Box, hard_max: int = 99) -> Optional[int]:
    """读票数。**带越界校验** —— 这是金钱链上最容易被读大骗过去的一格。

    `fail-closed 只挡 None，挡不住读大`（2026-07-27 实证）:
       「持有票券 **0/6**」被旧 strip 读成 `'9/0'`  返回 **9 票**，
       114 张零票屏里中招 **18 张**  `票数==0 就不出击` 这条源头金钱闸
       被读数直接架空。
     三道校验，任一不过就返回 None（宁可读不出，也不要读大）:
        分子 > 分母 = 串位（`9/0` 当场毙）
        分母 > hard_max = 不是票数（票券上限就那么点）
        分子 > hard_max = 同上
    """
    r = read_beside(obs, anchor, _TICKET_STRIPS.get(anchor.cls, TICKET_STRIP))
    if r is None or r[0] is None:
        return None
    cur, tot = r
    # 2026-08-12 补第四道（前三道有两道**依赖分母**，分母缺失时等于没校验）:
    #    `tot is None` 时  全被 `tot is not None` 短路跳过，只剩 `cur > 99`，
    #    于是「只读出分子」的帧**一路放行** —— 08-12 live 实锤：
    #    bounty/jfd 报「票数 6（屏上读出，唯一权威）」，而同一轮
    #    「步进器整行全灰 = 没票可扫」、扫荡 0~1 次，真值就是 **0**。
    #    收尾还据此算出「票 60(用了 6)」，**账目凭空多出 6 张**。
    #     票是 `N/M` 结构，**读不到分母就是没读全**，这一帧不采信。
    #    返回 None 不是"没票"：调用方 fail-closed 会等下一帧重读
    #      （[[ghost_click_and_ledger_overkill]]「屏上票数为唯一权威」），
    #      宁可多等几帧，也不要凭一个半截数去发动作或写台账。
    if tot is None:
        return None
    if cur > tot:
        return None
    if tot > hard_max:
        return None
    if cur > hard_max:
        return None
    return cur


QTY_MINUS, QTY_MINUS_GREY = "减号", "减号灰色"
QTY_PLUS, QTY_PLUS_GREY = "加号", "加号灰色"

_TILE = 5


def small_number(frame, rect: Rect, inset: float = 0.20) -> Optional[int]:
    """读**一两位数**（步进器数量、购买份数这种）。

    为什么不能直接用 `digits()`（2026-08-11 实测）:
       RapidOCR 的**检测**阶段读不出**孤立的单个字符** —— 一个 76x57、
       干干净净的白色 "0"，原图/反色/2x/Otsu/白边/rec-only 八种预处理
       **全部返回 None**。在 86 帧真步进器上 `digits()` 命中率只有 **8%**，
       而单字符正是最要命的那档（数量 0 vs 1）。
    解法：把它**横向平铺 5 份**，凑成检测器认得的"文本行"。
       实测 `内部平铺x5  '00000'`。
    这个做法**自带一致性校验**：5 份必须读出**同一个数**，
       不一致就说明 OCR 在瞎猜  返回 None（fail-CLOSED）。
       比"读一次然后信它"强得多 —— 相当于免费的 5 票共识。

    `inset` 是左右各裁掉的比例：步进器数字框两边紧挨着 −/+ 按钮的浅色边，
    不裁掉会把它们当笔画。
    """
    if frame is None:
        return None
    import cv2
    STATS["ocr_calls"] += 1
    try:
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(rect[0] * w)), max(0, int(rect[1] * h))
        x2, y2 = min(w, int(rect[2] * w)), min(h, int(rect[3] * h))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        dx, dy = int(cw * inset), int(ch * 0.10)
        crop = crop[dy:ch - dy, dx:cw - dx]
        if crop.size == 0 or crop.shape[1] < 4:
            return None
        result, _ = _engine()(cv2.hconcat([crop] * _TILE))
        if not result:
            return None
        raw = re.sub(r"\D", "", "".join(ln[1] for ln in result))
        if not raw or len(raw) % _TILE:
            return None
        k = len(raw) // _TILE
        parts = {raw[i * k:(i + 1) * k] for i in range(_TILE)}
        if len(parts) != 1:               # 5 份不一致 = OCR 在瞎猜
            return None
        return int(parts.pop())
    except Exception as e:
        print(f"[read] small_number 异常: {e}", flush=True)
        return None


def read_qty(obs: Observation, region: Optional[Rect] = None,
             hard_max: int = 999) -> Optional[int]:
    """读步进器里那个数字（MIN − **N** + MAX）。**消耗动作的唯一安全闸。**

    为什么非要读它不可（2026-08-11 特殊任務实帧）:
       AP 28、扫荡数量 **0**、MIN/−/+/MAX **全灰**，而「掃蕩開始」**是亮的**
       （conf 0.986，不是 `扫荡开始灰色`）。老代码只有「步进器整行全灰  没票」
       那一道闸，它在 AP 流上**分不开 0 和 1**：数量恰好等于上限 1 时，
       MIN/−/+/MAX **同样会全灰**。
        只看灰不看数 = 要么在数量 0 时点亮着的掃蕩鍵（弹「購買AP 💎30」，
         正是 30 青辉石那次近失的入口），要么把能扫的一次白白放掉。
       **游戏自己已经算好了「我买得起几次」并写在那个框里 —— 读它，别推理它。**

    通用性：商店购买 / 活动商店 / 批量掃蕩 用的是同一个步进器控件，
       所以这一个函数就够，不必每处再发明一遍。

    fail-CLOSED：读不出、越界、认不出步进器  一律 None，调用方**不许推进**。
    """
    minus = None
    for c in (QTY_MINUS, QTY_MINUS_GREY):
        b = obs.find(c, 0.40, region=region)
        if b is not None and (minus is None or b.conf > minus.conf):
            minus = b
    if minus is None:
        return None
    # 同一行、在减号右边的加号 —— 必须按行配对：顶栏的加号永远是亮的，
    #    全屏 find() 会把它当成步进器的右界，rect 直接横跨半个屏。
    row = (minus.cx, minus.cy - minus.h * 0.6, 1.0, minus.cy + minus.h * 0.6)
    plus = None
    for c in (QTY_PLUS, QTY_PLUS_GREY):
        for b in obs.all(c, 0.40, region=row):
            if b.cx > minus.cx and (plus is None or b.cx < plus.cx):
                plus = b                       # 取**最靠左**那个，别跨到别的控件
    if plus is None or plus.x1 <= minus.x2:
        return None
    rect: Rect = (minus.x2, minus.cy - minus.h * 0.42,
                  plus.x1, minus.cy + minus.h * 0.42)
    n = small_number(obs.frame, rect)
    if n is None:
        return None
    # 越界校验（同 read_ticket 的教训：fail-closed 只挡 None，挡不住读大）
    if n < 0 or n > hard_max:
        return None
    return n


EVENT_COIN_AREA = "货币数量显示区域"
_EVENT_COIN_REGION: Rect = (0.60, 0.0, 1.0, 0.25)


def read_event_coin(obs: Observation) -> Optional[int]:
    """活动商店的**当前币种余额**（右上角那个框）。

    2026-08-10 实测：`event_shop` 原本拿**左栏选中的币种 tab** 当锚
       （`read_beside(obs, cur)`）—— 那个 tab 右边是**币种名字，根本没有数字**
        余额恒为 None  `auto_buy` 的余额闸 fail-closed  **活动商店整天一件
       都没买**（而昨天同一条链是能清仓的，所以这缺口一直被当成"今天没货"）。
    真正的锚是 `货币数量显示区域`（conf 0.94-0.95，cx≈0.90 cy≈0.12），
       它是个**把图标和数字一起框住的区域**，所以不能用 read_beside（读右边），
       要**直接 OCR 整框**。
    别切左边留白：人眼真值 **5,265** 实测 —— 整框读出 `5265` ，
       左切 0.20 还对，**左切 ≥0.30 就变 `265`（首位没了）**。
       和青辉石/大赛币那两次一样，切多了就是读低，而"会读低就会读高"。
    """
    a = obs.find(EVENT_COIN_AREA, 0.30, region=_EVENT_COIN_REGION)
    if a is None:
        return None
    r = parse_count(digits(obs.frame, (a.x1, a.y1, a.x2, a.y2)))
    return r[0] if r and r[0] is not None else None


def read_topbar(obs: Observation, cls: str) -> Optional[int]:
    """顶栏某货币的**单帧**读数。单帧不可用于金钱决策，用 vote()。"""
    anchor = obs.find(cls, conf=0.25, region=TOPBAR)
    if anchor is None:
        return None
    r = read_beside(obs, anchor)
    return r[0] if r and r[0] is not None else None


class Vote:
    """多帧众数投票 —— 金钱决策的唯一合法读数方式。

    digit OCR 的不稳定是**双向**的（首位 DROP 6587587，右界过读 9999999），
    所以"取最少位"和"取最多位"都不对，只能投票取众数。
    ≥2 票才信；3 票强共识早退。**读不出返回 None，调用方必须 fail-closed。**
    """

    def __init__(self, need: int = 2, strong: int = 3, cap: int = 7):
        self.need, self.strong, self.cap = need, strong, cap
        self._reads: dict = {}

    def feed(self, obs: Observation, classes: Sequence[str]) -> None:
        for c in classes:
            if c in self._reads and self._reads[c].get("done"):
                continue
            v = read_topbar(obs, c)
            if v is None:
                continue
            slot = self._reads.setdefault(c, {"vals": [], "done": False})
            slot["vals"].append(v)
            if slot["vals"].count(v) >= self.strong:
                slot["done"] = True

    def result(self) -> dict:
        out = {}
        for c, slot in self._reads.items():
            vals: List[int] = slot["vals"]
            if not vals:
                out[c] = None
                continue
            top, n = Counter(vals).most_common(1)[0]
            out[c] = top if n >= self.need else None
        return out

    def settled(self, classes: Sequence[str]) -> bool:
        return all((self._reads.get(c) or {}).get("done") for c in classes)

    def reset(self) -> None:
        self._reads.clear()
