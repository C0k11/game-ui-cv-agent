# -*- coding: utf-8 -*-
"""金钱判据 —— **全 bot 唯一的一份**（pages / gate / interrupt 共用）。

花钱 = 点确认成交（买AP / 买票 / CAD）。
站在购买青辉石页、组合包页、特别贩售，不是花钱。
给购买键标框、日常第一枪进页领免费包，不是花钱。
程序不会点 CAD 购买键，也不会发 spend=青辉石。

08-17 用户定: 把组合包页写成真钱包停机，会让路由不敢进、标不敢做。
那条已删。purchase_context 只认成交框，不认货架页。

判据（加载中的帧不判）:

   弹窗体内出现 青辉石（cy > 0.12）
     顶栏余额必须排掉。买AP/买票确认框的价签走这条。

   购买青辉石 且 对话框控件同框（确认/取消/步进/货币数量区）
     大厅左侧广告位单独在场不算。隔半屏的确认框也不算。

   双键框内有数量步进器 = 购买/兑换框
     青辉石图标漏检时的结构闸。扫荡纯文字确认没有步进器，不误拦。

   组合包/组合包已选择 在场: 不再是购买语境。
     FreePackFlow 日常第一枪要进这一页。CAD 键由 flow 不点、闸拒未授权购买键。

   青辉石商店_已选择: 不 halt，只许退出/切回信用点 tab
"""
from __future__ import annotations

from typing import Optional

from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V

# 弹窗体 = 顶栏以下、左侧 tab 栏以右。
# cy<0.12 是顶栏余额；cx<0.12 是商店族页面的**左侧页签栏** ——
#    「青輝石」tab 的图标就挂在 (0.03, 0.30)，2026-08-08 信用点商店批量购买的
#    确认框上被它误报 HALT（对话框本体从来不会把价签排到 cx<0.12 去，
#    BA 的对话框都居中、价格行在中央区）。
BODY = (0.12, 0.12, 1.0, 1.0)
CONF = 0.40

# 对话框控件 —— 有这些才说明"这是个流程"，不是个广告位
_DIALOG_CHROME = [V.CONFIRM, V.CONFIRM_GREY, V.CANCEL, V.QTY_MAX, V.QTY_MAX_GREY,
                  V.QTY_MINUS, V.QTY_MINUS_GREY, V.CURRENCY_QTY_AREA]

# **收入语境**：屏幕正在「给」你东西时，弹窗体内的青辉石是**奖励**不是价签。
# 2026-08-08 live 实锤：活动首通的「獲得獎勵!」奖励卡里就有 **青輝石 x30**
#    （cy=0.511, conf 0.94） 打赢一场就被自己的金钱闸 HALT 掉。
#    这是 N2「信号≠场景」的**反向**形态：N2 是把广告位当购买框，这次是把
#    收款单当付款单。同一个青辉石图标，语境相反。
# **强信号**：只在"屏幕正在给你东西"那一屏出现，整屏可信。
_INCOME_STRONG = [V.GOT_REWARD, V.BATTLE_WIN, V.BATTLE_COMPLETE,
                  V.STORY_TAP_CONTINUE,
                  # 羁绊剧情面板：「羈絆劇情獎勵 青辉石x40 + 学生卡」是
                  #   **奖励预览**（08-09 帧证）。有这个键在场 = 在看奖励，
                  #   不是在付钱 —— 不排除的话每次进羁绊剧情都会误报 halt。
                  V.MOMO_ENTER_BOND, V.MOMO_GOTO_BOND]

# **弱信号**：这些是**页面上的常驻按钮**，不是"这一屏在给你东西"的证据。
#   2026-08-12 审计实锤：`领取奖励_黄` 是 arena 列表 / 每日任务页的常驻键
#   （pages.py 自己的注释就这么写的），而 arena 的「時間獎勵 +90/分」是持续
#   累积、永远领不完的 —— 也就是这个 cls 在战术大赛页上近乎每帧在场。
#   离线复现：拿仓库自己那张「購買AP 框」fixture，只额外加一个 `领取奖励_黄`
#   框，`purchase_context` 就从"购买框"变成 **None**，money_popup 打断跟着
#   不触发 —— 一个角落里的领取键关掉了三条判据。
#   所以它们只对**最松的那条**（"弹窗体内看见青辉石"）豁免，
#     对**结构性**判据（購買青輝石+对话框控件 / 双键框内有数量步进器）不豁免。
#     结构性判据本来就是为了治"青辉石检不出"的盲区，不能被一个词关掉。
_INCOME_WEAK = [V.CLAIM_REWARD_YELLOW, V.CLAIM_YELLOW, V.CLAIM_BLUE,
                V.CLAIM_ALL_YELLOW, V.CLAIM_ONCE_YELLOW]

_INCOME_MARKERS = _INCOME_STRONG + _INCOME_WEAK


def income_context(obs: Observation, strong_only: bool = False) -> Optional[str]:
    """屏幕正在**给**你东西（奖励页/结算页/领取页）。此时的青辉石是收入。

    `strong_only=True` 只认整屏可信的强信号，用在结构性判据上（见 `_INCOME_WEAK`）。
    """
    m = obs.find(_INCOME_STRONG if strong_only else _INCOME_MARKERS, 0.35)
    return f"{m.cls} conf {m.conf:.2f}" if m is not None else None


def signin_book_context(obs: Observation) -> Optional[str]:
    """阿罗娜签到簿/日历。板上的青辉石是 day 奖励图标，不是成交价签。

    全仓无「签到簿」专属 cls。签到相关只有 `每日领奖`(7, booklet 图标)。
    08-20 HALT 0003243: booklet 0；板上 BODY 同时有青辉石+信用点+体力。
    双键框 / 购买青辉石 不走这里，留给结构性判据。
    """
    if obs.find(V.SHOP_BUY_PYROXENE, CONF) is not None:
        return None
    if obs.find(V.CONFIRM, CONF) is not None and obs.find(V.CANCEL, CONF) is not None:
        return None
    pyx = obs.find(V.PYROXENE, CONF, region=BODY)
    if pyx is None:
        return None
    booklet = obs.find(V.NAV_DAILY_REWARD, CONF)
    if booklet is not None and obs.count(V.LOBBY_NAV, 0.35) < 3:
        return f"{V.NAV_DAILY_REWARD} conf {booklet.conf:.2f}"
    if (obs.find(V.CREDIT, CONF, region=BODY) is not None
            or obs.find(V.AP, CONF, region=BODY) is not None):
        return "签到簿日历奖励格"
    return None


def purchase_context(obs: Observation) -> Optional[str]:
    """非 None = 眼前是成交框（买AP/买票/付费确认），不是货架页。"""
    # 2026-08-12 live 误报急停：daily_mission 在「完成每日任務8次以上 8/8
    #    青辉石x20」那一帧被 halt —— 那是**奖励图标**（收入），不是价签。
    #    收入豁免本来是有的，但那一帧上 `全部領取` 黄键**被 Now Loading 盖住**
    #     `income_context` 一个标记都找不到  落到"体内有青辉石=购买框"。
    #    实测该帧只剩 3 个高 conf 框 + `加载中 0.87`，而那个青辉石 conf 只有
    #    **0.587**（半渲染的产物）。
    #     **加载帧上的检出不可靠，不拿它判金钱风险**。
    #    这不是开口子：真的购买确认框要用户交互才弹，不会出现在加载中；
    #      而且 bot 在「加载中」本来就由 interrupt 层按住不动作，
    #      不判 ≠ 放行去点。fail-CLOSED 的正确形态是**等页面稳定再判**。
    if obs.has(V.LOADING, 0.40):
        return None
    inc = income_context(obs)
    # 结构性判据只认强收入信号 —— 常驻领取按钮不许关掉它们
    inc_s = income_context(obs, strong_only=True)
    pyx = obs.find(V.PYROXENE, CONF, region=BODY)
    if pyx is not None:
        if inc is not None:
            return None          # 收入语境：奖励卡里的青辉石，不是价签
        if signin_book_context(obs) is not None:
            return None          # 签到簿日历格：进簿不是花钱
        return (f"弹窗体内出现青辉石 @({pyx.cx:.3f},{pyx.cy:.3f}) "
                f"conf {pyx.conf:.2f} = 购买框")
    buy = obs.find(V.SHOP_BUY_PYROXENE, CONF)
    if buy is not None and inc_s is None:
        # 屏上还看得到 ≥3 个大厅底栏入口 = 我们在**大厅**上，那个
        #    `購買青輝石` 就是左侧常驻广告位，不是购买框。
        #    2026-08-08 实测：大厅按返回键弹出「離開遊戲」确认框，它的
        #    `确认键` 配上这个广告位正好凑出下面这条判据  误报停整轮。
        #    （按返回那件事本身已经在 nav/base 里禁掉了，这里是第二道。）
        if obs.count(V.LOBBY_NAV, 0.35) >= 3:
            return None
        chrome = obs.find(_DIALOG_CHROME, CONF)
        if chrome is not None:
            # 对话框控件和「購買青輝石」得**属于同一个框**才算购买流程。
            #    2026-08-13 小号实测的误报: 制造锁着 -> 点入口弹一个单键框
            #    （確認 @cx=0.499），而大厅左侧广告位 `購買青輝石` @cx=0.116
            #    还在场 -> 两个八竿子打不着的东西凑成"购买流程" -> HALT 停整轮。
            #    上面那道 `LOBBY_NAV >= 3` 挡不住：框把底栏盖住了，数不到 3 个。
            #    真购买框里价签和確認键是**同一个框的两部分**，横向不会隔半屏；
            #      广告位在最左、框在正中，dx 稳定 >0.35。
            #    这是几何**相对关系**，不是写死坐标，换分辨率同样成立。
            if abs(buy.cx - chrome.cx) > 0.35:
                return None
            return (f"「購買青輝石」+ 对话框控件（{chrome.cls}）= 购买流程")
        # 大厅广告位 —— 不是购买框，别停
        return None

    #  **双键框内有数量步进器 = 购买/兑换框**（08-09 血泪，差点花 30 青辉石）
    #    实锤：AP 耗尽后 flow 又点了「掃蕩開始」，游戏弹「購買AP 是否購買AP120?
    #    單價30」，而**弹窗体内那两个青辉石图标一个都没检出**（判据 全瞎，
    #    memory 早记过这一格 v15 也只有 0.429 检出率） 闸放行了「確認」。
    #     不能只靠"看见青辉石"，要靠**结构**：游戏里带取消键的框如果还配了
    #      数量步进器，那就是在让你**买东西/换东西**。
    #    扫荡确认框（"要使用340AP掃蕩17次嗎"）是**纯文字**、无步进器，不会误拦；
    #      关卡面板的步进器不在双键框里（面板没有取消键），靠 band 排掉。
    #    误拦的代价只是停下来交人审；漏拦的代价是花青辉石 —— 不对称，宁可误拦。
    cf = obs.find(V.CONFIRM, CONF)
    cc = obs.find(V.CANCEL, CONF)
    if cf is not None and cc is not None and inc_s is None:
        band = (0.0, max(0.0, cf.cy - 0.35), 1.0, max(0.0, cf.cy - 0.02))
        step = obs.find([V.QTY_MAX, V.QTY_MAX_GREY, V.QTY_MIN, V.QTY_MIN_GREY,
                         V.PLUS, V.PLUS_GREY, V.QTY_MINUS, V.QTY_MINUS_GREY],
                        CONF, region=band)
        if step is not None:
            return (f"双键框内有数量步进器（{step.cls} @cy={step.cy:.2f}，"
                    f"確認 @cy={cf.cy:.2f}）= 购买/兑换框")
    return None


# 退出框只可能从**大厅**弹出来（按返回键）。
# **不含 facility**（08-09 审查实锤）：`facility` 是常态页 —— schedule 的
#    全體課程表面板、mining/club 掉 cls 时都落到它。把它当"可能弹退出框"的
#    底页，会让设施页里的「确认上课/确认制造」被当成退出框点取消  flow 再点
#    再取消，乒乓到 15 分钟超时（和 08-09「扫荡被取消 38 次」同形）。
#    pages.py 的 facility 签名本身就是 need=[回大厅,返回键] —— 那恰恰是
#    **游戏内**的证据，不是退出框的证据。
LOBBY_LIKE = ("lobby",)


def is_combo_pack_page(obs: Observation) -> bool:
    """组合包已选中，或屏上有免费标。特别贩售上只有未选中页签，不算。

    08-17: 这一页不再是 purchase_context。本函数只给 interrupt 决定
    要不要把 money_popup 交回 flow（选中了就交回，没选中就切 tab）。
    """
    return obs.find([V.COMBO_PACK_SEL, V.FREE], CONF) is not None


def system_dialog(obs: Observation, last_solid: Optional[str] = None,
                  strict: bool = False) -> Optional[str]:
    """系统级对话框（不是游戏内容）—— **確認 = 退出游戏**。

    判据：有 確認+取消，但**顶栏一个货币都没有**。游戏内的对话框永远挂在
    某个页面上，顶栏在；系统框把整个 UI 都盖住了。

    为什么这条判据必须放在**闸**里而不是 flow 基类里（2026-08-08 血泪）:
       我先把它写进 `Flow.on_confirm_dialog` 的默认实现 —— 但 craft/shop/
       event_shop/cafe/schedule/mining/arena/sweep **八个 flow 都覆写了那个方法**，
       于是护栏被整体绕过，craft 照样在退出游戏框上点了確認。
       **闸是每一发 tap 的必经之路，护栏放这里没人绕得过。**
       （这已经是同一个"修一处没 grep 全仓同形"第三次咬人了。）
    """
    # **最强判据：底页**（08-09 实锤，前两版都被绕过）。
    #    退出框只可能从**大厅**按返回弹出来；「要使用340AP掃蕩17次嗎」的底页
    #    是 stage_popup。顶栏货币/返回键这些"屏上证据"在弹窗压暗时全测不到
    #    （实测那帧只剩 3 个检出），只有底页身份是可靠的。
    # `unknown` **不是"不在大厅"，是"不知道"**（08-09 审查实锤）：
    #    Machine 新建时 last_solid 恒为 'unknown'，只在页面切换时才写；
    #    step 每次新进程、runner 每条 flow 都从这个窗口起步  旧写法
    #    `not in LOBBY_LIKE  return None` 在这里**放行了点確認 = 关游戏**。
    #     strict 模式（闸调用）下把"不知道"当成**可疑**，只有明确不在大厅
    #      才放行；非 strict（flow 基类那道）保持宽松，免得流程卡死。
    unknown_src = last_solid in (None, "", "unknown", "blank")
    if unknown_src:
        if not strict:
            return None
    elif last_solid not in LOBBY_LIKE:
        return None
    if not (obs.has(V.CONFIRM, CONF) and obs.has(V.CANCEL, CONF)):
        return None
    if obs.has([V.AP, V.CREDIT, V.PYROXENE], 0.35, region=(0.0, 0.0, 1.0, 0.12)):
        return None                      # 顶栏在 = 游戏内对话框，正常
    # 顶栏货币是弱证据：确认框把屏压暗时顶栏常检不出（day2 悬赏扫荡确认
    #    被误判成退出框）。「有叉叉=游戏框」也错 —— 真实退出框帧上就检出过
    #    0.96 的叉叉（离线 fixture 为证，测试当场枪毙了我这版）。
    #    结构证据 = **退出框只会从大厅弹出**：屏上有 返回键/回大厅按钮
    #    = 正在某个设施页里 = 游戏内对话框，放行。
    if obs.has([V.BACK, V.HOME], 0.35):
        return None
    # 同一族的第二条结构证据（2026-08-11 剧情挖矿 live 抓到）：
    #    **剧情播放页三个都没有** —— 没顶栏、没返回键、没回大厅 ——
    #    于是「是否略過此劇情？」这个纯游戏内对话框被判成疑似退出框，
    #    剧情挖矿链直接卡死在那儿（step 模式下动作链断掉时必现）。
    #    而它有一个大厅**不可能有**的东西：右上角的 `剧情menu`。
    #    退出游戏框只从大厅弹，大厅没有剧情menu  这条判据和上面那条一样硬。
    if obs.has(V.STORY_MENU, 0.40, region=(0.80, 0.0, 1.0, 0.18)):
        return None
    return "系统对话框（顶栏货币全无、也不在任何设施页上）—— 確認可能是**退出游戏**"

