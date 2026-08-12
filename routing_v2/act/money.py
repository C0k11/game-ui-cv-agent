# -*- coding: utf-8 -*-
"""⛔金钱判据 —— **全 bot 唯一的一份**（pages / gate / interrupt 共用）。

铁律: 青辉石**只进不出**。bot 绝不花青辉石、绝不碰真钱。

判据设计的两难，以及为什么长这样:

  【假阴性的代价】漏判 = 真的花掉青辉石，不可逆。
  【假阳性的代价】误判 = 每次开跑第一帧就 HALT，整条链跑不起来。
                 ——2026-08-07 已经吃过一次（顶栏 OCR 抖动误报 MONEY BREACH
                 急停整条 pipeline，而余额一分没少）。

  ⇒ 所以判据必须**结构化**，不能靠"屏上出现某个词就停"。
  ⛔ 2026-08-08 live 实测打脸的那一版: 我写的是"`购买青辉石` 在屏上 → 停"。
     结果**大厅左侧常驻一个买石广告位**，`购买青辉石` conf **0.95** 稳定在场
     ⇒ 每一轮的第一帧就 HALT。这是"把一个词当成一个场景"的典型错误。
     （和 §A9「把内容多当页面前提」是同一族病：**信号 ≠ 场景**。）

现在的判据 = 三条，任一成立就是"真的站在花钱的位置上":

  ① 弹窗体内出现 `青辉石`（cy > 0.12）
     —— 顶栏 cy<0.12 那个是余额显示，正常帧每帧都有，必须靠 region 排掉。
        这条是主判据，live 验过：购买框里商品价签就是弹窗体内的青辉石图标。

  ② `组合包` / `组合包已选择` 在场
     —— 组合包页，免費包旁边就是 CAD$16.99 / $3.99 的**真钱包**。

  ③ `购买青辉石` **且**有对话框控件（确认/取消/数量步进/货币数量区）
     —— 单独的 `购买青辉石` 只是大厅广告位；配上对话框控件才是真的购买流程。

  ④ `青辉石商店_已选择` 在场 = 站在花青辉石的 tab 上（不 halt，但只许退出）
"""
from __future__ import annotations

from typing import Optional

from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V

# 弹窗体 = 顶栏以下、左侧 tab 栏以右。
# ⛔cy<0.12 是顶栏余额；⛔cx<0.12 是商店族页面的**左侧页签栏** ——
#    「青輝石」tab 的图标就挂在 (0.03, 0.30)，2026-08-08 信用点商店批量购买的
#    确认框上被它误报 HALT（对话框本体从来不会把价签排到 cx<0.12 去，
#    BA 的对话框都居中、价格行在中央区）。
BODY = (0.12, 0.12, 1.0, 1.0)
CONF = 0.40

# 对话框控件 —— 有这些才说明"这是个流程"，不是个广告位
_DIALOG_CHROME = [V.CONFIRM, V.CONFIRM_GREY, V.CANCEL, V.QTY_MAX, V.QTY_MAX_GREY,
                  V.QTY_MINUS, V.QTY_MINUS_GREY, V.CURRENCY_QTY_AREA]

# ⭐**收入语境**：屏幕正在「给」你东西时，弹窗体内的青辉石是**奖励**不是价签。
# ⛔2026-08-08 live 实锤：活动首通的「獲得獎勵!」奖励卡里就有 **青輝石 x30**
#    （cy=0.511, conf 0.94）⇒ 打赢一场就被自己的金钱闸 HALT 掉。
#    这是 N2「信号≠场景」的**反向**形态：N2 是把广告位当购买框，这次是把
#    收款单当付款单。同一个青辉石图标，语境相反。
_INCOME_MARKERS = [V.GOT_REWARD, V.BATTLE_WIN, V.BATTLE_COMPLETE,
                   V.CLAIM_REWARD_YELLOW, V.CLAIM_YELLOW, V.CLAIM_BLUE,
                   V.CLAIM_ALL_YELLOW, V.CLAIM_ONCE_YELLOW, V.STORY_TAP_CONTINUE,
                   # ⭐羁绊剧情面板：「羈絆劇情獎勵 青辉石x40 + 学生卡」是
                   #   **奖励预览**（08-09 帧证）。有这个键在场 = 在看奖励，
                   #   不是在付钱 —— 不排除的话每次进羁绊剧情都会误报 halt。
                   V.MOMO_ENTER_BOND, V.MOMO_GOTO_BOND]


def income_context(obs: Observation) -> Optional[str]:
    """屏幕正在**给**你东西（奖励页/结算页/领取页）。此时的青辉石是收入。"""
    m = obs.find(_INCOME_MARKERS, 0.35)
    return f"{m.cls} conf {m.conf:.2f}" if m is not None else None


def purchase_context(obs: Observation) -> Optional[str]:
    """返回非 None = **真的**站在会花青辉石/真钱的位置上。文案即证据。"""
    # ⛔⛔2026-08-12 live 误报急停：daily_mission 在「完成每日任務8次以上 8/8
    #    青辉石x20」那一帧被 halt —— 那是**奖励图标**（收入），不是价签。
    #    收入豁免本来是有的，但那一帧上 `全部領取` 黄键**被 Now Loading 盖住**
    #    ⇒ `income_context` 一个标记都找不到 ⇒ 落到"体内有青辉石=购买框"。
    #    实测该帧只剩 3 个高 conf 框 + `加载中 0.87`，而那个青辉石 conf 只有
    #    **0.587**（半渲染的产物）。
    #    ⇒ **加载帧上的检出不可靠，不拿它判金钱风险**。
    #    ⚠这不是开口子：真的购买确认框要用户交互才弹，不会出现在加载中；
    #      而且 bot 在「加载中」本来就由 interrupt 层按住不动作，
    #      不判 ≠ 放行去点。fail-CLOSED 的正确形态是**等页面稳定再判**。
    if obs.has(V.LOADING, 0.40):
        return None
    inc = income_context(obs)
    pyx = obs.find(V.PYROXENE, CONF, region=BODY)
    if pyx is not None:
        if inc is not None:
            return None          # 收入语境：奖励卡里的青辉石，不是价签
        return (f"弹窗体内出现青辉石 @({pyx.cx:.3f},{pyx.cy:.3f}) "
                f"conf {pyx.conf:.2f} = 购买框")
    combo = obs.find([V.COMBO_PACK, V.COMBO_PACK_SEL], CONF)
    if combo is not None:
        return f"组合包页（{combo.cls} conf {combo.conf:.2f}）— 旁边就是真钱包"
    buy = obs.find(V.SHOP_BUY_PYROXENE, CONF)
    if buy is not None and inc is None:
        # ⛔⛔屏上还看得到 ≥3 个大厅底栏入口 = 我们在**大厅**上，那个
        #    `購買青輝石` 就是左侧常驻广告位，不是购买框。
        #    2026-08-08 实测：大厅按返回键弹出「離開遊戲」确认框，它的
        #    `确认键` 配上这个广告位正好凑出下面这条判据 ⇒ 误报停整轮。
        #    （按返回那件事本身已经在 nav/base 里禁掉了，这里是第二道。）
        if obs.count(V.LOBBY_NAV, 0.35) >= 3:
            return None
        chrome = obs.find(_DIALOG_CHROME, CONF)
        if chrome is not None:
            return (f"「購買青輝石」+ 对话框控件（{chrome.cls}）= 购买流程")
        # 大厅广告位 —— 不是购买框，别停
        return None

    # ④ ⛔⛔**双键框内有数量步进器 = 购买/兑换框**（08-09 血泪，差点花 30 青辉石）
    #    实锤：AP 耗尽后 flow 又点了「掃蕩開始」，游戏弹「購買AP 是否購買AP120?
    #    單價💎30」，而**弹窗体内那两个青辉石图标一个都没检出**（判据① 全瞎，
    #    memory 早记过这一格 v15 也只有 0.429 检出率）⇒ 闸放行了「確認」。
    #    ⇒ 不能只靠"看见青辉石"，要靠**结构**：游戏里带取消键的框如果还配了
    #      数量步进器，那就是在让你**买东西/换东西**。
    #    ⚠扫荡确认框（"要使用340AP掃蕩17次嗎"）是**纯文字**、无步进器，不会误拦；
    #      关卡面板的步进器不在双键框里（面板没有取消键），靠 band 排掉。
    #    ⚠误拦的代价只是停下来交人审；漏拦的代价是花青辉石 —— 不对称，宁可误拦。
    cf = obs.find(V.CONFIRM, CONF)
    cc = obs.find(V.CANCEL, CONF)
    if cf is not None and cc is not None and inc is None:
        band = (0.0, max(0.0, cf.cy - 0.35), 1.0, max(0.0, cf.cy - 0.02))
        step = obs.find([V.QTY_MAX, V.QTY_MAX_GREY, V.QTY_MIN, V.QTY_MIN_GREY,
                         V.PLUS, V.PLUS_GREY, V.QTY_MINUS, V.QTY_MINUS_GREY],
                        CONF, region=band)
        if step is not None:
            return (f"双键框内有数量步进器（{step.cls} @cy={step.cy:.2f}，"
                    f"確認 @cy={cf.cy:.2f}）= 购买/兑换框")
    return None


# 退出框只可能从**大厅**弹出来（按返回键）。
# ⛔⛔**不含 facility**（08-09 审查实锤）：`facility` 是常态页 —— schedule 的
#    全體課程表面板、mining/club 掉 cls 时都落到它。把它当"可能弹退出框"的
#    底页，会让设施页里的「确认上课/确认制造」被当成退出框点取消 ⇒ flow 再点
#    再取消，乒乓到 15 分钟超时（和 08-09「扫荡被取消 38 次」同形）。
#    pages.py 的 facility 签名本身就是 need=[回大厅,返回键] —— 那恰恰是
#    **游戏内**的证据，不是退出框的证据。
LOBBY_LIKE = ("lobby",)


def is_combo_pack_page(obs: Observation) -> bool:
    """当前是不是**組合包页**（免費組合包所在的那一页）。

    ⭐它和别的购买语境有本质区别：**bot 是主动进来拿免費包的**，而不是
       "意外撞上购买框"。⇒ 这一页不该无条件 halt 整轮（08-09 审查实锤：
       ShopFlow 点大厅广告位进来，下一帧就被 money_popup 停机，
       `on_combo_pack` 那整段代码永远执行不到）。
    ⚠放行的只是"别停机"，真正的钱仍被 Gate 拦：
       `_NAV_SAFE` 白名单外的点击一律人审，`money=True` 还要再过一道。
    """
    return obs.find([V.COMBO_PACK, V.COMBO_PACK_SEL], CONF) is not None


def system_dialog(obs: Observation, last_solid: Optional[str] = None,
                  strict: bool = False) -> Optional[str]:
    """系统级对话框（不是游戏内容）—— **確認 = 退出游戏**。

    判据：有 確認+取消，但**顶栏一个货币都没有**。游戏内的对话框永远挂在
    某个页面上，顶栏在；系统框把整个 UI 都盖住了。

    ⛔⛔为什么这条判据必须放在**闸**里而不是 flow 基类里（2026-08-08 血泪）:
       我先把它写进 `Flow.on_confirm_dialog` 的默认实现 —— 但 craft/shop/
       event_shop/cafe/schedule/mining/arena/sweep **八个 flow 都覆写了那个方法**，
       于是护栏被整体绕过，craft 照样在退出游戏框上点了確認。
       **闸是每一发 tap 的必经之路，护栏放这里没人绕得过。**
       （这已经是同一个"修一处没 grep 全仓同形"第三次咬人了。）
    """
    # ⛔⛔**最强判据：底页**（08-09 实锤，前两版都被绕过）。
    #    退出框只可能从**大厅**按返回弹出来；「要使用340AP掃蕩17次嗎」的底页
    #    是 stage_popup。顶栏货币/返回键这些"屏上证据"在弹窗压暗时全测不到
    #    （实测那帧只剩 3 个检出），只有底页身份是可靠的。
    # ⛔⛔`unknown` **不是"不在大厅"，是"不知道"**（08-09 审查实锤）：
    #    Machine 新建时 last_solid 恒为 'unknown'，只在页面切换时才写；
    #    step 每次新进程、runner 每条 flow 都从这个窗口起步 ⇒ 旧写法
    #    `not in LOBBY_LIKE → return None` 在这里**放行了点確認 = 关游戏**。
    #    ⇒ strict 模式（闸调用）下把"不知道"当成**可疑**，只有明确不在大厅
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
    # ⛔顶栏货币是弱证据：确认框把屏压暗时顶栏常检不出（day2 悬赏扫荡确认
    #    被误判成退出框）。⚠「有叉叉=游戏框」也错 —— 真实退出框帧上就检出过
    #    0.96 的叉叉（离线 fixture 为证，测试当场枪毙了我这版）。
    #    结构证据 = **退出框只会从大厅弹出**：屏上有 返回键/回大厅按钮
    #    = 正在某个设施页里 = 游戏内对话框，放行。
    if obs.has([V.BACK, V.HOME], 0.35):
        return None
    # ⭐同一族的第二条结构证据（2026-08-11 剧情挖矿 live 抓到）：
    #    **剧情播放页三个都没有** —— 没顶栏、没返回键、没回大厅 ——
    #    于是「是否略過此劇情？」这个纯游戏内对话框被判成疑似退出框，
    #    剧情挖矿链直接卡死在那儿（step 模式下动作链断掉时必现）。
    #    而它有一个大厅**不可能有**的东西：右上角的 `剧情menu`。
    #    退出游戏框只从大厅弹，大厅没有剧情menu ⇒ 这条判据和上面那条一样硬。
    if obs.has(V.STORY_MENU, 0.40, region=(0.80, 0.0, 1.0, 0.18)):
        return None
    return "系统对话框（顶栏货币全无、也不在任何设施页上）—— 確認可能是**退出游戏**"


def on_pyroxene_tab(obs: Observation) -> bool:
    """站在青辉石商店 tab 上。不 halt（还能退出去），但只许发退出类动作。"""
    return obs.has(V.SHOP_TAB_PYROXENE_SEL, 0.45)
