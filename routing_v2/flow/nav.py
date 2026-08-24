# -*- coding: utf-8 -*-
"""导航 —— 「从这里到那里」的**唯一**实现。

为什么导航必须集中（§A1 + §A2）:
   `回大厅按钮` 和 `返回键` 几乎每一页都在。老代码里每个 skill 各写一句
   "看到就点返回"，于是这两个按钮事实上变成了**全局兜底动作** —— 任何一帧
   主 cls 没检出（进战斗的转场帧最典型），bot 就把自己弹回大厅。
   用户现场目击："刚要进战斗，bot 又手贱去点返回大厅的房子按钮"。

   这里的规则是硬的：
     · 退出动作**只在已确认的、具名的页面上**发起（`st.page != "unknown"`）
     · 且必须**刚确认完**不久之外（转场帧不会连续 confirm_frames 帧一致）
     · UNKNOWN 页面永远返回 None
"""
from __future__ import annotations

from typing import Optional

from routing_v2.act.action import Action, swipe, tap_at, tap_box, wait
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView

# 页面层级 —— 「谁是谁的上一层」。
#
# 为什么必须显式建这张表（2026-08-12 用户点名:「返回和大厅按钮要结合位置语义
#   来判断是返回还是回大厅，要根据板块来的，比方说我们打完学园交流会，
#   这个时候就是返回任务大厅就行」）:
#   在这张表之前，退出逻辑只知道"屏上有哪个键"，不知道"我要去哪一层"。
#   于是 `prefer_home` 那个开关全仓没有一个调用点、`Flow.home_pages` 声明了
#   零引用 —— 也就是说「这条 flow 该退到哪儿」这个概念根本不存在，
#   `回大厅按钮` 事实上从来没被点过（另一个调用点挂在默认 False 的
#   `allow_home_escape` 后面）。留下的是"一路按返回退到底"，
#   而正确答案取决于**下一条 flow 的入口在哪一层**。
#
# 有了层级之后，「返回 还是 回大厅」不再是看屏幕猜，而是一道算术:
#   目标在祖先链上 -> 往上退；隔一层用返回键，隔两层以上且目标是大厅就用回大厅键。
#   目标不在祖先链上（在 cafe 里要去任务大厅）-> 先回大厅，再从大厅进。
_PARENT = {
    "task_hall": "lobby",
    "campaign_stage": "task_hall",
    "grid_quest": "campaign_stage",
    "bounty_branch": "task_hall",
    "bounty_stage": "bounty_branch",
    "jfd_academy": "task_hall",
    "jfd_stage": "jfd_academy",
    "arena": "task_hall",
    "event_page": "task_hall",
    "event_quest_list": "task_hall",
    "event_guide_hub": "task_hall",
    "event_ended": "task_hall",
    "event_shop": "event_page",
    "cafe": "lobby",
    "cafe_invite_list": "cafe",
    "schedule_area": "lobby",
    "schedule_region": "schedule_area",
    "craft": "lobby",
    "shop": "lobby",
    "arena_shop": "shop",
    "shop_pyroxene_tab": "shop",
    "combo_pack": "shop",
    "mail": "lobby",
    "daily_mission": "lobby",
    "club": "lobby",
    "momo_list": "lobby",
    "momo_chat": "momo_list",
    "story_hub": "lobby",
    "story_chapter_map": "story_hub",
    "story_nodes": "story_chapter_map",
}

# 从大厅进某一层的入口 cls。只登记**大厅直达**的那些；再深的一层靠逐层进。
_ENTRY = {"task_hall": V.NAV_TASKS}

# 任务大厅的正向身份只认专属磁贴，且至少两个同时在场。
# 目标玩法磁贴不参与身份判断，因为未解锁账号上它可能完全不存在。
_TASK_HALL_ANCHORS = (V.HUB_STORY, V.HUB_CAMPAIGN, V.HUB_SPECIAL)


def task_hall_anchor_count(obs: Observation, conf: float = 0.45) -> int:
    """当前帧检出的任务大厅专属磁贴数。"""
    return sum(obs.find(cls, conf) is not None for cls in _TASK_HALL_ANCHORS)


def task_hall_evidence(obs: Observation, conf: float = 0.45) -> bool:
    """任务大厅在场。国际服新皮只剩推图能过 0.45, 不再要两个专属磁贴。"""
    if obs.find(V.HUB_CAMPAIGN, conf) is None:
        return False
    if obs.find(V.BACK, conf) is None and obs.find(V.HOME, conf) is None:
        return False
    if obs.count(V.LOBBY_NAV, conf) >= 3:
        return False
    return True


def hub_tile_dot(obs: Observation, region, conf: float = 0.40):
    """tile 真名 cls=0 时, 退回点该卡上的红/黄点。

    08-20 live (`v2_20260820_155347/_ann/0000322_new_task_hall_ann.jpg`):
      国际服任务大厅九张卡换皮, 只有 `任务关卡推图` 过 0.45; 懸賞通緝 卡上
      黄点 0.88、戰術大賽 卡上红点 0.91 仍在。story_mining 早就用这条退路
      (`mining.py` 剧情 tile 0 点卡上黄点), 悬赏/大赛没有, 于是 80 帧后
      SKIPPED tile_dead。

    落点是**检出框**, 不是新硬坐标 —— 认不出卡但认得出卡上的点。
    region = (x1, y1, x2, y2) 归一化, 圈住该卡, 防止点到别的卡。
    學園交流會 那张卡同帧无红黄点, 所以这条退路对它不成立(不要硬套)。
    """
    x1, y1, x2, y2 = region
    hits = [b for b in (obs.all(V.DOT_RED, conf) + obs.all(V.DOT_YELLOW, conf))
            if x1 <= b.cx <= x2 and y1 <= b.cy <= y2]
    if not hits:
        return None
    return max(hits, key=lambda b: b.conf)


# 同窗黄「任務開始」会被预标成扫荡开始(同页签预标不一致, 不是页签反转)。
# cy>0.72 的扫荡开始当任务开始, 禁点。多个扫荡开始只取 cy 最小的。
_SWEEP_START_MAX_CY = 0.72


def real_sweep_start(obs: Observation, conf: float = 0.35):
    hits = [b for b in obs.all(V.SWEEP_START, conf) if b.cy <= _SWEEP_START_MAX_CY]
    if not hits:
        return None
    return min(hits, key=lambda b: b.cy)


def wake_hidden_lobby(obs: Observation, st: StateView, flow) -> Optional[Action]:
    """大厅闲置约 6 秒会收起全部 UI。只在刚从大厅掉到 blank/unknown 时唤醒。"""
    if st.last_solid != "lobby":
        return None
    if st.page not in ("unknown", "blank"):
        return None
    if st.overlay:
        return None
    navn = obs.count(V.LOBBY_NAV, 0.30)
    if len(obs.boxes) > 0 and navn >= 2:
        return None
    if not flow.pending("lobby_wake"):
        return None
    return tap_at(
        0.40, 0.55, "大厅藏UI 唤醒",
        justify="国际服大厅闲置约6秒收起全部UI, 成 blank/unknown。"
                "只在 last_solid=lobby 且(零框或底栏NAV<2)时点背景(0.40,0.55),"
                "不点正中以免压到立绘。战斗/剧情 last_solid 不是 lobby, 不进这条。",
        once="lobby_wake")


# 换皮期专用: 只对 `任务大厅入口` 这一个 cls 做二次低阈重推。
#   **不动全局 0.20**(那会让每帧涌进一堆垃圾框), 也**不写坐标**。
# 08-21 全量实测(飞轮 v19 67 帧, 推理底压到 0.01):
#   - 12 张已确认大厅的帧上, 这个 cls **恰好各只有 1 个候选**,
#     conf 0.0464-0.1402, 位置 (0.944, 0.9416), 帧间抖动 <0.001 —— 定位是准的,
#     崩的只是类分(新皮)。
#   - 55 张非大厅帧里只有 1 张有候选(组合包浮层压暗那张, conf 0.718),
#     而那本来就是同一个真入口。**零误报**。
#   所以低阈通道的安全边界是: 已确认在大厅 + 低阈候选唯一。
# v19 把新皮训回去之后, 把 _NAV_TASKS_FLOOR 调回 0.45 即可, 结构不用改。
_NAV_TASKS_FLOOR = 0.03
_task_entry_cache = {"seq": None, "box": None}


def weak_task_entry(obs: Observation):
    """大厅里那个任务大厅入口的**检出框**(换皮期低阈通道)。拿不准返回 None。

    落点仍然来自 cls 检出框, 不是坐标 —— JIT 复验、契约锚点都照常成立。
    """
    if obs.count(V.LOBBY_NAV, 0.35) < 3:
        return None                       # 没确认在大厅, 不开低阈通道
    if obs.frame is None:
        return None
    seq = getattr(obs, "seq", None)
    if seq is not None and _task_entry_cache["seq"] == seq:
        return _task_entry_cache["box"]
    from routing_v2.percept import detect
    cands = [b for b in detect.infer(obs.frame, ("ui",),
                                     conf_override=_NAV_TASKS_FLOOR)
             if b.cls == V.NAV_TASKS]
    box = cands[0] if len(cands) == 1 else None   # 多于一个 = 噪声, 宁可不动
    if seq is not None:
        _task_entry_cache["seq"] = seq
        _task_entry_cache["box"] = box
    return box


def hall_door_from_lobby(obs: Observation, *, reason: str,
                         expect=(), post=None, once: str = ""):
    """大厅进任务厅。入口 cls 过 0.45 就点框; 否则走**同一个 cls** 的低阈通道。

    08-21 用户定: 不要任何硬编码, 落点一律来自 cls。原来那发写死右下角坐标的
    盲点是在**猜控件位置**, 已删 —— 版面再改一次就会点错,
    而低阈通道会**如实检不出**并让上层去处理, 这才是 fail-closed 的形态。
    """
    if isinstance(expect, str):
        expect = (expect,)
    box = obs.find(V.NAV_TASKS, 0.45)
    why = reason
    if box is None:
        box = weak_task_entry(obs)
        if box is not None:
            why = (f"{reason}(新皮低阈通道: `{V.NAV_TASKS}` conf {box.conf:.3f}"
                   f" 唯一候选, 落点仍是检出框)")
    if box is None:
        return None
    return tap_box(box, why, expect=expect, post=post, once=once)


def hall_door_followup(obs: Observation, flow) -> Optional[Action]:
    """点过任务夹之后: 见推图就等确认; FruLink(无底栏无推图) BACK 一次。"""
    if flow.pending("lobby_enter"):
        return None
    if obs.find(V.HUB_CAMPAIGN, 0.45) is not None:
        return wait("已见推图，等任务大厅确认")
    if obs.count(V.LOBBY_NAV, 0.30) >= 3:
        return None
    n = flow.bump("hall_door_wait")
    if n > 8:
        return None
    if (obs.count(V.LOBBY_NAV, 0.30) < 2
            and obs.find(V.HUB_CAMPAIGN, 0.30) is None
            and flow.pending("frulink_back")):
        act = back_key(obs, "FruLink 挡门，系统返回一次")
        if act is not None:
            act.once_key = "frulink_back"
        return act
    return wait("点任务夹后等大厅或任务厅")


# 模态/弹窗类**底页** —— 它们没有"层级"，只能关掉，关掉后回到它盖着的那一页。
#   归位时必须先把它们关干净，否则下一条 flow 会在**别人的弹窗**上动手
#   （2026-08-12 体外复现: bounty 从 `on_stage_popup` 收工时弹窗还开着，
#    jfd 接手第一帧就在悬赏的关卡上点了「扫荡开始」；event 留下的编队页
#    被 arena 接手，第一帧就去勾「跳過戰鬥」准备出击）。
#   只放 `overlay=False` 的。覆盖层走 `close_overlay` —— 它们根本不会出现在
#     `st.page` 上，我第一版把它们混进这张表，等于写了一堆永不成立的分支，
#     而真正的覆盖层照样被留给下一条 flow。
_MODAL = {"stage_popup", "sweep_dialog", "formation", "squad_quick_edit"}


def close_overlay(obs: Observation, overlay: str) -> Optional[Action]:
    """关掉盖在底页上的覆盖层（确认框/奖励框/领取面板/结算动画）。

    顺序沿用 `ExitMixin.exit_step` 那条流血换来的:
      取消 > (结算动画的 SKIP) > 结果框确认 > 叉叉 > 返回键
      取消最优先：**结果弹窗永远没有取消键**（全语料 44,669 帧实证），
        所以有取消键 = 这是个决策框，退它只能点取消。
      叉叉排在确认后面：模态框上 conf 0.96 的叉叉照样可能完全不吃点击
        （悬赏弃 6 票的第三个根因，连点 6 次画面纹丝不动）。
    `确认键` 的安全性由金钱闸兜底 —— 真是购买框的话 `act/money.py` 先拦成 halt。
    """
    c = obs.find(V.CANCEL, 0.45)
    if c is not None:
        return tap_box(c, "归位: 取消（决策框唯一有效出口）")
    if overlay == "sweep_results":
        sk = obs.find(V.BATTLE_SKIP, 0.45)
        if sk is not None:
            return tap_box(sk, "归位: SKIP 掉结算奖励动画")
    b = (obs.find(V.CONFIRM, 0.45)
         or obs.find(V.STORY_TAP_CONTINUE, 0.40))
    if b is not None:
        return tap_box(b, f"归位: 关掉 {overlay}（结果框没有取消键）")
    x = obs.find(V.CLOSE_X, 0.55)
    if x is not None:
        return tap_box(x, f"归位: 叉掉 {overlay}")
    return None


def depth(page: str) -> int:
    """离大厅几层。大厅=0，未登记的页面返回 -1。"""
    if page == "lobby":
        return 0
    n, cur = 0, page
    while cur in _PARENT:
        cur = _PARENT[cur]
        n += 1
        if cur == "lobby":
            return n
    return -1


def ancestors(page: str) -> list:
    """从近到远的祖先链，末尾是 lobby。未登记页面返回空。"""
    out, cur = [], page
    while cur in _PARENT:
        cur = _PARENT[cur]
        out.append(cur)
    return out if out and out[-1] == "lobby" else []


# 这些页面上，"退出"是安全且有意义的
_EXITABLE = {
    "cafe", "cafe_invite_list", "schedule_region", "schedule_area", "craft",
    "shop", "arena_shop", "mail", "daily_mission", "club", "momo_list",
    "momo_chat", "story_hub", "story_nodes", "task_hall", "arena",
    "bounty_branch", "bounty_stage", "jfd_academy", "jfd_stage",
    "campaign_stage", "event_page", "event_quest_list", "event_ended", "event_shop",
    "facility",          # 没登记的设施页 —— 有退出控件，能安全走人
    "combo_pack", "shop_pyroxene_tab", "confirm_dialog", "stage_popup",
    "sweep_dialog", "formation",
}


def to_lobby(obs: Observation, st: StateView) -> Optional[Action]:
    """逐层退出当前页面 —— **只在已确认的具名页面上动手，且只用返回键。**

    这个函数**不做「该返回还是该回大厅」的决定** —— 那个决定在 `route()` 里，
       由目标层（`Flow.entry_page`）算出来，不看屏上有哪个键。
       `route()` 只在**层级图里没登记的页面**（facility / unknown）上回落到这里，
       那时候连"我在第几层"都不知道，唯一安全的动作就是逐层退一步。

    曾经这里有个 `prefer_home` 开关，用来切换"逐层退 / 直奔大厅"——
       **全仓没有一个调用方传过它**，等于 `回大厅按钮` 在 live 路径上从没被点过，
       而注释还写着"由调用方声明意图"。开关和它那段说明一起删了，
       语义改由 `route(target=...)` 承载。
    """
    if st.page == "lobby":
        return None
    if st.page not in _EXITABLE:
        # UNKNOWN / battle / 结算页 —— 一律不动。等它自己走完，或等 runner
        #   的 stuck 处理接管。这是 §A1 的具体落地。
        return None
    # `facility` 是**泛化**签名（只要 回大厅+返回键 同时在场就算），
    #    它认得出"我在某个设施里"，但认不出"是哪个"。进战斗/换页的过渡期
    #    也可能连续几帧只剩这两个按钮  一确认就退出，就又变成用户现场看到的
    #    「刚要进战斗，bot 又手贱去点返回大厅」（§A1）。
    #     泛化页面要**多待一会儿**才允许走人。真的卡在某个设施里，3 秒
    #      绰绰有余；而任何转场都活不过 90 帧。
    if st.page == "facility" and st.frames_in_page < 90:
        return None
    # 顺序：取消 > 叉叉 > **返回上一层** > 回大厅。
    #    模态框上的叉叉 conf 0.96 也可能完全不吃点击（悬赏弃 6 票的根因之一）。
    #
    # **返回排在回大厅前面**（2026-08-10 用户点名：「你去 event 怎么退到大厅，
    #    只需要退回任务大厅就行了啊，现在根本没有连贯性在里面」）：
    #    这个函数是**下一个 flow 不认识当前页面时的兜底**。老顺序把 HOME 放在
    #    前面  无论在哪儿都一步蹦回大厅，**中间层根本没机会被下一个 flow 看见**。
    #    实测代价：bounty 收尾后 jfd 接手，走的是
    #      bounty_stage  回大厅  任务大厅  学院交流会（白走两步），
    #    而 bounty/jfd/特殊任务/大赛**入口全都在任务大厅这一层**。
    #    改成逐层退之后：bounty_stage 返回 bounty_branch 返回 task_hall，
    #    jfd 在 task_hall 就认出自己的入口，直接进去。
    #    对入口真在大厅的 flow（event/cafe/schedule…）只是多按一两下返回，
    #      每一层仍然会被重新分类，不会走丢；退到大厅后 HOME 那条依然兜底。
    c = obs.find(V.CANCEL, 0.45)
    if c is not None:
        return tap_box(c, "nav: 先取消模态框（叉叉在模态上不吃点击）")
    x = obs.find(V.CLOSE_X, 0.55)
    if x is not None:
        return tap_box(x, "nav: 先关掉弹窗")
    # **要么一路返回，要么一路回大厅，绝不混用**（用户 2026-08-12 点名）。
    #    `返回键`=退一层 / `回大厅`=跨越所有层跳回大厅 —— 语义不同，
    #    混着用就成了 live 里那串「点一下返回、紧接着点一下回大厅」。
    #    这个函数的语义是**逐层退**（见上面那段注释：让下一个 flow 在中间层
    #    认出自己的入口，bountyjfd 那次白走两步就是教训）， **只用返回键**；
    #    退不动时交给  系统返回键，而不是半路换成另一套策略。
    b = obs.find(V.BACK, 0.55)
    if b is not None:
        return tap_box(b, "nav: 返回上一层（逐层退，本次退出全程只用返回键）")
    # 最后手段：**系统返回键**（安全性由 `back_key()` 统一把关）。
    #   2026-08-08 实测：大厅上的「社交」浮层（社團/好友/幫手 三张卡）**一个
    #   退出控件都没有** —— 底栏被压暗所以 YOLO 检不到。bot 进去就出不来。
    return back_key(obs, "nav: 屏上没有任何退出控件  系统返回键（浮层专用）")


def route(obs: Observation, st: StateView, target: str = "lobby",
          plan: Optional[dict] = None):
    """朝 `target` 那一层走**一步**。返回 `(Action|None, plan)`。

    和 `to_lobby` 的区别: `to_lobby` 只知道"离开这儿"，`route` 知道"去哪儿"。
       目标由**下一条 flow 的入口在哪一层**决定（`Flow.entry_page`），
       所以 jfd 打完、下一条是同在任务大厅的 event 时，只退到任务大厅就停；
       而 arena 打完、下一条 mail 的入口在大厅时，直接一步回大厅。

    `plan` 是本次归位选定的策略，调用方原样保存并回传（首次传 None）。
       用户 2026-08-12 的硬规矩:「要么返回要么大厅，同一次退出过程中不许换策略」。
       策略在归位开始时算一次，中途屏上冒出/消失哪个键都不改 —— 混用的表现
       就是 live 里那串「点一下返回、紧接着点一下回大厅」，前一下等于白点。
    """
    plan = dict(plan) if plan else {}
    # 覆盖层先关 —— 它盖在底页上，**不参与"我在哪一页"的竞争**，所以
    #    `st.page` 完全可能已经等于目标层而屏上还压着一个确认框/奖励框。
    #    不关就交班 = 下一条 flow 第一帧就在别人的框上点確認。
    if st.overlay:
        a = close_overlay(obs, st.overlay)
        if a is not None:
            return a, plan
        # 覆盖层还在但这一帧没有真按钮(確認/点击继续). 「获得奖励」是横幅
        #    不是按钮, 08-15 归位点了 @(0.500, 0.224) 就是把它当出口.
        #    只要 overlay 没关就等, 别在奖励层下点返回/回大厅.
        return wait(f"归位({target}): 等 {st.overlay} 上的真按钮"
                    f"（获得奖励是横幅, 不点）"), plan
    if st.page == target:
        return None, {}                    # 到位，策略作废
    # 模态框没有层级，只能关掉。**必须先关干净**：不关就交班，下一条 flow
    #   会在别人的弹窗上动手（体外复现: jfd 在悬赏关卡上点了扫荡开始）。
    if st.page in _MODAL:
        # 08-16 remain arena: 编队页 BACK 弹出确认+取消, close_overlay 点取消,
        #    再 BACK, 乒乓烧完 45s. 去大厅优先回大厅键.
        if target == "lobby":
            h = obs.find(V.HOME, 0.55)
            if h is not None:
                return tap_box(h, "归位(大厅): 模态页回大厅键，避免返回+取消乒乓"), plan
        c = obs.find(V.CANCEL, 0.45)
        if c is not None:
            return tap_box(c, f"归位({target}): 取消（模态框唯一有效出口）"), plan
        x = obs.find(V.CLOSE_X, 0.55)
        if x is not None:
            return tap_box(x, f"归位({target}): 关掉弹窗"), plan
        b = obs.find(V.BACK, 0.55)
        if b is not None:
            return tap_box(b, f"归位({target}): 返回键关掉弹窗"), plan
        return back_key(obs, f"归位({target}): 弹窗上没有退出控件"), plan
    if st.page == "lobby":
        e = _ENTRY.get(target)
        if e is None:
            return None, plan              # 大厅到不了的目标，交给 flow 自己进
        if plan.get("entry_sent"):
            return wait(f"归位({target}): 已点入口，等进页（不 38 帧连点）"), plan
        if e == V.NAV_TASKS:
            a = hall_door_from_lobby(
                obs, reason=f"归位({target}): 从大厅进入",
                expect=(V.HUB_CAMPAIGN,),
                post=lambda p=plan: p.__setitem__("entry_sent", True))
            return a, plan
        box = obs.find(e, 0.45)
        if box is None:
            return None, plan
        return tap_box(box, f"归位({target}): 从大厅进入",
                       expect=(V.HUB_CAMPAIGN,),
                       post=lambda p=plan: p.__setitem__("entry_sent", True)), plan
    chain = ancestors(st.page)
    if not chain:
        # 未登记的页面（facility / unknown / blank）—— 沿用老的保守行为。
        return to_lobby(obs, st), plan
    if target not in chain:
        # 目标不在祖先链上（在咖啡厅里要去任务大厅）: 先回大厅，到了再从大厅进。
        target = "lobby"
    dist = chain.index(target) + 1
    if not plan.get("mode"):
        # 隔两层以上、且要一路回到大厅 —— 回大厅键一步到位，胜过按 N 次返回。
        plan["mode"] = "home" if (target == "lobby" and dist >= 2) else "back"
        plan["misses"] = 0
    if plan["mode"] == "home":
        h = obs.find(V.HOME, 0.55)
        if h is not None:
            return tap_box(h, f"归位(大厅): 回大厅键，跨过 {dist} 层"
                              f"（本次退出全程只用这一种）"), plan
        plan["misses"] = plan.get("misses", 0) + 1
        if plan["misses"] < 30:
            return wait(f"归位(大厅): 这一帧没检出回大厅键，等它"
                        f"（{plan['misses']}/30，不改用返回键）"), plan
        plan["mode"] = "back"              # 等不到就明说降级，不悄悄换
    b = obs.find(V.BACK, 0.55)
    if b is not None:
        return tap_box(b, f"归位({target}): 返回键退一层，还差 {dist} 层"
                          f"（本次退出全程只用这一种）"), plan
    return back_key(obs, f"归位({target}): 屏上没有返回键，用系统返回键"), plan


def back_key(obs: Observation, why: str) -> Optional[Action]:
    """系统返回键 —— **全 bot 唯一的发起处**（§A2）。

    大厅上按返回会弹「通知 / 是否結束？/ 取消·確認」，那个確認是
       **退出游戏**。2026-08-08 实测把游戏关掉过一次。
       判据：屏上还看得到 ≥3 个大厅底栏入口 = 我们在大厅（或大厅浮层）上。
    剧情过场也不吃返回键，但那是 interrupt，走不到这里。

    这个函数存在的意义就是**只有一处**：我第一版把同一道闸分别写进了
      `nav.to_lobby` 和 `ExitMixin.exit_step`，**漏了 runner._recover**，
      于是它照样在大厅上按了 6 次返回 —— 正是 memory 里那条
      「修一处没 grep 全仓同形」。
    """
    if obs.count(V.LOBBY_NAV, 0.35) >= 3:
        return None
    return Action(kind="key", keycode="KEYCODE_BACK", reason=why)


def blank_escape(st: StateView, min_frames: int = 45) -> Optional[Action]:
    """空屏（零检出）持续够久  点中央唤回 UI。

    两道前提，缺一不可:
       `page == "blank"`（**len(boxes) == 0**）：屏上什么都没有，也就没有
         任何按钮会被误点。这与老代码盲点「編輯模式」那个 bug 的区别就在这 ——
         那次屏上是有框的（emoticon 已检出），UI 其实在。
       **上一个认得出的页面必须是大厅**。2026-08-08 live 实锤：战斗中
         很多帧 UI 模型零检出  被判 blank  跑去点屏幕正中央，而战斗里
         那一下**可能触发学生技能**。只有大厅点背景才会把 UI 收起来，
         别的地方的 blank 一律是加载/过场，等就行了。
    """
    from routing_v2.act.action import tap_at
    if st.page != "blank" or st.frames_in_page < min_frames:
        return None
    # 黑名单而不是白名单：真正危险的只有"战斗相关页面"（那里点中央可能
    #    触发学生技能）。用白名单（只许大厅之后）的话，**进程刚起来时
    #    last_solid 还是 unknown，bot 会永久卡死在 blank 上**（08-08 实测）。
    if st.last_solid in ("battle", "battle_result", "formation",
                         "stage_popup", "sweep_dialog",
                         # 剧情族（08-09 羁绊剧情实锤）：剧情转场是**纯黑帧**，
                         #    而剧情里点屏幕中央会**推进对话/关掉 menu**，
                         #    等于 bot 自己在乱按剧情。剧情的黑屏一律等，
                         #    真卡住了有 story_cutscene interrupt 负责逃生。
                         "story_cutscene", "story_nodes", "momo_chat",
                         "momo_list", "bond_story_panel"):
        return None
    return tap_at(0.5, 0.5,
                  f"空屏持续 {st.frames_in_page} 帧（上一个页面是大厅）"
                  f"  点背景唤回 UI",
                  justify="屏上零检出，没有任何按钮可被误点；且上一个认得出的"
                          "页面是大厅 —— 只有大厅点背景会收起 UI")


def list_swipe(obs: Observation, anchors, why: str, *, rows: float = 3.0,
               post=None) -> Optional[Action]:
    """列表滑动 —— **几何量全部从检出推**，一个写死的数都不留。

    用户 2026-08-13 定的全局规矩:「**先扫**，确定滑的位置，也确定有没有目标
       然后**再滑**，不然怎么适配其他分辨率以及 aspect ratio」。
       "扫"由调用方做（先找目标，找到就根本别调这里）；这里只负责把
       **滑哪条轴、滑多远**从屏上那一列条目推出来。

    在这之前全仓 5 处滑动写的都是「起点 0.72、终点 0.40」这种常量 ——
       那是拿**某一个分辨率下量出来的比例**当普适值，正是 memory
       [[read_layer_icon_units]] 那条「屏幕比例只在标定它的那个分辨率上成立」
       的同族违例（实测设备就有 19 种分辨率）。

    `anchors` = 和目标**同一列**的条目 cls（邀请键 / 货架价签 / 左栏 tab …）:
       它们的 cx 给轴线、框高给"一行多高"、cy 给可视区下沿。
    返回 None = 屏上连锚点都没有  **不许瞎滑**（fail-closed）。
    """
    bs = obs.all(anchors, 0.35)
    if not bs:
        return None
    xs = sorted(b.cx for b in bs)
    hs = sorted(max(b.y2 - b.y1, 0.008) for b in bs)
    cx = xs[len(xs) // 2]                       # 中位, 别被离群锚带偏
    rowh = hs[len(hs) // 2]
    # **行距优先用相邻行的 cy 间距**，锚框自身高度只是兜底（2026-08-13 用户
    #    抓到: 活动商店的商品卡比上面的「購買」按钮高好几倍，拿按钮高度当
    #    行高，3 行按钮高 = 半张卡，滑动幅度太小、同一屏反复扫）。
    #    cy 间距才是"下一行离这一行多远"的事实; 网格布局下同排多列的 cy
    #    相近，先按 rowh*0.5 聚成行再取相邻行距的中位数。
    row_cys = []
    for cy in sorted(b.cy for b in bs):
        if not row_cys or cy - row_cys[-1] > rowh * 0.5:
            row_cys.append(cy)
    gaps = [b - a for a, b in zip(row_cys, row_cys[1:])]
    if gaps:
        gap = sorted(gaps)[len(gaps) // 2]
        if rowh < gap < 0.35:                   # 间距离谱(跨栏/漏检)就不采
            rowh = gap
    y0 = min(0.92, max(b.cy for b in bs) + rowh * 0.6)
    y1 = max(0.08, y0 - rowh * rows)
    if y0 - y1 < rowh * 0.8:                    # 推不出有效距离就别滑
        return None
    return swipe(cx, y0, cx, y1, why, post=post)


def enter(obs: Observation, cls: str, why: str = "") -> Optional[Action]:
    """从大厅点某个设施入口。"""
    box = obs.find(cls, 0.45)
    return tap_box(box, f"nav: 进入 {why or cls}") if box is not None else None


def lobby_enter(flow, obs: Observation, cls: str, why: str = "",
                expect=(), conf: float = 0.45, post=None):
    """大厅只点当前 job 入口. 08-16: 同帧多入口不抢; once+expect 挡 38 帧连点."""
    if cls == V.NAV_TASKS:
        follow = hall_door_followup(obs, flow)
        if follow is not None:
            return follow
    if not flow.pending("lobby_enter"):
        return wait(f"已点 {why or cls} 入口，等进页（不 38 帧连点）")
    if cls == V.NAV_TASKS:
        return hall_door_from_lobby(
            obs, reason=f"nav: 进入 {why or cls}",
            expect=expect, post=post, once="lobby_enter")
    box = obs.find(cls, conf)
    if box is not None:
        return tap_box(box, f"nav: 进入 {why or cls}",
                       once="lobby_enter", expect=expect, post=post)
    return None


def to_task_hall(obs: Observation, st: StateView) -> Optional[Action]:
    if st.page == "task_hall":
        return None
    if st.page == "lobby":
        return hall_door_from_lobby(obs, reason="nav: 进入 任务大厅",
                                    expect=(V.HUB_CAMPAIGN,))
    return to_lobby(obs, st)


def hub_tile(obs: Observation, cls: str) -> Optional[Action]:
    """在任务大厅点某个玩法 tile。"""
    box = obs.find(cls, 0.45)
    return tap_box(box, f"nav: 任务大厅  {cls}") if box is not None else None


def dot_on(obs: Observation, box, radius: float = 0.05) -> bool:
    """某个 tile / 入口上有没有挂红点。

    大厅红点**要按距离判归属**（老代码实测: Mail 有红点但 claims=0 是
       **判断正确**，那个红点属于旁边的九宫格）。所以这里用最近邻 + 半径，
       不是"屏上有红点就算"。
    """
    if box is None:
        return False
    # 08-16 live taskhall1: 黄点在 tile 右上角, 名字框中心对不上 radius 0.05。
    #    命中=点在框内, 或离框心 <=radius, 或离框右上角邻近(大厅入口仍走框心)。
    tr_r = max(radius, 0.08)
    for d in obs.all([V.DOT_RED, V.DOT_YELLOW], 0.40):
        if box.contains(d.cx, d.cy):
            return True
        if ((d.cx - box.cx) ** 2 + (d.cy - box.cy) ** 2) ** 0.5 <= radius:
            return True
        if ((d.cx - box.x2) ** 2 + (d.cy - box.y1) ** 2) ** 0.5 <= tr_r:
            return True
    return False


# 剧情 hub 的同类剧情卡是**堆叠**的(2026-08-21 用户口述 + CC live 实测)。
#   实测四件套: flywheel_v19_ui_20260821 的 v19_024 / 045 / 047 / 048。
#   主线剧情有 第1部 / 第2部 两张叠在一起:
#     前卡完整可见, 标签框宽 0.124-0.135(框住"第N部主線劇情"整串);
#     后卡被前卡压住只露一条, 标签框宽 0.062(只剩"線劇情"之类的残字)。
#   **点前卡 = 进那部剧情; 点后卡 = 把它翻到最前面, 不进入。**
#   短篇 / 支线各只有一张卡(框宽 0.077-0.079), 没有后卡。
#   同一张卡会吐几个重叠框(NMS 残留, conf 0.09-0.21), 要先按 IoU 去重。
_STORY_CARD_CONF = 0.25
# 黄点贴在它那张卡的右上角; 卡每往后退一层, 点就往右下退一档。
#   实测两档: 前卡 cx=0.410 / 后卡 cx=0.490(差 0.080), 而前卡标签右缘 ~0.389
#    前卡 delta=+0.021, 后卡 delta=+0.101。阈值取中间的 0.06。
_STORY_DOT_FRONT_DX = 0.06


def _iou(a, b) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    ua = a.w * a.h + b.w * b.h - inter
    return inter / ua if ua > 0 else 0.0


def story_cards(obs: Observation, cls: str, conf: float = _STORY_CARD_CONF):
    """把同一类剧情卡的检出拆成 `(前卡, [后卡...])`。没检出返回 `(None, [])`。

    前卡 = 去重后**最宽**那个框(后卡被压住必然更窄, 这是遮挡的必然结果,
      不是写死的尺寸阈值)。后卡按 cx 从小到大排, 第一个就是紧挨着前卡那张。
    """
    raw = sorted(obs.all(cls, conf), key=lambda b: -b.conf)
    keep = []
    for b in raw:
        if all(_iou(b, k) < 0.5 for k in keep):
            keep.append(b)
    if not keep:
        return None, []
    front = max(keep, key=lambda b: b.w)
    backs = sorted((b for b in keep if b is not front), key=lambda b: b.cx)
    return front, backs


def story_dot_on_front(front, dot) -> bool:
    """这个黄点挂的是不是**前卡**。front/dot 任一为 None 时当"不是"。"""
    if front is None or dot is None:
        return False
    return (dot.cx - front.x2) < _STORY_DOT_FRONT_DX


def story_stack_dot(obs: Observation, front, conf: float = 0.40):
    """取剧情卡顶那一行的黄点(卡顶 y 远在标签之上, 用 front 的上缘定带)。

    只认落在前卡上缘**之上**、且横向不比前卡左缘更左的那些 —— 隔壁短篇/支线
    卡上的黄点(cx 0.71)不能算进来。
    """
    if front is None:
        return None
    hits = [d for d in obs.all(V.DOT_YELLOW, conf)
            if d.cy < front.y1 and front.x1 - 0.05 <= d.cx <= front.x2 + 0.30]
    return min(hits, key=lambda d: d.cx) if hits else None
