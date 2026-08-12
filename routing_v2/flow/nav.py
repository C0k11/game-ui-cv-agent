# -*- coding: utf-8 -*-
"""导航 —— 「从这里到那里」的**唯一**实现。

⛔为什么导航必须集中（§A1 + §A2）:
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

from routing_v2.act.action import Action, tap_box, wait
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView

# 这些页面上，"退出"是安全且有意义的
_EXITABLE = {
    "cafe", "cafe_invite_list", "schedule_region", "schedule_area", "craft",
    "shop", "arena_shop", "mail", "daily_mission", "club", "momo_list",
    "momo_chat", "story_hub", "story_nodes", "task_hall", "arena",
    "bounty_branch", "bounty_stage", "jfd_academy", "jfd_stage",
    "campaign_stage", "event_page", "event_quest_list", "event_ended", "event_shop",
    "facility",          # ⭐没登记的设施页 —— 有退出控件，能安全走人
    "combo_pack", "shop_pyroxene_tab", "confirm_dialog", "stage_popup",
    "sweep_dialog", "formation",
}


def to_lobby(obs: Observation, st: StateView,
             prefer_home: bool = False) -> Optional[Action]:
    """离开当前页面。**只在已确认的具名页面上动手。**

    ⛔⛔**「什么时候返回、什么时候回大厅」由目的决定，不由"屏上有哪个键"决定**
       （用户 2026-08-12:「不要又点返回又点大厅按钮，要么返回要么大厅，
         **根据逻辑语义**重新修改」；我第一版只是把 HOME 删掉，
         **那是删症状不是修根因** —— 用户当场指出）。

       两个键语义完全不同:
         · `返回键` = 退**一层**；退到中间层就停，下一个 flow 可能正好在那儿
                      有入口（bounty→jfd 都在任务大厅，逐层退能省两步）
         · `回大厅` = 跨越所有中间层**直达大厅**

       ⇒ 规则（调用方声明意图，函数不猜）:
         `prefer_home=False`（默认）= **逐层退**，全程只用返回键。
             用在"收工后交给下一个 flow"——中间层也可能是它要的。
         `prefer_home=True`        = **直奔大厅**，优先 HOME。
             用在"我要回大厅重新出发"（入口只在大厅、或迷路了要归零）。
       ⚠无论哪种，**同一次退出过程中不许换策略** —— 混用就是 live 里那串
         「点一下返回、紧接着点一下回大厅」，前一下等于白点。
    """
    if st.page == "lobby":
        return None
    if st.page not in _EXITABLE:
        # ⛔UNKNOWN / battle / 结算页 —— 一律不动。等它自己走完，或等 runner
        #   的 stuck 处理接管。这是 §A1 的具体落地。
        return None
    # ⛔⛔`facility` 是**泛化**签名（只要 回大厅+返回键 同时在场就算），
    #    它认得出"我在某个设施里"，但认不出"是哪个"。进战斗/换页的过渡期
    #    也可能连续几帧只剩这两个按钮 ⇒ 一确认就退出，就又变成用户现场看到的
    #    「刚要进战斗，bot 又手贱去点返回大厅」（§A1）。
    #    ⇒ 泛化页面要**多待一会儿**才允许走人。真的卡在某个设施里，3 秒
    #      绰绰有余；而任何转场都活不过 90 帧。
    if st.page == "facility" and st.frames_in_page < 90:
        return None
    # ⛔顺序：取消 > 叉叉 > **返回上一层** > 回大厅。
    #    模态框上的叉叉 conf 0.96 也可能完全不吃点击（悬赏弃 6 票的根因之一）。
    #
    # ⭐**返回排在回大厅前面**（2026-08-10 用户点名：「你去 event 怎么退到大厅，
    #    只需要退回任务大厅就行了啊，现在根本没有连贯性在里面」）：
    #    这个函数是**下一个 flow 不认识当前页面时的兜底**。老顺序把 HOME 放在
    #    前面 ⇒ 无论在哪儿都一步蹦回大厅，**中间层根本没机会被下一个 flow 看见**。
    #    实测代价：bounty 收尾后 jfd 接手，走的是
    #      bounty_stage → 回大厅 → 任务大厅 → 学院交流会（白走两步），
    #    而 bounty/jfd/特殊任务/大赛**入口全都在任务大厅这一层**。
    #    改成逐层退之后：bounty_stage →返回→ bounty_branch →返回→ task_hall，
    #    jfd 在 task_hall 就认出自己的入口，直接进去。
    #    ⚠对入口真在大厅的 flow（event/cafe/schedule…）只是多按一两下返回，
    #      每一层仍然会被重新分类，不会走丢；退到大厅后 HOME 那条依然兜底。
    c = obs.find(V.CANCEL, 0.45)
    if c is not None:
        return tap_box(c, "nav: 先取消模态框（叉叉在模态上不吃点击）")
    x = obs.find(V.CLOSE_X, 0.55)
    if x is not None:
        return tap_box(x, "nav: 先关掉弹窗")
    # ⛔⛔**要么一路返回，要么一路回大厅，绝不混用**（用户 2026-08-12 点名）。
    #    `返回键`=退一层 / `回大厅`=跨越所有层跳回大厅 —— 语义不同，
    #    混着用就成了 live 里那串「点一下返回、紧接着点一下回大厅」。
    #    这个函数的语义是**逐层退**（见上面那段注释：让下一个 flow 在中间层
    #    认出自己的入口，bounty→jfd 那次白走两步就是教训），⇒ **只用返回键**；
    #    退不动时交给 ⑤ 系统返回键，而不是半路换成另一套策略。
    if prefer_home:
        h = obs.find(V.HOME, 0.55)
        if h is not None:
            return tap_box(h, "nav: 直奔大厅（本次退出全程只用回大厅键）")
    b = obs.find(V.BACK, 0.55)
    if b is not None:
        return tap_box(b, "nav: 返回上一层（逐层退，本次退出全程只用返回键）")
    # ⭐最后手段：**系统返回键**（安全性由 `back_key()` 统一把关）。
    #   2026-08-08 实测：大厅上的「社交」浮层（社團/好友/幫手 三张卡）**一个
    #   退出控件都没有** —— 底栏被压暗所以 YOLO 检不到。bot 进去就出不来。
    return back_key(obs, "nav: 屏上没有任何退出控件 → 系统返回键（浮层专用）")


def back_key(obs: Observation, why: str) -> Optional[Action]:
    """系统返回键 —— **全 bot 唯一的发起处**（§A2）。

    ⛔⛔大厅上按返回会弹「通知 / 是否結束？/ 取消·確認」，那个確認是
       **退出游戏**。2026-08-08 实测把游戏关掉过一次。
       判据：屏上还看得到 ≥3 个大厅底栏入口 = 我们在大厅（或大厅浮层）上。
    ⛔剧情过场也不吃返回键，但那是 interrupt，走不到这里。

    ⚠这个函数存在的意义就是**只有一处**：我第一版把同一道闸分别写进了
      `nav.to_lobby` 和 `ExitMixin.exit_step`，**漏了 runner._recover**，
      于是它照样在大厅上按了 6 次返回 —— 正是 memory 里那条
      「修一处没 grep 全仓同形」。
    """
    if obs.count(V.LOBBY_NAV, 0.35) >= 3:
        return None
    return Action(kind="key", keycode="KEYCODE_BACK", reason=why)


def blank_escape(st: StateView, min_frames: int = 45) -> Optional[Action]:
    """空屏（零检出）持续够久 → 点中央唤回 UI。

    ⛔两道前提，缺一不可:
      ① `page == "blank"`（**len(boxes) == 0**）：屏上什么都没有，也就没有
         任何按钮会被误点。这与老代码盲点「編輯模式」那个 bug 的区别就在这 ——
         那次屏上是有框的（emoticon 已检出），UI 其实在。
      ② **上一个认得出的页面必须是大厅**。⛔2026-08-08 live 实锤：战斗中
         很多帧 UI 模型零检出 ⇒ 被判 blank ⇒ 跑去点屏幕正中央，而战斗里
         那一下**可能触发学生技能**。只有大厅点背景才会把 UI 收起来，
         别的地方的 blank 一律是加载/过场，等就行了。
    """
    from routing_v2.act.action import tap_at
    if st.page != "blank" or st.frames_in_page < min_frames:
        return None
    # ⛔黑名单而不是白名单：真正危险的只有"战斗相关页面"（那里点中央可能
    #    触发学生技能）。用白名单（只许大厅之后）的话，**进程刚起来时
    #    last_solid 还是 unknown，bot 会永久卡死在 blank 上**（08-08 实测）。
    if st.last_solid in ("battle", "battle_result", "formation",
                         "stage_popup", "sweep_dialog",
                         # ⛔剧情族（08-09 羁绊剧情实锤）：剧情转场是**纯黑帧**，
                         #    而剧情里点屏幕中央会**推进对话/关掉 menu**，
                         #    等于 bot 自己在乱按剧情。剧情的黑屏一律等，
                         #    真卡住了有 story_cutscene interrupt 负责逃生。
                         "story_cutscene", "story_nodes", "momo_chat",
                         "momo_list", "bond_story_panel"):
        return None
    return tap_at(0.5, 0.5,
                  f"空屏持续 {st.frames_in_page} 帧（上一个页面是大厅）"
                  f" → 点背景唤回 UI",
                  justify="屏上零检出，没有任何按钮可被误点；且上一个认得出的"
                          "页面是大厅 —— 只有大厅点背景会收起 UI")


def enter(obs: Observation, cls: str, why: str = "") -> Optional[Action]:
    """从大厅点某个设施入口。"""
    box = obs.find(cls, 0.45)
    return tap_box(box, f"nav: 进入 {why or cls}") if box is not None else None


def to_task_hall(obs: Observation, st: StateView) -> Optional[Action]:
    if st.page == "task_hall":
        return None
    if st.page == "lobby":
        return enter(obs, V.NAV_TASKS, "任务大厅")
    return to_lobby(obs, st)


def hub_tile(obs: Observation, cls: str) -> Optional[Action]:
    """在任务大厅点某个玩法 tile。"""
    box = obs.find(cls, 0.45)
    return tap_box(box, f"nav: 任务大厅 → {cls}") if box is not None else None


def dot_on(obs: Observation, box, radius: float = 0.05) -> bool:
    """某个 tile / 入口上有没有挂红点。

    ⛔大厅红点**要按距离判归属**（老代码实测: Mail 有红点但 claims=0 是
       **判断正确**，那个红点属于旁边的九宫格）。所以这里用最近邻 + 半径，
       不是"屏上有红点就算"。
    """
    if box is None:
        return False
    d = obs.nearest([V.DOT_RED, V.DOT_YELLOW], to=(box.cx, box.cy), conf=0.40)
    if d is None:
        return False
    return ((d.cx - box.cx) ** 2 + (d.cy - box.cy) ** 2) ** 0.5 <= radius
