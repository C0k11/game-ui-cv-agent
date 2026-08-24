"""Build the ui_v2 dataset for ui_yolo26m_v5 (warm-start fine-tune).

v5 plan (vs ui_v1 which v4 trained on):
  - Schema = MASTER _classes.txt (451 cls, incl 450 选择购买). ui_v1 was 450.
  - REAL sources are DEDUPED by content md5 — run_20260521_103956_distinct was
    628 unique frames blown to 12003 via 200x oversample copies (95% dup). We
    keep the 628 uniques. Heavy dup was the v1/v2/v3 overfit driver; v4 only
    survived it via aug+synth. Cut it.
  - NEW real data merged: run_20260531_110516 (1052) + run_20260531_143201 (37
    shop frames — fills 选择购买/已选中/绿勾).
  - cls92 战术大赛对战选择区域 KEPT (no DROP_CLASSES).
  - MODERATE re-oversample: classes with 8..TARGET unique frames  duplicate to
    TARGET (~30) so dedup doesn't starve the old-only event/battle classes that
    aren't in the new captures. Classes with <8 unique are NOT duped (that's the
    overfit trap — they rely on synth / future real captures instead).
  - 450 选择购买: explicit higher target (50 ≈ ×3 of its 17 real frames) per user.
  - SYNTH kept (441/442/449 bond): _synth_bond{,_goto,_enter} — proven in v4.
  - Val = _ui_val_pool (blind to weak cls; only for early-stop mechanics — real
    judgement is dashboard visual on the shipped model).

Output: D:/Project/ml_cache/models/yolo/dataset/ui_v2/
Usage: py scripts/build_ui_v2.py [--clean]
"""
from __future__ import annotations
import argparse
import hashlib
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"
MASTER = RAW / "_classes.txt"
OUT_ROOT = Path("D:/Project/ml_cache/models/yolo/dataset/ui_v2")

sys.path.insert(0, str(REPO / "scripts"))
from build_ui_dataset import sanitize_label_text  # noqa: E402

REAL_SOURCES = [
    "run_20260521_103956_distinct",  # 628 unique (dedup from 12003)
    "run_20260527_094158",
    "run_20260527_101545",
    "run_20260518_002646",
    "run_20260529_000756",
    "run_20260529_123209",
    "run_20260531_110516",           # NEW 1052
    "run_20260531_143201",           # NEW 37 shop
    # 2026-08-04 移除 "run_20260518_163513" —— 注释曾写「+335 daily-skill UI」,
    # **实测该队列 2,197 个框全部是头像类 143-394, UI 类零框**。经 _keep_ui_lines
    # 剥掉头像段后 335/335 帧标签变空  被当**纯背景负样本**喂进 UI 训练 335 次。
    # 而原图上满屏都是要检的 UI: MomoTalk 弹窗 + 右上角叉叉(cls19) / 咖啡厅
    # 邀請键(cls32)×6 / 底栏八个入口 / 顶栏体力·信用点·青辉石 / 红黄点。
    #  等于反复教模型「这些 UI 不存在」。v14 在这些帧上 conf=0.25 **零检出**,
    #   与 memory 里两个悬案对得上: 「MomoTalk unreachable」「咖啡厅弱」。
    # 该队列对 fused_avatar 仍有价值(2,197 个头像框), 只是**不该进 UI 训练集**。
    #  教训: 源码注释说它是什么**不算数**, 必须去数真实 label 内容。
    "run_20260531_173326",           # +706
    "run_20260531_174456",           # +285
    "run_20260531_175038",           # +36
    "run_20260603_134626",           # +27 (06-03 补录批, 进 train)
    "run_20260603_134649",           # +31
    "run_20260603_140153",           # +31
    "run_20260603_170116",           # +31
    "run_20260603_175151",           # +30
    "run_20260603_175217",           # +30
    "run_20260603_175238",           # +30
    "run_20260603_175257",           # +24
    "run_v6weak_20260603",           # +1130 飞轮弱cls ( 开训前补头像/摸头 + 确认烧录帧已清)
    # emoticon merge (v6): 200 cafe frames, 592 Emoticon_Action(451) bubbles +
    # teacher-relabeled cafe UI (咖啡厅收益/邀请卷 etc) via build_emoticon_ui_source.py.
    # Folds the standalone emoticon_yolo26n into the ui model — pipeline then runs
    # one fewer YOLO per cafe tick. 2026-03 captures, md5-disjoint from above.
    "_emoticon_v2",
    "run_20260607_193003",           # v7 飞轮: 女仆背景 lobby/cafe 弱类(制造入口等) 783标
    "run_20260607_140123",           # v7 飞轮: 银发背景 lobby/cafe 弱类 1409标
    # v8 飞轮 (2026-06-09/10 全技能 live 素材, curate_flywheel 去重 + 多 teacher
    # add-only 补标 + 红黄点 HSV 仲裁清洗 + hub badge 模板锚定补 452):
    # 制造入口544 / 任务大厅入口542 / 双倍三倍617+76hub / 短篇网格 / 剧情战斗结算。
    # 两个 06-10 val run 已抽离进 _val_v8flywheel(整 run 抽, 防同 session 泄漏)。
    "run_20260610_v8queue",
    "run_20260610_024533",           # 用户手标批量扫荡 dialog 全套(127帧, 新类 455-468 主源)
    # v8b: 旧款箭头增压 — v8 训到 ep20 左右切换在旧风格(lobby/任务屏白chevron)上
    # 灾难遗忘(momo新款47/47 vs 旧款15/238), 旧款帧×重编码副本拉回锚点。
    "_arrow_boost",
    # v9 飞轮 (2026-06-11 全天 live 干净帧, 用户全手标/人审, review_v9_pools 复查
    # 标签零问题): 完整日常编排全技能素材 — 455/456 第二session / 450×110 /
    # 452 hub ribbon×576 (v8 val 盲区解药) / 格黑娜vs阿拜多斯 87帧 (v8 混淆解药) /
    # dialog 调暗大厅 / schedule Location Select 滚动多态。
    "run_20260611_044844_clean",
    "run_20260611_050507_clean",
    "run_20260611_051200_clean",
    "run_20260611_052955_clean",
    "run_20260611_053359_clean",
    "run_20260611_053709_clean",
    "run_20260611_055934_clean",
    "run_20260611_061637_clean",
    "run_20260611_061938_clean",
    "run_20260611_064804_clean",
    "run_20260611_071526_clean",
    "run_20260611_072242_clean",
    "run_20260611_073139_clean",
    "run_20260611_073341_clean",
    "run_20260611_074607_clean",
    # v9 晚间专录: 战术大赛商店(新类469-473 主源, 含472/473能量饮料+471货币) +
    # cafe emoticon 高帧×2 (451×747 — 摸头折叠进 ui 的底气)。
    "run_20260611_205439",
    "run_20260611_205540",
    "run_20260611_212919",
    # v9 防遗忘考古回收 (2026-06-11 全历史盘点, 用户"都有用的, 免得遗忘很多cls"):
    # 171121 = 06-03 被弃 ex-val(513帧6,809框 UI 大池, 弃因=与v6c train同session泄漏
    #  进 train 反而合法; 入库前 HSV 修复 29 处黄点红点 — 它没吃过 v8 那轮清洗)。
    # _ui_val_pool = 05-27 老风格旧 val(51帧1,010框 红黄点/货币/邀请键 rehearsal)。
    # 考古明确排除: 021030(与v8queue同内容已重编码, 双标签互污) / 173604+183022(纯
    # 头像归fused域) / _synth_*(用户06-06砍的毒) / expanded·full·static(古schema不兼容)。
    "run_20260603_171121",
    "_ui_val_pool",
    #  v10 素材 (2026-06-13, 用户授权开训)
    # 用户手标全审: 313帧(星修110+455补128, commit fd40b59)。
    "run_20260612_191319",
    # bot飞轮 _clean 池(6/12-13 整链/商店/schedule/arena_shop live): v9预标 +
    # 三道清洗(红黄点conf0.55门槛滤位置先验低conf假阳 / 绿勾空框HSV删 / 空帧立绘删)。
    # 残留高conf红黄点假阳无法自动删(用户接受, 飞轮负样本逐步破先验)。
    # chainlive(452, curate自trajectory)被砍 — 与下列_clean池同run内容重复。
    "run_20260612_205142_clean",
    "run_20260612_205201_clean",
    "run_20260612_211227_clean",
    "run_20260612_211617_clean",
    "run_20260612_213424_clean",
    "run_20260612_213916_clean",
    "run_20260612_214301_clean",
    "run_20260612_215139_clean",
    "run_20260612_215555_clean",
    "run_20260612_215845_clean",
    "run_20260613_044708_clean",
    "run_20260613_050835_clean",
    "run_20260613_051748_clean",
    "run_20260613_053333_clean",
    # v11: 2026-06-13 全天 live (整链/arena_shop/schedule/bounty/jfd step_mode walk
    # + 最终 autonomous arena/mail/daily/batch_sweep). v10 预标, 用户 dashboard 人审
    # (无可见假阳, 仅 cafe 漏摸1 — 451 弱待明天 cafe 帧强化). 18 空帧已删。
    "run_20260613_171257_clean",
    "run_20260613_171928_clean",
    "run_20260613_174617_clean",
    "run_20260613_175555_clean",
    "run_20260613_180407_clean",
    "run_20260613_183133_clean",
    "run_20260613_183328_clean",
    "run_20260613_185122_clean",
    "run_20260613_190244_clean",
    "run_20260613_190646_clean",
    "run_20260613_203352_clean",
    #  v12 素材 (2026-06-14, 用户 dashboard 人审"标注无问题")
    # 任务大厅 skill live (arena/mail/daily/batch_sweep) + arena_shop step_mode walk
    # 的干净帧, v11 预标. 关键: run_20260614_205540 = 用户手录战术大赛商店素材(245帧,
    # 469战术大赛商店未选中tab的解药 — 之前仅27实例饿着; 含470已选择/471货币/472下级/473一般
    # 能量饮料). 5 空帧已删入 _unlabeled_backup.
    "run_20260614_202243_clean",
    "run_20260614_203531_clean",
    "run_20260614_205111_clean",
    "run_20260614_205540",
    "run_20260614_210716_clean",
    "run_20260614_211302_clean",
    "run_20260614_213324_clean",
    "run_20260614_213728_clean",
    "run_20260614_213959_clean",
    #  v13 素材 (2026-07-08, 用户 dashboard 人审)
    # 0616全skill/0626/0706-0708 大合并池(原9源池已删, 唯一副本): 活动(遊戲開發部
    # 大作戰)全链 quest/编队/加成/商店/里程碑 + reset日 daily + shop/arena_shop。
    # 管线清洗过: 红黄点HSV闸(-1659假点) / WIN误标397(-18) / 405<->474语义分家(51)
    # / 96活动任务锚定补标(+485) / 475奖励资讯改名清洗 / 战斗帧已分离(battle池,
    # 另训战斗模型) / 147空帧已删。新类: 474距离奖励获得结束 / 475奖励资讯。
    "run_20260617_merged_clean",
    #  v14 素材 (2026-07-17, 用户 dashboard 人审 + 误标四类批量清零)
    # 0711 大合并池: 7/11 daily 全链(战术大赛/悬赏/信用商店/青辉石商店/momo)。
    # 103 灰态清理过(5 灰暗删, 亮蓝 124 保留)。
    "run_20260711_merged_clean",
    # 0717 大合并池(16 源池, 帧名前缀溯源 evshop/0709m/0717a…): 午夜派对活动商店
    # (evshop 77帧: 103 OCR辅助标注165亮态/101_已选择30/102x150/104x19, 灰态遮罩全剔)
    # + 7/09-7/17 daily 走线 dHash 稀疏后 distinct 帧 + idx77 当期活动入口(用户手标)
    # + cls451 摸头(prefill span 修复后补标53框)。误标清零: Challenge 语义框/405<->474
    # 混淆(2框)/灰确认 2023(41框)。
    "run_20260717_merged_ui_clean",
    #  v15 教学循环产出 (2026-08-01/02 与用户逐帧协作人审, 非预标)
    # 新 cls 485 开始制造灰色 / 486 材料不足 / 487 蓝矿 / 489 购买灰色 的
    # **唯一**训练来源; 另含 404 全部选择灰 与 110 Bonus 的漂移修正。
    # 只收这两个**人审过**的队列 —— 155 个 run_*_clean 里那 7,057 帧是
    # v14 **预标**(scripts/prelabel_flywheel_inplace.py), 直接入训 = 拿模型
    # 自己的输出喂自己(自举偏差), 违反「预标≠标注」纪律, 必须先过 dashboard。
    "flywheel_20260801",              # 592 帧: craft材料不足/信用点商店买完态/Bonus
    "flywheel_event_shop_20260728",   # 107 帧: 活动商店售罄黑条/確認灰/stepper灰
    # 75 帧 craft 材料不足场景(walk 原始录制, 跨 5 天): 486 材料不足 71 框 +
    # 487 蓝矿 75 框 —— 这两类原本各只有 1 框, 等于白建。入队时已用 S<94
    # 判据把 v14 误检的 71 个亮态 444 改成 485(不改就把今天修的又毒回去)。
    "flywheel_craft_20260802",
]
SYNTH_SOURCES = []   # v7: 砍头像 synth(头像归 fused v6 专精; rehearsal 仅 unified 才需要)  ui v7 纯 UI+emoticon 真实帧
                 # ️ v6c (2026-06-06 用户决策): 砍 _synth_ui_swap — UI 只用真实帧根治 synth 过拟合
                 #    (v6b 实锤: UI val 0.892 高 / live 崩, 咖啡厅入口 val>0.9 / live 仅 0.25)。头像 synth
                 #    影响小保留。UI 弱类(咖啡厅入口/开始制造/CAFE_EARNINGS)暂靠 skill 兜底(cafe/craft 外推),
                 #    v7 再上飞轮真实帧(run_20260606_flywheel 519帧, 标注后)补。
                 # 删: _synth_bond/goto/enter — 假阴性毒 (2026-06-04)
VAL_SOURCES = [
    "run_20260606_flywheel",  # v7 主 val: 06-06 飞轮 477帧(独立 session 防泄漏, 含 UI 弱类靶子). 旧 06-03 val 弃用(171121 与 v6c train 同 session 泄漏 / 183022 头像 / _ui_val_pool 旧盲)
    # v8 增补 val: 从 v8queue 按整 run 抽的 38 帧(06-09/10 新界面覆盖)。稀有类保护:
    # 任何类被抽走 >40% 或全局剩 <20 实例的 run 不许抽(momo/剧情场景类各 session
    # 垄断, 实测仅 2 run 可安全抽出 — 别强抽, v9 等场景跨天重复后再扩)。
    "_val_v8flywheel",
    # v12 新场景 val (2026-06-16, 用户 dashboard 复审"当 val 没问题"): 0616 全 skill
    # 测试 session 飞轮 48 池合并 v12 预标 2086 帧。补旧 val 的盲区: arena PvP /
    # 批量扫荡 dialog / 特殊任务扫荡 / buy_pyroxene 战术大赛商店 / hub — 旧 val 这些=0。
    # 铁律: 此池整批只进 val, 永不加进 REAL_SOURCES(同 session live 帧, 进 train =
    # 近邻泄漏 val 虚高)。性质: v12-预标人审 silver(非手标 gold) 能测"对 v12 基线有无
    # 回归 + 新场景覆盖", 但测不出 v12 自身盲区(那需独立手标 gold val)。内部 96% 近邻重复
    #  有效 ~40 distinct 场景, mAP 被 dup 计数加权, 看趋势别看绝对值。
    "_val_v12flywheel_0616",
    # v15 缺口 val (build_val_gap_queue 挑的 41 帧, 覆盖 57 类): 命中 16 个
    # 原本 val=0 的缺口类(50房间未解锁/82入场键没解锁/83得星_0/93-97活动族/
    # 122-123部队/128-129战斗HUD/134自动战斗关闭/399地区升级/456批量扫荡/474)。
    # 整批只进 val, 永不进 REAL_SOURCES(同 session live 帧进 train = 近邻泄漏)。
    "_val_v15_gap",
]

#
#  v16 飞轮并入（2026-08-11）—— 历代 live session 的 *_clean 目录
#
# 盘点结果: raw_images 里 **142,078 个 UI 框已标好但从没接进 build**（162 目录），
# 相当于当时训练集的 36%。典型缺口 `购买灰色`(489): 已入训 213 / 未入训 1,206，
# v15 对它召回仅 21%。
#
# **划分单位是「天」不是 run**（memory val_set_crisis 三道闸第一条）: 同一天的
#    多个 run 是同一次 bot 跑的连续片段, 逐 run 划分 = 近邻泄漏, val 会虚高。
#    所以这里只写**天**, 目录名由 _clean_dirs() 展开 —— 让防泄漏规则在代码结构上
#    就看得见, 而不是埋在 157 行目录名里。
# 这些是**预标 silver, 不是人手标 gold**。已核过的:
#    `购买灰色` 1,206 框 —— 裁图与**已入训人审样本**三方对照, V=130/B-R=68
#    对上已入训灰态 V=131/B-R=72（定稿判据 V<150 且 B-R<90），标得对。
#    其余类**尚未逐类人审**, 见 build 末尾打印的「本次新增 per-cls 增量」表。
FLYWHEEL_TRAIN_DAYS = ["20260615", "20260715", "20260721", "20260723",
                       "20260724", "20260726", "20260727", "20260728",
                       "20260729", "20260731"]
# 0722 / 0725 两天整体留作 val（unused_material_audit 选的就是这两天：
# 它们覆盖的 cls 最广, 且当天不是任何 train 类的唯一来源）。
FLYWHEEL_VAL_DAYS = ["20260722", "20260725"]


def _clean_dirs(days) -> list:
    """展开某几天的 `run_<day>_*_clean` 目录。只认磁盘上真实存在的。"""
    out = []
    for d in sorted(p.name for p in RAW.iterdir() if p.is_dir()):
        m = re.match(r"run_(\d{8})_.*_clean$", d)
        if m and m.group(1) in days:
            out.append(d)
    return out


# **从 train 挪到 val 的两天**（2026-08-11）——补 val=0 的缺口。
#    挪整天不挪单个 run：同一天的多个 run 是同一次 bot 跑的连续片段，
#    逐 run 划 = 近邻泄漏（val_set_crisis 三道闸第一条）。
#    0717 补 货币/货币_已选择/货币数量显示区域/後日談；
#    0612 补 困难关卡选中(158)/普通关卡(156)/前置小装备/大装备已选中。
#    代价 = train −22,316 框（−5%）。挑这两天是因为「每搬千框补几个类」
#      的性价比最高，且搬完没有任何类掉到保护线以下。
MOVED_TO_VAL = [
    #  第二批（补剧情/攻略族的 val）
    # 挑天的判据从「val 从 0 变成 >0」改成「**该天至少贡献 8 框**」：
    #    val=1~3 等于也测不出（仓库已经有 15 个这种类），把 0 变成 1 是自欺。
    #    换判据后 0529+0607 胜出，给出的是能用的量：
    #      跳过故事键不可用 139 / 简易攻略 25 / 剧情观看 14 / 剧情中断退出 12 / 2部队 12
    # 代价：跳过故事键不可用 train 20970。可接受。
    "run_20260529_000756", "run_20260529_123209",
    "run_20260607_193003", "run_20260607_140123",
    #  第一批
    "run_20260717_merged_ui_clean",
    "run_20260612_191319", "run_20260612_205142_clean", "run_20260612_205201_clean",
    "run_20260612_211227_clean", "run_20260612_211617_clean",
    "run_20260612_213424_clean", "run_20260612_213916_clean",
    "run_20260612_214301_clean", "run_20260612_215139_clean",
    "run_20260612_215555_clean", "run_20260612_215845_clean",
]

REAL_SOURCES += _clean_dirs(FLYWHEEL_TRAIN_DAYS) + [
    "v2step_20260810",       # routing_v2 step 模式飞轮 582帧/7,859框
    # 任務資訊 顶部两个页签的**两态**（小号现采，8帧×2框）:
    #    495 集中指挥(绿底未选中) / 496 简易攻略_已选中(白底) —— 这两个类
    #    **全仓再没有第二个来源**，所以整批进 train（学不到比测不出更糟）。
    #    代价：它俩暂时 val=0，等主号或另一天补。
    #    和 v2alt_20260811(val) 同一天同 session  严格说有泄漏，但两边内容
    #      不重叠（这边只有关卡弹窗页签，那边是剧情节点），共有的只是顶栏/返回键
    #      这类到处都是的通用件。
    "v2alt_tabs_20260811",
    # 大号战術大賽实时对战（2026-08-11，用户指路「取消勾选跳過戰鬥就能看实时对战」）:
    #    96帧 —— 跳过战斗(勾选)47 / 跳过战斗未选45 / **战斗失败33**(LOSE 横幅，
    #    train 原本只有 6 框) / 战斗三倍速12 / 战斗暂停。
    #    日常链默认开着「跳过战斗」，所以这些战斗内 UI 平时根本采不到。
    "v2main_20260811",
    # **走格子域整族从来没接进过源**（2026-08-12 查 train/val 覆盖时才发现）:
    #    497-510 十三个类（走格子_格子/可走/迷雾/起点黄灰/队伍箭头/BOSS标记/
    #    PHASE结束·自动结束两态/道具/我方/敌方）在建好的集里 **train=0**，
    #    而这两个目录里明明躺着 328 帧标注。 又一次「接了源≠进了集」的**更前一步**：
    #    连源都没接。走格子是 InQuest/SolveChallenge/GridQuest 三条 flow 的共同前置。
    #    两个池同属 08-11 **同一天**  不能一个 train 一个 val（防泄漏闸单位是「天」），
    #      整批进 train。代价是这 13 个类 val=0，等换大号另采一天补 val。
    "v2grid_20260811",       # 201帧 走格子实战
    "v2gridmap_20260811",    # 127帧 走格子地图
    # 小号 Cok11（2026-08-12 现采）—— 大号采不到的态:
    #    511-520 任务页签 5×2 态（**每个页签有自己的主题色**，互相不能泛化）/
    #    521 任务开始_灰色 / 522 清辉石宝箱 / 524 中断任务(亮灰) / 525 重新挑战(亮态)/
    #    527 选择购买灰色 / 468 立即前往 / 任务大厅 7 磁贴全带锁。
    #    这些新类**全仓只有这一个来源**  整批进 train，val=0，等大号补。
    "v2alt2_20260812",
    # 小号 2026-08-12 第二批（318帧）—— 补的全是「族里只标了一态」的另一半:
    #    511-520 任务页签 5×2 态（每个页签有独立主题色, 互相不能泛化  必须各自够量）:
    #      未选中各 ~173 框 / 选中各 ~43 框（每帧只贡献 1 个选中态, 所以跑了 45 轮）
    #    495 集中指挥(绿·未选中) / 496 简易攻略_已选中(白) 各 +25，
    #    421 集中指挥已选中 / 422 简易攻略(蓝) 各 +25 —— 关卡弹窗来回点页签采的，
    #      实测底色 集中指揮未选中 G=231 / 簡易攻略未选中 B=231 / 选中=白，与
    #      [[stage_popup_tabs]] 记的判据完全对上。
    #    493 批量扫荡方案 +210 / 494 已选中 +35 —— 7 个方案页签轮点 5 轮，
    #      框位用**历史口径**(框整个 tab, w≈0.116, 间距 0.124)，245/245 底色校验零错。
    #    这批帧多样性天生低（同一页面反复点） dHash 只拆得出 9 帧 val，其余进 train。
    "v2alt3_20260812",
    # -- 2026-08-13 锁类 + 购买/购买灰色 补强批 --
    #    锁 cls50 泛化: 大厅制造锁/社团卡锁/咖啡厅锁钮/任务大厅5种tile锁/
    #      创建日活動一覽锁/战斗AUTO横幅锁(此前 cls50 只有课程表一个域,
    #      lobby/club/cafe/task_hall 四域 v16 召回实测全 0%)。
    #      框来自 NCC 模板匹配 + 逐帧白锁体分割(与模型无关的独立来源)。
    #    购买/购买灰色: 候选**是模型自己 conf>=0.05 的检出**, 但入训的依据
    #      不是"模型说它是", 而是 **用户逐条看宽上下文审图确认过**
    #      (2026-08-13 当晚, 357 个低分框 conf 0.23-0.60, 用户判定全对,
    #       含我误判成亮态的那几个灰键)。
    #      这条界线要记牢: 拦的是**没人看过**的预标(历史 7,057 帧那批),
    #        不是"来源是模型"本身。人审过的低分检出 = 难例挖掘, 该训;
    #        我当晚按"来源=模型"一刀切撤回, 被用户当场纠正, 已全部还原。
    #      高分档(>=0.6)模型本来就认得, 一并留着无害。
    #    0813 全天同一天, 整批 train, val=0 靠后续换号采(防泄漏闸单位=天)。
    #    注意 MONEY 抓拍帧近邻重复度高(055226/134818 尤甚), build 的内容
    #      去重会砍掉大部分, dashboard 里按唯一帧看。
    "v2_20260813_055226", "v2_20260813_060254", "v2_20260813_061039",
    "v2_20260813_061611", "v2_20260813_062015", "v2_20260813_064800",
    "v2_20260813_084907", "v2_20260813_085236", "v2_20260813_085435",
    "v2_20260813_090249", "v2_20260813_090713", "v2_20260813_092843",
    "v2_20260813_094246", "v2_20260813_094835", "v2_20260813_095318",
    "v2_20260813_100010", "v2_20260813_100215", "v2_20260813_100531",
    "v2_20260813_100651", "v2_20260813_111444", "v2_20260813_111900",
    "v2_20260813_112515", "v2_20260813_112738", "v2_20260813_113434",
    "v2_20260813_113611", "v2_20260813_114042", "v2_20260813_114530",
    "v2_20260813_114834", "v2_20260813_121803", "v2_20260813_121853",
    "v2_20260813_123753", "v2_20260813_131458", "v2_20260813_133201",
    "v2_20260813_133816", "v2_20260813_134041", "v2_20260813_134818",
    "v2alt_tour_20260811", "walk_20260813_082858", "walk_20260813_083604",
    # --- v17 本轮待标暂存区（2026-08-13 深夜建）----
    #    老前端下拉不支持搜索, 所以把本轮要标的帧硬链接归并成两个显眼目录,
    #    源 run 目录(带 _trace.jsonl)原样留档、**永不加进任何源列表**,
    #    否则同一帧两条路进 build = 重复计数。
    #    _TRAIN_v17_0813  387帧: 主线 normal 走格子(2-4/2-5, 浅底小图 = 模型
    #      2-4 盲区同域) + H2-1 全程 + 关卡列表/翻区/入场导航 + manualwalk 44
    #      (其中 14 帧已带 txt) + 金钱闸 HALT 2 帧。
    #    切法按**关卡**不按 run: 今天全是 08-13, 天粒度切不动, 关卡是拿得到的
    #      最强隔离边界 -- val 那批(H2-2)是 Hard 新地图/新敌人/带迷雾, 与这边
    #      任何一关都不重叠。仍属同一天同账号, 有残余乐观偏差, 记在案。
    "_TRAIN_v17_0813",
    # v18: 08-16/18 标修复后的格子/任务大厅包。md5 与已有源撞上的进不了第二份。
    # 不加 _TRAIN_v18_yt / _TRAIN_v18_yt_ace (已并进本目录, 再加会双计)。
    "_TRAIN_v18",
    # 08-12 飞轮已人审过的帧, 含 0000427 活动店 8x103。无 txt 的帧 build 会跳过。
    "flywheel_v16_20260812",
]
REAL_SOURCES = [s for s in REAL_SOURCES if s not in set(MOVED_TO_VAL)]
VAL_SOURCES += _clean_dirs(FLYWHEEL_VAL_DAYS) + [
    # 整批只进 val: 08-11 是**独立的一天**, 与 0810 天然不同源。
    #    新 cls `批量扫荡方案`/`制造槽_空` 靠这一批才有 val（0810 进 train）。
    "v2step_20260811",
    "v2walk_20260811",       # 批量掃蕩方案页签 7帧/49框
    # **战斗 UI 的 val 白捡**（2026-08-11 —— 用户点醒「本地有素材的，你一直不去找」）:
    #    0710 这一天从来没接进过任何源列表, 而它正好带着我原本打算花 AP 进战斗
    #    去现采的那批：战斗1倍速 192 / 战斗2倍速 74 / 重新开始键·继续键·放弃键 各 27。
    #     拿它当 val, **train 一框不掉、也不泄漏**（0710 不在 FLYWHEEL_TRAIN_DAYS）。
    "run_20260710_104427", "run_20260710_104718",
    "run_20260710_110430", "run_20260710_110759",
    "defeat_candidates_v10",   # 战斗失败 28 框 —— 唯一有它的非 train 来源
    # 没用 `run_20260606_flywheel_labels_bak`: 它是 VAL 里 `run_20260606_flywheel`
    #    的**备份副本**, 同源重复, 进 val 只会让指标虚高。
    # 没用 axis_* 战斗考古: 那是录屏来源, 分辨率/渲染都是另一个域,
    #    而且战斗 UI 的缺口 0710 已经补上了。
    # **从 data/trajectories 里挖出来的**（2026-08-11，用户第二次点醒
    #    「找以前老的没用上的，我们本地这么多素材肯定有」）:
    #    trajectories 有 **876 目录 / 113,590 帧 / 标注 0**，比整个 raw_images 还大，
    #    而且每帧的 `tick_*.json` 里存着当年那版模型的 `yolo_boxes` + `sub_state`
    #     **筛选阶段完全不用跑 YOLO**（16 核读 json，几分钟扫完）。
    #    只抽**train 里不存在的天**：0601/0602/0807 两边都没有；
    #      0717/0722/0725 本来就是 val 天。0610/0611/0529 是 train 天，
    #      它们的 trajectory 是同 session 原片，抽去 val = 近邻泄漏，**不碰**。
    #    这批推翻了我一个结论：我判过「剧情未完成态只能开小号采」——
    #      **0602 那天这号还在推进度，`new` / `剧情图标未完成` 全都在**。
    #    0807 的快速製造是**另一套配方**（節點設定/三次節點），
    #      和 flywheel_craft_20260802 那 75 张视觉上不同  蓝矿终于有真 val。
    #    标注 = 现役权重全类预标（json 里那些是旧模型的，只当筛选线索）。
    "_traj_val_0602",        # 296帧: new280 / 剧情图标未完成300 / 已完成300 / 进入章节8
    "_traj_val_0807",        # 236帧: 蓝矿10 / 材料不足10 / 开始制造灰色10 / 後日談26
    "_traj_val_misc",        # 24帧: 跳过战斗未选5 / 战斗胜利1 / 战斗开始1
    # **raw_images 里"有图没标"的帧**（2026-08-11 全盘扫出 27,561 张）:
    #    这些目录早就在源列表里, 但帧没有 .txt  build 直接跳过, 等于白放着。
    #    只取**两边都没有的干净天**做 val（0722/0725 是 val 天, 已就地补标）。
    "_val_unlab_20260730",   # 129帧: 货币252 / 货币_已选择129
    "_val_unlab_20260807",   # 194帧: **蓝矿70** / 材料不足70 / 开始制造灰色70 / 後日談76
    "_val_unlab_20260808",   # 94帧
    "_val_unlab_20260809",   # 48帧: 奖励资讯9
    # **小号**（用户 2026-08-11 现建, 授权探索）: 剧情全是未完成态,
    #    正好补主号上**根本不存在**的那一族。按用户口述的规则导航:
    #    主线看右侧面板的 `黄点` 定位小章节；短篇/支线是网格, 用 `new` 角标当锚。
    "v2alt_20260811",        # 16帧: 剧情图标未完成12 / new33
    # 第二遍：按 `sub_state` 页面上下文从 trajectories 干净天里捞的战斗帧
    #    （第一遍靠旧模型 cls 只捞到 5 帧，换页面上下文捞到 84）
    "_val_cand_battle",      # 97帧: 跳过战斗未选84 / 蓝矿8 / 战斗胜利·开始各1
    # **同池按 dHash 距离拆 val**（2026-08-12，用户纠正我"按天整划"守太死）:
    #    按天整划那道闸当初是为**战斗帧**定的（battle_v10 val 96% 是相邻帧泄漏）。
    #    UI 元素位置固定、外观近乎像素级一致  真正让 val 失效的是**相邻帧**，
    #    不是"同一天"。但不能按固定间隔抽：实测 K=4 抽样有 33-44% 的 val 帧
    #    与最近 train 帧 dHash 距离 <8（近乎同图）。
    #     改成贪心筛「与所有 train 帧最小距离 >= 14」，实测拆出来的 val 帧
    #      距离中位 106(grid) / 71(alt2)，最小 14。
    #    `v2gridmap_20260811` **不拆** —— 地图连拍画面几乎不动，相邻帧距离
    #      中位仅 4，91% 会泄漏，整批留 train。
    "v2grid_20260811_val",     # 91帧: 走格子 497-510 的 val
    "v2alt2_20260812_val",     # 49帧: 任务页签 511-520 / 521·522·524·525·527 的 val
    "v2alt3_20260812_val",     # 9帧: 同上第二批（页签帧内容高度重复，能拆的就这么多）
    # --- v17 本轮 val 暂存区（2026-08-13 深夜建, 与 _TRAIN_v17_0813 配对）---
    #    325帧 = H2-2 的三个 run 全部（19:17 / 19:29 / 19:38）。
    #    为什么整关进 val: 走格子全族 14 个 cls 现在 **val=0**, train 又只来自
    #      `v2gridmap_20260811` 一个目录 -- 没量尺是 v14 那场事故的复发形态。
    #      H2-2 是 Hard 新地图/新敌人/带迷雾, 与 train 侧的 2-4/2-5/H2-1 无同关
    #      重叠, 是今天这批里拿得到的最强隔离边界。
    #    仍是同一天同账号, 顶栏/队伍卡这类通用件必然共享, 指标偏乐观, 记在案;
    #      下次换号/换天再采一批独立的走格子 val 顶掉它。
    #    铁律: 源 run（191728 / 192934 / 193846）永不加进 REAL_SOURCES。
    "_val_v17_0813",
    # v18 格子 holdout 208。与 _TRAIN_v18 同图的会被 md5 踢出, 不要删 train 救 val。
    # 量尺仍不测 489/450/527/528; 类号未逐框, 不当干净。
    "_val_v18",
] + MOVED_TO_VAL

TARGET = 30            # moderate oversample floor (was 200 — the overfit driver)
TARGET_OVERRIDE = {450: 50}   # 选择购买: ~x3 its 17 real frames
MIN_UNIQUE = 8         # don't oversample classes thinner than this (overfit trap)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


#: link_image 的实际落地方式统计, build 末尾打印出来当体检
_LINK_STAT: Counter = Counter()


def link_image(src: Path, dst: Path) -> None:
    """把源图接到数据集里 —— **优先硬链接**, 退符号链接, 最后才真拷贝。

    2026-08-03 实测: 原来只有 `symlink_to`  `copy2` 两级, 而 Windows 建符号链接
    需要管理员权限或开发者模式, 普通进程必抛 OSError  **一个链接都没建成**,
    train 23,032 + val 2,195 张全是真实拷贝, 白占 **23.15 GiB**。
    修法: 插一级 `os.link` 硬链接 —— NTFS 上**不需要任何特权**, 零额外占用,
    对读取完全透明。前提是同卷: 源 D:\\Project\\ai game secretary\\data\\raw_images
    与目标 D:\\Project\\ml_cache\\... 都在 D 盘 (跨卷会抛 OSError, 自动退下一级)。
    硬链接与源共享 inode: 改其中一个会改另一个。这里只读图片不改, 安全;
      但**别把它当备份**。
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)                 # 硬链接: 同卷、免特权、零占用
        _LINK_STAT["hardlink"] += 1
        return
    except OSError:
        pass
    try:
        dst.symlink_to(src.resolve())     # 符号链接: 需特权, 跨卷可用
        _LINK_STAT["symlink"] += 1
        return
    except OSError:
        pass
    shutil.copy2(src, dst)                # 真拷贝: 兜底, 占空间
    _LINK_STAT["copy"] += 1


# v7: ui = 纯 UI+emoticon — drop 头像段(143-394, 归 fused v6 专精)。flywheel / cafe / momo
# 真实帧由 v6c(nc455)预填含头像框, 对 ui v7 多余(否则 val 被头像 GT 干扰 + train 学多余头像)。
# ️ 原始 raw_images 标注不动(保留头像给未来 unified), 仅 build 输出 ui_v2 时过滤。
HEAD_LO, HEAD_HI = 143, 394
# **战场实体**类不进 UI 模型（2026-08-11 并入飞轮时挡下，用户当场点名）:
#    476 我方 / 477 敌方 / 478 塞特的愤怒 / 479 Boss / 480 主教 / 481 球 /
#    482 黑白 / 483 大蛇 —— 这些是**战场上的角色和目标物**, 归 battle 模型。
#    老 UI train 里它们基本是 0 框, 新并入的飞轮目录却带了几千框  等于顺手给
#    UI 模型开一个它从没有过的能力, 且 val=0 完全测不出来。
#    （memory battle_side_confusion: 敌我 22.5% 单向错, 根因是训练集 4.7:1
#      失衡, 不是随便喂点数据能解决的；combat_expansion_backlog 写明
#      「明确不做 per 角色战场 cls」）
# **战斗 UI 留着** —— 用户原话「战斗模型和 ui 有重叠, 也只是部分 ui 而已」:
#    128 战斗暂停 / 129 三倍速 / 130·134 自动战斗 / 131-133 重新开始·继续·放弃 /
#    135·412 倍速 / 136 战斗胜利 / 411 战斗开始 / 436-437 跳过战斗 /
#    447 战斗图标已完成 / 467 战斗完成 / 484 战斗失败 —— 这些是**界面控件**, 归 UI。
DROP_UI = set(range(476, 484))


def frames_in(sd: Path):
    """源目录里的帧。**jpg + png 都要收。**

    2026-08-11 实锤的静默数据丢失：这里原来只有 `sd.glob("*.jpg")`，
       而 routing_v2 的 step/walk 飞轮是 **ADB screencap 出来的 PNG**
       （`v2step_20260810` 582帧 / `v2step_20260811` 471帧 /
         `v2walk_20260811` 26帧，jpg 数量全是 **0**）。
        这三批"已经接进源列表"的素材**一帧都没进数据集**，
         连带三个新 cls（492 制造槽_空 / 493·494 批量扫荡方案）在建好的集里
         **train=0 val=0** —— 而源目录里明明躺着标注文件。
       **没有任何报错**：源列表照常打印、drift 校验照常通过、[done] 照常输出。
         我是去数「建好的集里 492-494 有几框」才发现的。
        教训：接了源不等于进了集，**必须回头数建好的那一份**。

        **烧了框的渲染图一律排除**（2026-08-13）: runner 早期把标注渲染图写成
       顶层的 `*_ann.jpg`（现在改写进 `_ann/` 子目录，但磁盘上那些老的还在）。
       `glob` 是非递归的，子目录里的扫不到；顶层那些**会被照收** ——
       一旦把 live 录制目录接进源，等于拿"模型自己画的框"去教模型
       （[[flywheel_label_import]] 那条 overlay 烧录陷阱的复刻版）。
    """
    return sorted(f for f in list(sd.glob("*.jpg")) + list(sd.glob("*.png"))
                  if not f.stem.endswith("_ann"))


def _keep_ui_lines(cleaned: str, nc: int):
    out = []
    for ln in cleaned.splitlines():
        if not ln.strip():
            continue
        c = int(ln.split()[0])
        if c >= nc or HEAD_LO <= c <= HEAD_HI:   # 越界 或 头像段  drop
            continue
        if c in DROP_UI:                          # 战场角色类  归 battle 模型
            continue
        out.append(ln)
    return out


def label_classes(cleaned: str):
    cs = set()
    for ln in cleaned.splitlines():
        ln = ln.strip()
        if ln:
            cs.add(int(ln.split()[0]))
    return cs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    # 为什么要能换 master: 加新 cls 必须先扩类表, 但 `_classes.txt` 一旦
    #    行数 != 线上权重 nc, `routing_v2/percept/detect._name_table` 会**整表
    #    退回权重自带的名字** —— 那 15 个改名 / 18 个废案标记全部失效,
    #    `_废弃77_活动入口` 之类会重新吐给 flow(进错活动那个坑)。
    #     训练期间用 `_classes_next.txt` 建数据集, 等新权重上线再把它
    #      改名成 `_classes.txt`。**别为了省事直接改线上那张表。**
    ap.add_argument("--master", default=None,
                    help="类表文件名(默认 _classes.txt; 加新类时用 _classes_next.txt)")
    args = ap.parse_args()

    master_path = (RAW / args.master) if args.master else MASTER
    if not master_path.is_file():
        print(f"[!] master 类表不存在: {master_path}")
        return 2
    print(f"[schema] 用类表 {master_path.name}")
    master = master_path.read_text(encoding="utf-8").splitlines()
    nc = len(master)
    print(f"[schema] master = {nc} classes (last: {master[-1]!r})")

    # Verify each source's classes.txt is a PREFIX of master (no reorder drift).
    for s in REAL_SOURCES + SYNTH_SOURCES + VAL_SOURCES:
        cf = RAW / s / "classes.txt"
        if not cf.exists():
            continue
        sc = cf.read_text(encoding="utf-8").splitlines()
        if sc != master[:len(sc)]:
            for i in range(min(len(sc), nc)):
                if sc[i] != master[i]:
                    print(f"[!] SCHEMA DRIFT in {s}: idx {i} = {sc[i]!r} vs master {master[i]!r}")
                    return 1
    print("[schema] all sources prefix-match master ")

    if args.clean and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    img_tr = OUT_ROOT / "images" / "train"
    lbl_tr = OUT_ROOT / "labels" / "train"
    img_va = OUT_ROOT / "images" / "val"
    lbl_va = OUT_ROOT / "labels" / "val"
    for d in (img_tr, lbl_tr, img_va, lbl_va):
        d.mkdir(parents=True, exist_ok=True)

    #  gather REAL frames, dedup by content md5
    seen_md5 = {}
    uniques = []   # (src, stem, jpg, cleaned_txt, classes)
    dropped_overflow = 0
    n_raw = 0
    for s in REAL_SOURCES:
        sd = RAW / s
        if not sd.is_dir():
            print(f"[!] missing real source {sd}")
            continue
        for jpg in frames_in(sd):
            txt = sd / (jpg.stem + ".txt")
            if not txt.exists():
                continue
            n_raw += 1
            h = md5(jpg)
            if h in seen_md5:
                continue
            seen_md5[h] = True
            cleaned = sanitize_label_text(txt.read_text(encoding="utf-8"))
            # drop any out-of-range cls (>= nc)
            keep = _keep_ui_lines(cleaned, nc)
            dropped_overflow += len(cleaned.splitlines()) - len(keep)
            cleaned = ("\n".join(keep) + "\n") if keep else ""
            uniques.append((s, jpg.stem, jpg, cleaned, label_classes(cleaned)))
    print(f"[real] {n_raw} raw frames  {len(uniques)} unique (dedup removed "
          f"{n_raw - len(uniques)} dup copies); dropped {dropped_overflow} out-of-range boxes")

    #  synth frames (no dedup)
    synth = []
    for s in SYNTH_SOURCES:
        sd = RAW / s
        if not sd.is_dir():
            continue
        for jpg in frames_in(sd):
            txt = sd / (jpg.stem + ".txt")
            if not txt.exists():
                continue
            cleaned = sanitize_label_text(txt.read_text(encoding="utf-8"))
            keep = _keep_ui_lines(cleaned, nc)
            cleaned = ("\n".join(keep) + "\n") if keep else ""
            synth.append((s, jpg.stem, jpg, cleaned, label_classes(cleaned)))
    print(f"[synth] {len(synth)} frames")

    base = uniques + synth   # one entry each (the de-duped real + synth)

    #  moderate oversample: bring 8..TARGET classes up to TARGET
    cls_to_frames = defaultdict(list)
    for i, (_, _, _, _, cs) in enumerate(base):
        for c in cs:
            cls_to_frames[c].append(i)
    dup_entries = []   # indices into base, duplicated
    for c, idxs in cls_to_frames.items():
        n = len(idxs)
        tgt = TARGET_OVERRIDE.get(c, TARGET)
        if n < MIN_UNIQUE or n >= tgt:
            continue
        need = tgt - n
        for k in range(need):
            dup_entries.append(idxs[k % n])
    print(f"[oversample] +{len(dup_entries)} dup entries (target {TARGET}, "
          f"450{TARGET_OVERRIDE[450]}, min_unique {MIN_UNIQUE})")

    #  write train (uniques + synth once, then dups with __dN suffix)
    written_tr = set()   # dst stems this build produced — stray purge below
    def write_entry(entry, dst_stem):
        s, stem, jpg, cleaned, _ = entry
        # Source may vanish between scan and write (live dashboard labeling
        # session deletes frames while a build runs — 2026-06-10). Skip, don't die.
        if not jpg.exists():
            print(f"[skip] source vanished mid-build: {jpg.name}")
            return
        link_image(jpg, img_tr / f"{dst_stem}.jpg")
        (lbl_tr / f"{dst_stem}.txt").write_text(cleaned, encoding="utf-8")
        written_tr.add(dst_stem)

    # 2026-08-04 空标签帧一律不进训练集。
    # 旧行为: **无条件写入**, `neg` 只是事后计数器 —— 日志那句 "N negatives"
    # 读起来像"我加了 N 张负样本"(feature), 实际是"有 N 张标签是空的"(bug),
    # 措辞误导正是它长期没被发现的原因。
    # 实测(移除 run_20260518_163513 后剩 170 张): **164 张是「特殊作戰」页漏标** ——
    # 屏上明明有 返回键·齿轮·回大厅·艦橋/機庫/大廳 tab·ALERT·「距離挑戰獲得結束」
    # (正是 cls474)·底部劇情/倉庫/商店, 标签却全空  反复教模型"这些 UI 不存在"。
    # 只有极少数(如纯色加载屏)才是合法负样本。
    # 权衡: 漏标的毒性 >> 显式负样本的价值(YOLO 每张图的背景区域天然就是负样本),
    # 且两者无法自动区分(要看图)  **全部跳过**, 宁可少几张合法负样本。
    neg = 0
    for entry in base:
        s, stem = entry[0], entry[1]
        if not entry[3].strip():
            neg += 1
            continue
        write_entry(entry, f"{s}__{stem}")
    dup_count = defaultdict(int)
    for bi in dup_entries:
        entry = base[bi]
        s, stem = entry[0], entry[1]
        dup_count[bi] += 1
        write_entry(entry, f"{s}__{stem}__d{dup_count[bi]}")
    n_train = len(base) - neg + len(dup_entries)
    print(f"[train] {n_train} frames ({len(base) - neg} unique+synth, "
          f"{len(dup_entries)} dups, 跳过空标签 {neg})")

    #  val from VAL_SOURCES (held-out 多域: ui+头像+摸头, 跨多个 run)
    # 2026-08-03 补 dedup: 这里**一直没有去重**, 而 REAL 侧有 —— 后果实测:
    #    val 2642 帧里只有 2200 张唯一图, **442 帧(16.7%)是字节完全相同的重复**,
    #      最大一组 **96 张一模一样**(_val_v12flywheel_0616 同一 run 连拍)  mAP 被
    #      重复帧加权, 那个场景的成绩等于乘了 96 倍, 量尺直接失真;
    #    还有 11 张 val 图与 train 里的图**逐像素相同** = 拿训练样本当考题。
    # 两件事一起治: val 内部按 md5 去重, 且与 REAL 侧 md5 撞上的直接剔出 val。
    written_va = set()
    n_val = n_val_dup = n_val_leak = n_val_empty = 0
    seen_val_md5: dict = {}
    for vsrc in VAL_SOURCES:
        vd = RAW / vsrc
        for jpg in frames_in(vd):
            txt = vd / (jpg.stem + ".txt")
            if not txt.exists():
                continue
            vh = md5(jpg)
            if vh in seen_md5:          # 与 train 同图  泄漏, 不许进 val
                n_val_leak += 1
                continue
            if vh in seen_val_md5:      # val 内部重复  只留第一张
                n_val_dup += 1
                continue
            seen_val_md5[vh] = True
            cleaned = sanitize_label_text(txt.read_text(encoding="utf-8"))
            keep = _keep_ui_lines(cleaned, nc)
            if not keep:
                # 与 train 侧同理: 空标签当考题 = "这张图什么都没有", 模型正确
                # 检出反而算 FP 白扣分。实测剩 3 张, 全部跳过。
                n_val_empty += 1
                continue
            cleaned = "\n".join(keep) + "\n"
            stem = f"{vsrc}__{jpg.stem}"   # source-prefix  防跨 run 同名(frame_000000)互相覆盖
            link_image(jpg, img_va / f"{stem}.jpg")
            (lbl_va / f"{stem}.txt").write_text(cleaned, encoding="utf-8")
            written_va.add(stem)
            n_val += 1
    print(f"[val] {n_val} frames from {len(VAL_SOURCES)} sources: {VAL_SOURCES}"
          f"  (dedup 去掉 val 内重复 {n_val_dup} + 与 train 同图 {n_val_leak}"
          f" + 空标签 {n_val_empty})")

    #  hygiene: purge strays + caches (2026-06-11 实锤双病根)
    # Build is incremental (no rmtree without --clean), so entries dropped from
    # sources (deleted frames / removed pools / renamed stems) linger as orphan
    # jpg+txt — ultralytics globs the dir, so STRAYS GET TRAINED with stale
    # labels (131 found tonight). And cache='disk' .npy never get reclaimed
    # (190.9GB of v8-era cache found). Purge anything this build didn't write.
    n_stray = n_npy = 0
    for d, keep in ((img_tr, written_tr), (lbl_tr, written_tr),
                    (img_va, written_va), (lbl_va, written_va)):
        for f in d.iterdir():
            if f.suffix == ".npy":
                f.unlink(); n_npy += 1
            elif f.suffix in (".jpg", ".txt") and f.stem not in keep:
                f.unlink(); n_stray += 1
    for c in OUT_ROOT.rglob("*.cache"):   # ultralytics scan caches — stale lists
        c.unlink()
    print(f"[hygiene] purged {n_stray} stray files + {n_npy} npy caches")

    #  data.yaml
    yaml = [f"path: {OUT_ROOT.as_posix()}", "train: images/train", "val: images/val",
            f"nc: {nc}", "names:"]
    for i, n in enumerate(master):
        yaml.append(f"  {i}: '{n.replace(chr(39), chr(92) + chr(39))}'")
    (OUT_ROOT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    # 图片落地方式体检: copy 不为 0 就说明有图在白占磁盘(见 link_image 注释)
    _ls = dict(_LINK_STAT)
    print(f"[link] {_ls}" + ("   有真拷贝, 检查是否跨卷" if _ls.get("copy") else ""))
    print(f"[done] {OUT_ROOT}  (nc={nc}, train={n_train}, val={n_val})")

    #  report key class counts (instances in train)
    inst = defaultdict(int)
    for entry in base:
        for ln in entry[3].splitlines():
            if ln.strip():
                inst[int(ln.split()[0])] += 1
    for bi in dup_entries:
        for ln in base[bi][3].splitlines():
            if ln.strip():
                inst[int(ln.split()[0])] += 1
    print("[key cls instances in train]")
    for c in (92, 450, 54, 403, 55, 441, 442, 449):
        print(f"  {c} {master[c][:22]:<22} {inst.get(c,0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
