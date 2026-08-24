# routing_v2 — 日常路由重写

> 起因（用户 2026-08-08）：「这个每日流程一直都没修复好过」「一看就是狗屎山代码
> 开始左右脑互博」「我建议是单独开一个 folder 重新写路由了」。
>
> 本目录 = 全新路由层。老代码（`brain/skills/`）**不动、不删**，作为参照与回滚。
> 删除条件与逐项审计见 **[MIGRATION.md](MIGRATION.md)**。

---

## 现在能跑什么（2026-08-15）

连续多关 + 混 Normal/Hard 已实现（`campaign.stages`，空则退回单关 `stage`），**未本轮 live**。3-2 仍按 profile 单关在跑（绑格+相位时钟都 live 过：r1 脚下走 right-up；r2 位移证据后进 r3，没有 150 tick 重发谎报没走动；卡 r3 右邻 BOSS 无格）。用户否掉拿 BOSS/道具当落点的硬搞，无格就 wait，等 v18 补格子本体（迷雾/可走已并进 497）。任务还挂图上，不要 `run` 归位，不要另开一条 campaign。

```bash
py -X utf8 -m routing_v2 probe                 # 看一帧：页面身份 + 检出 + 锁定框图
py -X utf8 -m routing_v2 step --flow event     # 单步：该点什么；加 --go 真点并验到达
py -X utf8 -m routing_v2 run --flows mail      # 跑（默认逐帧门控，每发要人放行）
py -X utf8 -m routing_v2 run --flows mail --auto
py -X utf8 -m routing_v2 config --schema       # 配置字段和写死的选项
py -X utf8 -m routing_v2 health                # cls 健康度 / 死判据
py -X utf8 -m routing_v2.tests.test_offline    # 完整离线回归，不碰设备
```

**验收状态**

| 层 | 状态 |
|---|---|
| 完整离线回归 | 通过：页面身份、推进闸、金钱闸、账号分桶、锁入口、奖励出口和六槽编队判据 |
| 连续推关 | `campaign.stages` 可跳号、可 Normal+Hard 混；空则退回单关 `stage`。离线已过，本轮未 live |
| 感知层 | UI v17，nc=528，36 个废案已屏蔽。scrcpy 仍有 PPS 断流噪声，watchdog 可恢复但未结案 |
| 08-15 live | reward 归位、cafe 锁钮、alt 分桶已过；锁着的 bounty/jfd/arena 分别 2.6/4.7/6.8s SKIPPED；Normal 3-1 CLEAN 207s/28tap，青辉石 220->250（首通+30），AP 768->758。Normal 3-2 没通：绑格+相位时钟 live 过；卡 r3 右邻无格。用户否掉 BOSS/道具硬点（点 (0.73,0.58) 落缝人没动已撤回）。无格=wait，补标 v18_grid_miss_20260815（3 张干净帧），不开训。石 250，AP 745，skip_rounds=2。任务还挂图上，不要 `run` 归位 |
| 六槽编队闸 | 5 张满编和 4 张空/部分/淡入实帧离线通过；活动 guide hub 无 formation。3-1 部署页出现过出击并二次点出，但 campaign 自点出击，没走六槽闸 |
| 训练数据缺口 | 右侧四 tile 玩法名 0/0 坐实; 大厅白锁标在 cls50 进了 v17 但 live 0 检出。锁态只标锁, 不要在锁着的 tile/钮上标玩法名或换厅 34/27 |

**写这一版当天，新架构自己又逮到 6 个真 bug**（都已修，且都进了离线回归）：
`wm size` 报竖屏导致 tap 飞出屏幕 / 大厅买石广告位被当购买框每轮误停 /
JIT 复验和"落点故意偏移"互相打架 / 连发闸被框抖动绕过 /
计数器数的是意图不是事实 / 对话框和底页抢"我在哪一页"导致抖动空转。
—— 这正说明**离线回归 + 到达验证**这两道东西值得先建。

---

# 第一部分 · 精细 defect compact（一天 live 实测出来的全部毛病）

每条都有帧证据或实测数字，不是推测。**重写时逐条对照，别再犯。**

## A. 架构级病根（这些才是"一直没修好"的原因）

### A1 禁禁 兜底默认值是破坏性动作
```python
x = F(bs, ["弹窗叉叉", "回大厅按钮", "返回键"], 0.5)
if x is not None: return "STRAY", x       # 兜底 -> 点回大厅
```
`回大厅按钮` **几乎每一页都在**。任何一帧主 cls 没检出（进战斗的转场帧最典型），
兜底就触发 -> **把自己点回大厅**。用户现场看到："刚要进战斗，bot 又手贱去点返回
大厅的房子按钮"。
**铁律：兜底默认必须是 no-op。任何破坏性动作（返回/回大厅/ESC/关窗）都必须有
明确的、专属该状态的 cls 支撑。**

### A2 禁禁 判据散落在各 skill，彼此不知道 -> 同一个 bug 修 N 次
今天一天在老代码里打了 7 个补丁，6 个不同根因，但形态完全一致。
`if page is not None: return action_back(...)` 这一句在 arena/cafe/schedule/
ticket_sweep **4 处复制粘贴**，而"剧情过场不吃 ESC"这件事**一处都不知道**。
**铁律：状态判定与动作决策只能有一个地方定义。**

### A3 禁禁 单帧当真相（今天出现 4 次）
| 现场 | 单帧误判成 | 真相 |
|---|---|---|
| cafe 进厅转场帧 | 「UI 被隐藏」-> 盲点写死坐标压在「編輯模式」上 | 只是 UI chrome 还没渲染 |
| 点完入场键的过渡帧 | 「无可打关，收工」 | 弹窗盖住列表 |
| 我的"到达验证" | 「已到达」 | 只是轮播翻页 |
| 活动 verify | 「上期领奖页」 | 当期活动，只是刚开只解锁 1 关 |
**铁律：状态判定必须连续 N 帧一致才生效（转场帧是一瞬，真状态是持续的）。**

### A4 禁禁 用 ADB 抓帧 = 所有时序 bug 的放大器
实测（2026-08-08，4K）：
```
ADB screencap + decode   1588 ms   <- 占 98.6%
YOLO 推理(ui)               23 ms
ADB tap                     32 ms
scrcpy latest()              0 ms  <- 流已在跑，取帧零成本
```
用 ADB 抓帧 -> 每次决策 1.6s；而活动入口轮播周期实测 **3.00s**（页间还有 0.08s
空白过渡）=> 必然点在翻过去的那一页上。
**铁律：帧源只用 scrcpy；ADB 只用于 tap。禁止在热路径 ADB 抓帧。**

### A5 禁禁 硬编码 wait 代替状态判定
老代码与我第一版脚本里全是 `sleep(2)/sleep(4)/sleep(6)`。
用户原话：「点了入场 popout 识别到了再点下一步，**通过识别到 cls 而不是硬给
wait time**」「**点击也要对齐 fps**」。
**铁律：零 wait。只在"下一步的 cls 出现"时推进；状态不变 N 帧才判定丢发重试。**

### A6 禁禁 为模型缺陷写的兜底，会在模型修好后反噬
`craft` 的灰键闸写成 `if find_cls(开始制造) and _btn_is_grey(...)` ——
这个前提成立**只因为 v14 把灰键误检成亮态**。v15 检出 `开始制造灰色` =>
`find_cls(亮态)` 返回 None => **整道闸被跳过** => 落到"MAX 外推"盲点禁用按钮。
**这类断裂 grep 不出来，只能 live 撞见。**
**铁律：兜底必须标注它补偿的是哪个模型缺陷 + 缺陷消失后的行为，并写进测试。**

### A7 禁 主动关掉唯一的防线去和时序赛跑
`event_quest` 检出 405 后 `act["_atomic_no_gate"] = True` —— 注释自己写着
「检出405->tap 落屏延迟 >1.5s 就点到切页后的 474 卡（误入三连根因）」。
**知道失效模式，却选择赛跑而不是加闸。**
**铁律：已知失效模式必须加闸，不许用"够快就不会错"来赌。**

### A8 禁 写死坐标跨版面必错
`_POS_QUEST_TAB = (0.635, 0.151)` 是 2026-07-07 在**另一个活动的版面**上标的。
CODE:BOX 是 `[Story | Quest]` 两页签，0.635 正好落在 **Story** 上。
**铁律：坐标只能从当帧 cls 推导；确需固定位时必须绑定"版面指纹"并在不匹配时
fail-closed。**

### A9 禁 判据把"内容多"当页面成立的前提（新活动首日全线崩）
| 位置 | 旧判据 | 新活动实况 |
|---|---|---|
| `_on_quest_list` | 要 ≥2 个**已解锁**入场键 | 1 开 + 4 锁 |
| `_survey` | 无条件滑到底找尾关 | 滑走唯一能打的关 |
| `partial list` | 要齐 `_tail_quests`=3 行 | 只有 1 行 |
| `_on_popup` | 只认**扫荡面板** | 首通弹的是「章節資訊 + 進入章節」 |
**铁律：页面身份只取决于结构（有一列关卡行），不取决于有几关可打 / 打过没。**

### A10 禁 共识 ≠ 真值（金钱层险些酿成事故）
青辉石顶栏 OCR 裁切窗口用的是"网格共识"标定值 `(6.0, 1.20)` ——
27 帧实测 **20 对 / 7 错**，把 `18,036` 读成 `18,003` => kill-switch 报
`MONEY BREACH` 急停整条 pipeline，而余额一分没少。
重标（人眼真值 18,036，27帧×42组参数）：`y_pad 0.40~0.50` 区间 **27/27 零错读**，
定 `(0.10, 5.5, 0.45)`，修后 9/9。
禁**危害双向：会读低就会读高，读高会掩盖真实掉钱。**
注意 信用点同法未修：真值 53,495,497，9 帧只对 3 次。
**铁律：任何数值判据必须挂人眼真值标定，禁止用"多参数共识"当真值。**

### A11 禁 `train=0` 不等于"漏训"，可能是废案
我先判「进错活动的根因 = cls77/78 没训练」，**错了**。
用户纠正：「我们的 cls 就叫做『距離結束還剩』和『距离奖励获得结束』，
加了 `_活动入口` 是废案」。真正在用的 405/474 **train 949/130，live 实帧检出 0.88**。
**铁律：判一个类"缺数据"之前，先确认它不是废案。**（废案改名加 `_废弃N_` 前缀，
detect 出口丢弃；禁**不能删行**，删了行号索引会让全库标注错位。
2026-08-15 又废 502/510：按答案走只点格子本体，不标 BOSS/敌方。
2026-08-16 又废 507：按答案走只点格子本体，不标道具。）

## B. 感知层缺口（重写时要一并补数据）

| 屏上信号 | 现状 | 影响 |
|---|---|---|
| 「Battle Complete」横幅 | `467 战斗完成` train 90 / **全仓零引用** | 判不了"打完没" |
| 战斗胜利 | `136` train 33 / val 0 —— **但 live 实测检得出**(12.6s 那帧) | 可用但样本薄 |
| **战斗失败** | `484` 在 **UI 模型 0 框**（只在 battle 域） | 禁**输了看不见** |
| 奖励卡上的 `Bonus` 粉标 | **无类** | 判不了"加成生效没" |
| `Best Record!` 徽章 | **无类** | 判不了"纪录刷没刷新"（而它**永久锁定**） |
| 活动关卡完成度 | `98 活动剧情关卡_已看`/`99 活动站斗关卡_已打` **均 0 框** | 判不了"这关打过没" |
| 关卡得星 | 只有 `_0` 和 `_3` 两态 | "打了但没满星"识别不出 |
| 编队页 | `1部队高亮`/`2部队`/`出击` 可用；模型没有 EMPTY/槽位占用类 | 出击前还要六槽固定名牌彩条正向证明；离线通过，live 待验 |

### B2. 硬坐标 = 感知债（08-20，进 v19）

全仓 `tap_at` 只有 3 处。出现硬坐标就是 cls 弱或框标错，过渡码留下，训回删。

| 坐标 | 缺的覆盖 | 删条件 |
|---|---|---|
| (0.944, 0.942) 任务大厅入口 | 类已有，裸大厅 live 约 0.08-0.15，门控 0.45。浮层压暗能到 0.85，禁止只喂那种 | 裸大厅稳帧 >=0.45 |
| (0.40, 0.55) 大厅藏 UI | 闲置收 chrome 后零框，靠 last_solid 猜。补 大厅藏UI | 能认出藏 UI 页 |
| (0.50, 0.50) 空屏 | blank=零检出。补 Now Loading / 过场黑 身份 | 加载/过场能正向认 |

同类债（挂框但大偏移）：剧情 tile 0 点黄点、主线框在四字、章节行无类 +0.11、活动 405 +0.075、momo 未读 -0.25。旧 NEED32 没有任务大厅入口，补训名单必须加上。

## C. 环境级坑（重写时必须内建处理）

- **MuMu adb 端口会漂**：实例重启 7555 -> 16384。唯一权威 = `MuMuManager info` 的 `adb_port`。
- **adbd 会卡死**：今天死 1 次。三级阶梯：`disconnect+kill-server+start-server+connect`
  ->（setprop 无效）-> `MuMuManager control -v 0 restart`（要手动 monkey 拉游戏）。
- **`dumpsys mCurrentFocus` 不可信**：报前台是 `app.lawnchair`（桌面），实际游戏好好在大厅。
  MuMu + Unity 全屏不注册 window focus。真存活 = 进程在 + **连续两帧 md5 变化** + YOLO 认出大厅。
- **scrcpy 流 17.0s 寿命**：MuMu 对每个镜像流有内在寿命，`ScrcpyFeed` 已内建轮换。
- **活动入口轮播周期 3.00s**（页间 0.08s 空白）。只在**刚翻成目标态那一瞬**点。
- **HUB 活动入口落点要 +0.075**：405/474 的框标的是**倒计时气泡**，可点的是下面卡片本体。

## D. 金钱铁律（新架构里必须是最底层，不可绕过）

- 花钱 = 点确认成交（买AP/买票/CAD）。购买青辉石/组合包页要进要标，只领免费；CAD 键程序不点。
- 读不出余额就不点花钱键。货架页本身不停机。
- 用户自己抽卡/升级造成的余额下降记「外部变动」，更新基线，不 HALT 整轮。
- bot 自己点了购买/刷新/确认付费之后青辉石下降 = MONEY BREACH，停整轮。
- 弹窗体内出现青辉石 = 购买框，拦下付费 tap（不放松 bot 点购买键）。
- `forbid_premium_currency` / `ap_purchase_limit=0` / `purchase_caps` 全 0。
- 每 tick 落 balance 台账；EXTERNAL 与 BREACH 分开，余额对不上不杀进程。

---

# 第二部分 · 新架构设计

## 1. 分层（每层只做一件事）

```
config/          用户配置（前端可选）-> 纯数据，无逻辑
  profile.json   一份配置 = 一次跑什么、怎么跑

percept/         感知层：帧 -> 结构化观测
  feed.py        scrcpy 帧源（0ms）+ 17s 轮换 + 存活探针
  detect.py      YOLO 封装（ui/battle/avatar 三模型路由）
  read.py        数字读取（顶栏货币/票数），**人眼真值标定表**
  observe.py     Observation 对象：boxes + page + balances + 帧龄/seq

state/           状态层：观测 -> 状态（**唯一定义处**）
  pages.py       页面身份 = cls 组合（结构判据，不看"有几个"）
  machine.py     状态机 + **连续 N 帧确认** + 兜底 no-op

act/             动作层
  tap.py         ADB tap（32ms）；落点必须来自当帧 cls
  gate.py        金钱闸 / 落地前复验 / 连发闸（集中一处，不散落）

flow/            流程层：每个玩法一个 flow，只描述"看到 X 就做 Y"
  daily.py  bounty.py  jfd.py  event.py  cafe.py  schedule.py
  craft.py  shop.py    mail.py mission.py  mining.py  momotalk.py

app/
  runner.py      主循环：feed -> detect -> state -> flow -> act
  api.py         前端 API（配置读写 / 启停 / 逐帧门控）
```

## 2. 主循环（零 wait，对齐 fps）

```python
while running:
    fr, age, seq = feed.latest()
    if seq == last_seq: continue          # 同一帧不重复决策 = 对齐 fps
    obs   = perceive(fr)                  # YOLO 23ms
    state = machine.update(obs)           # 连续 N 帧确认
    act   = flow.decide(state, obs, cfg)  # 没有对应 cls -> None（no-op）
    if act and gate.allow(act, obs):      # 金钱/复验/连发 集中判
        tap(act)
```
- **没有对应 cls 就返回 None**，绝不"兜底点点看"。
- 重发条件 = **状态连续 N 帧未变**，不是计时器。

## 3. 用户可配置项（前端按钮/选择框）

### 3.1 日常固定全开；推图 / 剧情 / MomoTalk 另开
日常顺序写死，前端不许拖、不许手关。去不去由票/锁/红点/台账判定。
`batch_sweep` / `special_sweep` 未实现，不出现在日常列表。
```jsonc
"modules": {
  "daily_routine": true,      // 社团 / 制造 / 商店
  "cafe":          true,
  "schedule":      true,
  "bounty":        true,
  "jfd":           true,
  "event":         true,
  "arena":         true,
  "mail":          true,
  "daily_mission": true,
  "story_mining":  false,     // 推图/其它，默认关
  "momotalk":      false,
  "campaign":      false,
  "batch_sweep":   false,     // 未实现
  "special_sweep": false
}
```

### 3.2 悬赏通缉（用户点名：前端选刷什么）
```jsonc
"bounty": {
  "branches": ["教室"],            // 词表写死: 教室 / 高架公路 / 沙漠铁道
  "difficulty": "highest",         // highest | fixed
  "fixed_stage": null,
  "use_tickets": "all",            // all | keep_n
  "keep_tickets": 0
}
```

### 3.3 学院交流会 JFD（用户点名：前端选刷什么）
```jsonc
"jfd": {
  "academies": ["千年", "三一", "格黑娜"],   // 顺序 = 优先级
  "difficulty": "highest",
  "use_tickets": "all",
  "keep_tickets": 0
}
```

### 3.4 剧情挖矿 / MomoTalk 好感度（用户点名）
```jsonc
"story_mining": {
  "enabled": false,
  "sources": ["羁绊剧情", "主线剧情", "活动剧情", "後日談"],
  "target_students": [],        // 空=全部; 有值=只挖这些人的羁绊
  "max_ap": 0,                  // 0=不限
  "skip_cutscene": true,
  "stop_on_battle": false       // 剧情里遇战斗: 打 or 跳过
},
"momotalk": {
  "enabled": false,
  "reply_policy": "affinity_first",  // affinity_first | first_option | random
  "target_students": [],
  "max_students_per_run": 0
}
```
章节图本屏都是「完成」时滑一次再扫 new，不干等。节点图无入场键回章节图，不整条收工。

### 3.5 活动
```jsonc
"event": {
  "clear_first_with_team": 1,      // *首通用部队1(用户规则)
  "bonus_team": 2,                 // *加成队 = 部队2
  "order": "clear_then_bonus",     // *先 Q1->Qn 通关, 再打加成(用户规则)
  "shop_plan_before_bonus": true,  // *先商店推算定缺哪种币, 再按币种编队
  "farm_stages": {"10":1, "11":2},
  "ap_reserve": 0,
  "shop": {"auto_buy": true, "currencies": [], "furniture": false,
           "skip_last_tab": true}  // tab3 盒抽币绝不自动买
}
```

### 3.6 咖啡厅 / 课程表 / 制造 / 商店 / 大赛
```jsonc
"cafe":     {"invite_targets": [], "skip_invite": false, "headpat": true, "floors": [1,2]},
"schedule": {"target_students": [], "relationship_first": true},
"craft":    {"use_acceleration_ticket": false, "phase_priority": ["光辉","花朵"], "quantity": "MAX"},
"shop":     {"common_priority": [], "tactical_priority": [], "refresh_times": 0,
             "credit_buy": false, "arena_shop": true},
"arena":    {"level_diff": 0, "stop_at_rank1": true, "max_refresh": 0}
```

### 3.7 全局安全 / 运行（禁 金钱项不可从前端放宽）
```jsonc
"safety": {
  "forbid_premium_currency": true,   // 禁 锁死, 前端只读
  "ap_purchase_limit": 0,            // 禁 锁死
  "purchase_caps": {"arena":0,"bounty":0,"scrimmage":0,"lesson":0},
  "money_step_needs_human": true     // 锁死，金钱步逐帧人审
},
"run": {
  "step_mode": true,                 // 逐帧门控
  "frame_source": "scrcpy",          // 禁 不提供 adb 选项
  "confirm_frames": 3,               // 状态连续 N 帧才认
  "retry_frames": 38,                // 状态不变 N 帧判丢发重试; 宽松契约地板 10 帧, 不跟本值等比缩小
  "max_minutes": 90
}
```

## 4. 前端形态
- 每个 module 一个开关（挖矿/MomoTalk 默认关，用户按钮开 —— 用户明确要的）
- 悬赏分支、JFD 学院、活动关卡：**进页时扫 cls 动态列出可选项**，不写死清单
- 金钱项只读展示，不给放宽入口

## 5. 迁移与验收
1. `routing_v2` 与老 `brain/skills` **并存**，前端切换；老代码不删（回滚用）
2. 每个 flow 上线前：**step_walk 逐帧走一遍 + 帧证据**，通过才允许自主跑
3. 每个 flow 必须给出**竣工判据**（CLEAN / LEFTOVER / UNKNOWN），不许"跑完了"就算完
4. 战斗层（`battle/`）后续单独重写 —— 用户已预告

---

# 附：今日已验证可直接复用的资产

- **状态机原型**：`scratchpad/loop.py` —— 真机跑通，10 次出击 / 3.6 分钟，
  全 cls 驱动零 wait。`classify()` 逐帧验证过。注意 它的 `STRAY` 兜底就是 A1 那个 bug，
  搬过去时**必须删掉**。
- **单步驱动器**：`scratchpad/step1.py` —— 看一步/走一步 + **真到达验证**
  （只认页面身份变化，不认状态串变化）。
- **实测常量**：轮播 3.00s / HUB 落点 +0.075 / scrcpy 0ms / YOLO 23ms / tap 32ms /
  青辉石 strip `(0.10, 5.5, 0.45)`。
- **今日修好并 live 验过的 5 处**（先移植过来，别重犯）：craft 灰态 cls 优先、
  cafe 连续帧确认、青辉石 strip 重标、interceptor 剧情逃生、活动首日三处放宽。

---

# 附二：新架构自己踩出来的坑（2026-08-08 当天，都已修+进回归）

写完当天做真机验证，新层自己又贡献了 6 个 bug。记在这里，因为每一个都是
**老代码里同一族病换个形态复发**：

### N1 禁`wm size` 报的是竖屏面板 -> tap 飞出屏幕
MuMu 在 display 0/2/3 上一律返回 `2160x3840`（竖屏），而游戏是横屏
`3840x2160`，`input tap` 用的也是横屏坐标空间。信 `wm size` => 归一化
`y=0.953` 换算成 **3662 > 2160**，整发点到屏幕外，表现成"点了没反应"。
（老代码把 `3840x2160` 写死反而碰巧是对的 —— 这就是写死坐标能活很久的原因：
它在唯一被测过的那个配置上是对的。）
**修** = `Device.calibrate(frame_w, frame_h)`：物理面板给两个边长，
**哪个当宽由 scrcpy 首帧的朝向决定**。-> `percept/device.py`

### N2 禁大厅的买石广告位被当成购买框，每轮第一帧就 HALT
我第一版金钱判据写的是"`购买青辉石` 在屏上 -> 停"。实测大厅左侧**常驻**一个
买石广告位，conf **0.95** 稳定在场。**信号 ≠ 场景**，和 §A9「把内容多当页面
前提」是同一族病。
**修** = 判据结构化：弹窗体内青辉石 / `购买青辉石`+同框对话框控件 /
双键框内步进器。组合包货架页不是成交框。-> `act/money.py`

### N3 禁JIT 复验和"落点故意偏移"互相打架
JIT 要求"落点仍在锚点框内"，而 HUB 活动入口的落点是**故意** +0.075 打到框外
的（框标的是倒计时气泡）。=> 每一发带偏移的点击都被自己的闸丢掉。
**离线回归当场逮到的**，没上真机。
**修** = 复验改判**锚点框有没有移动**，不判落点包含。-> `act/gate.py`

### N4 禁连发闸被框抖动绕过（一个大厅入口连点 12 次）
dedup 的 key 用 `(round(x,3), round(y,3), cls)`，而 `邮件箱` 的框每帧抖
±0.001（0.892<->0.893）=> 每帧都被当成**新目标**，闸形同虚设。
这是"连发族"的**第 4 种形态**：前三种是 after-ack 挂错地方 / 判定带宽不一致 /
没有 after-ack 的重发，这一种是**同一性判据太严**。
**修** = 按距离判同一性（0.02 归一化）。-> `act/gate.py`

### N5 禁计数器数的是意图不是事实
flow 在 `decide()` 里 `self.bump("claims")`，被闸吞掉也照样涨 => 日志里
"領取 #24/#25"而屏上只领了几次。**memory 里那条最贵的教训「日志是意图不是
事实」在新代码里原样复发。**
**修** = `Action.counter` 声明计数器名，**runner 只在真的 tap 出去之后才 +1**。

### N6 禁禁对话框和底页抢"我在哪一页" -> 抖动空转
邮件页上「一次領取黄色」和结算「確認键」交替被检出，两者当成两个**页面**竞争，
页面身份在 `mail <-> ack_dialog` 之间抖了几十次；flow 每次都拿到"刚换页"的状态、
闸每次都放行 => 同一个領取键点了 **65 下**。
**修** = 引入 **overlay** 概念：对话框/奖励框盖在底页上，**不参与页面竞争**，
单独连续 N 帧确认；派发顺序 = 覆盖层 -> 底页 -> offsite。
-> `state/pages.py` `state/machine.py` `flow/base.py`

**元教训**：这 6 个里有 3 个是**离线回归或到达验证**逮到的，另外 3 个是
`--auto` 跑一条最安全的 flow 逮到的。所以顺序应该永远是
**离线回归 -> 单步验到达 -> 拿最安全的 flow 自主跑 -> 才轮到真正干活的 flow**。
