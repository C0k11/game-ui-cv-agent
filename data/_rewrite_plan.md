# 全 skill 重写路线图 (post-probe 实现总纲)

> 2026-06-01。两天交互 probe 完成，9 份 probe log 已 commit (`f47ba14`)。
> 本 doc = 总纲；每个 skill 的**点击 spec 看对应 `data/_*_probe_log.md`**。
> compact 后直接照此 + 各 MD 实现。

## 目标
删旧 OCR 时代 skill + 据 probe MD 写**全新纯-YOLO「状态→点击」skill**。
模板 = 已重写干净的 `brain/skills/cafe.py` + `schedule.py`（照它们的风格）。

## 文件处理
- ✅ 保留：`base.py`(BaseSkill+action_click/click_cls/find_cls/action_*) · `ui_classes.py`(cls名) ·
  `pipeline.py`(截图/YOLO/OCR/interceptor/trajectory 基础设施) · `cafe.py` · `schedule.py`(模板)
- ♻️ 重写(照 MD)：`club.py`(社交签到→_social) · `craft.py`(快速製造→_craft) · `shop.py`(信用点动态买→_shop) ·
  `mail.py`(一次領取→_mail) · `momo_talk.py`(→_mining MomoTalk段) · `story_mining.py`(→_mining 剧情段) ·
  `bounty.py`(→_missions) · `arena.py`(→_missions arena段)
- ➕ 新建：`buy_pyroxene.py`(→_buy_pyroxene) · `jfd.py`(学院交流会→_missions JFD段) ·
  `daily_mission.py`(每日任务领奖→_daily_reward)
- 🗑 删除：`ap_planning.py` `daily_tasks.py`(刷体力未就绪) · `event_activity.py` `event_progress.py`(活动,跳过) ·
  `pass_reward.py` `story_progress.py` · `campaign_sweep.py`(AP-farming 未就绪；如 bounty/arena 改独立入口则删)

## pipeline.py 集成点
- `_skill_registry` (≈L853)：增删 skill 实例
- `SKILL_YOLO_MAP` (≈L242)：每 skill 检测器装载（多数 base；arena/bounty/jfd +battle；cafe +cafe+avatar；schedule +avatar；momotalk/story 仅 base）
- `DailyRoutineSkill` 调度表 (≈L877 注释)：日常采集 sub-skill 列表（更新为新 skill）
- `DEFAULT_SKILLS` / daily 顺序
- `_SKILL_BADGE_MAP` (≈L995)：should_run 红/黄点 gate

## daily 顺序（用户定）
购买青辉石(免费组合包,**第1**) → 社交(社團签到) → 制造(快速製造启动) → 商店(信用点动态买) →
咖啡厅 → 课程表 → bounty → JFD → arena → MomoTalk挖矿 → 剧情挖矿 → mail → **每日任务领奖(最后,解锁前置)**

## ★ 全局硬规则（贯穿所有 skill，probe 验证）
1. **绝不花青辉石/真钱**：任何"购买/确认/出击"前，确认框币种必须是 信用点/票券/AP/免费 —— 见"青辉石/CAD"立即取消。票=0 不进战斗(防买票框)。
2. **OCR 只识别数字**（票数 X/Y、信用点余额、体力）；导航纯 YOLO cls。`run_digit_ocr(frame, region)`+`parse_count`。
3. **不硬编码像素坐标**：全归一化(0-1)；under-trained cls 才用归一化 GAP 兜底(documented)。
4. **tap 验证+重试**：MuMu ADB 偶发丢点 → 关键 tap「点→验证屏变→没变重试」。
5. **瞬时 cls 用连续轮询**：如 `学生发送信息中`(438) 只闪一瞬 → WGC 高频轮询，别单帧定论。
6. **GOT_REWARD/获得奖励 dismiss**：点 `点击继续字样`(142) 或 `获得奖励` 头部框，**别点屏幕中心**(压物品图标)。
7. **掃蕩完成/reward 弹窗 re-detect**：扫荡确认后有动画过渡，轮询等 `掃蕩完成`/确认键 出现再 dismiss。

## 各 skill 关键流程速查（详见 MD）
- **buy_pyroxene**：大厅`购买青辉石`(retry)→商店→`组合包未选择`tab→找`免费`cls→点其正下`购买`→确认框(验`免费`)→`确认键`→`获得奖励`(点头部框)→红点清=done。⛔左右CAD真钱包
- **club(社交)**：大厅`社交入口`→`社團`卡(点卡身~0.235,0.52非红点,浮层点空白会收起)→自动弹簽到→`确认键`→done(10AP进信箱)
- **craft**：大厅`制造入口`→`一次领取黄色`(可领则领,`灰`跳过)→`快速制造`→`MAX_可点击`(在则点)→`开始制造`→确认框→done。⛔`立即完成`(耗製造券)；`一次領取`若弹"立即完成"带券/石→取消
- **shop**：大厅`商店入口`→一般tab→`全部選擇`→`选择购买`→确认框【digit-OCR 持有&總價】持有≥總價+buffer则`确认键`否则`取消键`→`获得奖励`→done。⛔`青輝石`tab
- **bounty**：任務hub→`悬赏通缉`(dot-gate)→OCR票数(0则exit)→分支(dashboard `bounty_branches`)→stage list(`关卡得星_3`+`入场键`bbox配对,滑到底选最高)→任務資訊→`MAX_可点击`则MAX→`扫荡开始`→确认框验"懸賞通緝票券"非青辉石→`确认键`→`掃蕩完成`(re-detect)`确认键`→票0 done
- **jfd(学院交流会)**：同bounty结构。分支(dashboard `jfd_academy`:三一/格黑娜/千年)。**多耗 AP**(~15/次)→digit-OCR体力够不够。确认框"學園交流會票券+AP"非青辉石
- **arena**：任務hub→`战术大赛`(dot)→点掉所有`领取奖励_黄`(获得奖励→点击继续)→选最上`战术大赛对战选择区域`(cls92)→`攻击编制`→`出击`→**自动战斗**→`戰鬥結果`确认键→(可能`達成賽季最高紀錄`确认键,白嫖青辉石)→回arena。**每场~25s `等待時間`冷却**(轮询归零)。票0 done
- **mail**：大厅`邮件箱`(红点)→`一次领取黄色`→`获得奖励`(点头部框)→`一次领取灰色`=done
- **daily_mission(每日领奖)**：大厅左`每日领奖`(0.045,0.358≠任务大厅hub)→任務(全體tab)→循环`全部领取_黄`→reward(点击继续)直到灰→再扫单个`领取_黄`(meta任务)→done。**最后跑**(其他日常完才解锁)。⛔`立即前往`未完成任务不碰
- **momo_talk(挖矿)**：大厅`MomoTalk`→`momotalk学生聊天区域按钮`(对话tab)→while有`学生momotalk信息未读`:点最上学生(点行左侧头像~0.22,非badge)→while(`学生发送信息中`|`学生信息回复选项`|`前往羁绊剧情`):发送中→等;回复选项→点;前往羁绊剧情→点→`进入羁绊剧情`→剧情场景→跳过(见下)→稳定无三者=该生done→下一个
- **story_mining(剧情挖矿)**：任務hub→`剧情`→分类(主线/短篇/支线,各黄点;重播不挖)。主线:卷左滑找`new`篇(vs`完成`)→点选中→章节列表找`黄点`章(黄点y定位点行)→节点列表找`剧情图标未完成`+`new`+`入场键`(非`入场键没解锁`)→`入场键`→`进入章节`(139)→剧情场景。短篇/支线:grid找`new`卡(没有`右切换`翻页)→进→同节点逻辑
- **剧情跳过(共用)**：剧情场景**自动播放**→尽快`剧情menu`(431,0.94,0.05)→`跳过故事键`(141,0.945,0.164)→`确认键`→`获得奖励`(≈80青辉石/话!)→`点击继续字样`→`中断`(剧情中断退出433)退节点/或自动续播则重复→节点`剧情图标已完成`=done

## 待补 cls（v6 重训，见 `_training_gaps_from_probe.md`）
活動進行中 · 特殊任务委託tiles · 社交好友/幫手卡 · 制造空槽/剩餘時間/立即完成 · arena等待時間 ·
学生发送信息中(438欠训28帧,瞬时) · 社團card(recall) · 立即前往(daily-mission)。skill 先按现有 cls 写，缺的用 GAP/坐标兜底 + 标注待 v6。

## 执行顺序建议
1. 先 buy_pyroxene(最简,建立模式) → 2. mail → 3. craft → 4. shop(动态OCR) → 5. club →
6. bounty → 7. jfd → 8. arena(PVP+冷却) → 9. daily_mission → 10. momo_talk → 11. story_mining
每个：写新→`py_compile`→接 pipeline registry/SKILL_YOLO_MAP→(用户 probe 验证)→删对应旧文件。
最后统一清 pipeline 死引用 + DEFAULT_SKILLS + daily 顺序 + 删 5 个 obsolete 文件。
