# 飞轮素材人审清单 (2026-07-17 整备)

> 整备动作(2026-07-17 执行完毕): ①dHash 贪心稀疏(Hamming≤6, 3,101→988 帧, **2,113 冗余帧** → `data/_dup_backup_20260717/` 可恢复)
> ②registry active 模型预标(ui v13 / battle v9 / fused_avatar v6, `yolo_prefill_run.py` merge 模式, span 路由): **988 帧 9,812 框**
> ③烧录抽查: 顶条 HUD 目检干净(ADB/scrcpy 源) ④UI 常规池预标空帧 20 张 → `_unlabeled_backup/_20260717_prefill_empty/`
> ⑤**双重毒清除**: defeat_candidates_v10 的 28 帧是 botplay 池副本 — botplay 侧同帧(无 cls484 标注=矛盾标签)已移
> `data/_defeat_dedup_backup_20260717/`; defeat 池 battle merge 复验 = 仍纯 484×28, 人审框未动。

## 训练材料路由(谁喂谁 — build 时 box 级过滤, 不用物理分池)

| 池组 | 去向 | 说明 |
|---|---|---|
| `defeat_candidates_v10` (28帧/28框 cls484 战斗失败, **已人审✓**) | **battle v10** | 用户已标, 无需再审; v10 build SRCS 加此池, nc 18→19 |
| `run_20260715_*_botplay_clean` ×7 (526帧) | **battle v10**(HUD/我方/敌方/胜利/败) + **fused_avatar**(手牌头像 143-394) + ui(结算/编队 chrome) | combat 2.0 实战帧: DEFEAT 页/瞄准态/技能卡全在这批 |
| `flywheel_event_shop_20260717` (77帧, 未稀疏保全) | **ui v14** | 购买按钮/价格条/貨幣tab = v13 盲区解药(screen_flow 实锤③), 预标必稀疏, **人审逐帧补框** |
| `run_2026070x/1x_*_clean` UI 池 ×15 (稀疏后 385帧) | **ui v14** | 日常场景补强(活动页/商店/大赛/悬赏/课程表/咖啡厅) |

## ⛔人审否决项(用户 2026-07-17 定)

- **Challenge tab/行上出现任何语义框一律删**: Challenge=抄轴/手打专属, 词表不给 Challenge 建"可打"类。
  已清 3 帧错框(v13 把活动 Challenge 高亮黄字误检成 80「普通关卡选中」— 该类本义是主线 Normal tab 指示器):
  `run_20260708_event_probe_clean/frame_000002` + `run_20260709_merged_clean/120806__frame_000371` + `123518__frame_000017`。
  这三帧保留为负样本(教 v14: Challenge 高亮≠80)。主线任务页 Normal tab 的 80 正标(`run_20260715_000407_clean/frame_000016`)保留勿删。
  Challenge 行的「入场键」「关卡得星_0」是纯视觉类, 保留无害(bot 防线在 sweep/rerun 的 关卡得星_3 精确正锚)。

## 人审入口(dashboard 标注页) — 2026-07-17 已合并为两组

**①`run_20260717_merged_ui_clean`(442 帧 5,471 框)— 全部 UI 材料一个池**(16 源池合并, 帧名前缀=源池溯源:
evshop=活动商店 / 0709m=7月9日大池 / 0717a=今天daily走线 / 其余日期简写)。审点:
- `evshop__*` 77帧: **cls103 购买按钮已 OCR 辅助标满(+537框, 文本精确匹配, 抽查全准)**; 剩人工 **貨幣tab(101/102)/余额条(104)** + 抽查
- `0717a__*` 53帧: 任务 hub 帧上人工画 **idx77 当期活动入口** 框(v13 未训新类)
- 其余快速过目纠错(预标底子好)
- ⛔Challenge 上任何语义框一律删(见否决项)

**②战斗材料独立池(battle v10 SRCS 按池加, 不合并)**:
- `defeat_candidates_v10`(28帧 cls484, 已人审✓ 免审)
- `run_20260715_*_botplay_clean` ×7(498帧): DEFEAT 口径对齐/瞄准态/技能卡抽查/战斗胜利帧留意 val

## 惯例检查结论(本次已执行)

- ✅ 冗余: 3,101→988 帧(77% 近邻重复移备份; event_shop/botplay 保全未稀疏 — 弱类/战斗动态)
- ✅ 烧录: 顶条 HUD 抽查全干净(ADB/scrcpy 源, overlay 物理不可见)
- ✅ 空帧: 预标后 0-box 帧已移 `data/_unlabeled_backup`(event_shop/botplay 除外 — 弱区空帧=人工补框对象, 保留)
- ✅ 格式: label 全 5 列 master 索引(yolo_prefill_run 保证), classes.txt=master 副本
- ⚠ val 纪律: 这批全部是连续 live 录制 → **审完只进 train, 不切 val**(近邻泄漏铁律); 新 val 仍需独立走查 session

## 405/474 混淆审计(2026-07-17, 用户抓误标后全量核查)

- 两池(merged_ui + 0711_merged)全部 405/474 框 OCR 复核(判定键=「獲得」474专属 vs「還剩」405专属 —
  ⚠ba_rec 系统性吞「獎勵」二字, 第一版"无獎勵判405"规则差点反向改错 73 框, dry-run 拦下):
  **ok405=184 / ok474=75 / 修正 2 框**(405→474: `0715c__frame_000004` + `0717a__frame_000006`,
  hub 左上「距離獎勵獲得結束」上期领奖 banner 被标当期入口 — 不修 v14 会把领奖倒计时学成活动入口, 7/15 进错活动坑复活)。
- 人工 1 帧: `run_20260711_merged_clean/144712_frame_000002` cls405 框内 OCR 零字(目检定 405/474)。

## 本次迭代训练源(用户 2026-07-17 定稿)

**ui v14 = 现有 REAL_SOURCES + `run_20260717_merged_ui_clean`(442) + `run_20260711_merged_clean`(1,624)**(两池人审完就 build);
battle v10 = v9 九池 + defeat_candidates_v10 + botplay×7(battle 域框)。

## 训练顺位建议(审完后)

1. **ui v14**: build_ui_v2 REAL_SOURCES += 本批 UI 池+event_shop → 修商店盲区+idx77+购买按钮/价格条
2. **battle v10**: build_battle_v10 SRCS = v9 九池 + defeat_candidates_v10 + botplay×7(battle 域框) → nc19 补 DEFEAT cls, 根治"判败靠兜底"
3. fused_avatar 顺带吃 botplay 手牌头像框(box 级路由自动)
