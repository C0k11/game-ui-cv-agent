# 迁移与删除审计 —— 老代码什么时候可以删

> 用户 2026-08-08：「干好了也就是说可以把除开训练材料以及重要的素材资产，
> 可以把老的狗屎山代码删除了（记得可利用的都得搬完）」

**结论先说：现在还不能删。** 新层写完了、离线 40 项全过、真机跑通了
`mail` 全流程，但**其余 12 条 flow 一条都还没在真机上走过**。
下面是逐项审计 + 删除的准入条件。

---

## 一、已搬过来的（老代码里的真金）

| 老位置 | 新位置 | 备注 |
|---|---|---|
| `brain/scrcpy_feed.py` | `percept/feed.py` | 17.0s 流寿命 + 预热轮换 + 断流/静止判据，逐行搬 |
| `brain/mumu_port.py` | `percept/device.py` | 端口漂移解析 + UTF-8 坑 |
| `brain/pipeline.py` 的 `icon_strip` / strip 标定表 / `run_digit_ocr` / `parse_count` | `percept/read.py` | **三条几何铁律 + 人眼真值标定值**原样保留 |
| `brain/pipeline.py` 的 `_read_topbar_clean` 投票 | `percept/read.py::Vote` | ≥2 票才信 / 3 票早退 |
| `brain/skills/ui_classes.py` | `state/vocab.py` | 加了**训练量健康度**和 `require()` 死判据闸 |
| `brain/screens.py` 页面签名 | `state/pages.py` | 结构判据重写，去掉全部"内容多才算数"的前提 |
| `brain/pipeline.py` 的 JIT 复验 / 金钱 kill-switch | `act/gate.py` + `act/money.py` + `act/ledger.py` | 判据集中一份 |
| `scratchpad/loop.py` 的状态机 | `state/pages.py` + `flow/battle.py` | *`STRAY` 兜底**已按约定删掉** |
| `scratchpad/step1.py` 的真到达验证 | `app/cli.py::cmd_step` | 只认页面身份变化 |
| 13 个 skill 的流程知识 | `flow/*.py` | 见下表 |

### flow 覆盖对照

| 老 skill | 新 flow | 状态 |
|---|---|---|
| `daily_routine.py` `club.py` | `flow/facilities.py::ClubFlow` | 写完，**未 live** |
| `craft.py` | `facilities.py::CraftFlow` | 写完，**未 live**（灰态优先已修） |
| `shop.py` `buy_pyroxene.py` | `facilities.py::ShopFlow` | 写完，**未 live** |
| `mail.py` | `facilities.py::MailFlow` | OK **live 跑通 CLEAN** |
| `daily_mission.py` | `facilities.py::DailyMissionFlow` | 写完，**未 live** |
| `cafe.py` | `flow/cafe.py` | 写完，**未 live** |
| `schedule.py` | `flow/schedule.py` | 写完，**未 live** |
| `ticket_sweep.py` `bounty.py` | `flow/sweep.py` + `bounty.py` | 写完，**未 live** |
| `jfd.py` | `flow/jfd.py` | 写完，**未 live**；*已改用 cls 选学院 |
| `event_quest.py` | `flow/event.py` | 写完，**未 live** |
| `scripts/buy_event_shop.py` | `flow/event_shop.py` | 写完，**未 live** |
| `arena.py` `arena_shop.py` | `flow/arena.py` | 写完（大赛商店买体力未做） |
| `story_mining.py` | `flow/mining.py` | 写完，**未 live** |
| `momo_talk.py` | `flow/momotalk.py` | 写完，**未 live** |
| `batch_sweep.py` `special_sweep.py` | — | **没搬**（默认关，活动期 AP 全给活动） |
| `combat_brain.py` | — | **没搬**，战斗层用户已预告要单独重写 |

---

## 二、禁绝对不能删

- `data/raw_images/` `data/**/*.jsonl` `dataset/` `runs/` —— 训练素材与权重
- `data/raw_images/_classes.txt` —— **按行号索引，连改都要小心**
- `data/model_registry.json` —— 新层也从这里解析模型
- `scripts/train_yolo26.py` `scripts/build_ui_v2.py` 等训练/建库脚本
- `data/ocr_model/ba_rec.onnx`
- `vision/` —— OCR 归一化词表（新层的 `read.py` 暂未用，但训练侧在用）

## 三、可以跟着一起删的（依赖老 skill，且新层不需要）

`scripts/bot_play_quest.py` `scripts/sweep_event.py` `scripts/event_bonus_rerun.py`
`scripts/step_probe.py` `scripts/step_walk.py` `scripts/live_capture_verify.py`
`tests/replay/`，以及仓库根目录那一堆 `_probe_*.py` / `_shopdbg*.py` / `_cal_*.py`
（都是一次性探针，功能已被 `py -m routing_v2 probe/step` 取代）。

注意 `scripts/audit_cls_usage.py` 要**改**不要删 —— 它的扫描目标要从 `brain/` 换成
`routing_v2/`，那道 `--fail-on-dead` 闸仍然有用。

## 四、`server/app.py`（4898 行）

它同时是**标注前端**（数据集浏览、预标、类表编辑）和**老 pipeline 的控制台**。
标注那半边跟老 skill 无关，必须留。删的时候只能拆：把 `DailyPipeline` 相关的
路由摘掉，换成 `from routing_v2.app.api import router as v2_router`。
**这一步单独做，别和删 skill 混在一起。**

---

## 五、准入条件（满足才动手删）

对每一条 flow：

1. `py -m routing_v2 step --flow <name>` **逐帧走一遍**，每一步都看锁定框图确认
   落点对；
2. `py -m routing_v2 run --flows <name>` 自主跑通，`outcome` 是 CLEAN 或
   **说得出理由的** LEFTOVER；
3. 台账里青辉石 **零变动**；
4. 涉及金钱的那几条（`shop` / `event_shop`）必须带 `--money-ok` 由人逐帧审过一次。

全部 13 条过了之后，删除按这个顺序：

```bash
git checkout -b pre-delete-snapshot && git checkout main   # 先留个还原点
```
1) 删根目录一次性探针 -> 2) 删 `scripts/` 里依赖老 skill 的 -> 3) 拆 `server/app.py`
的 pipeline 半边 -> 4) 删 `brain/skills/` -> 5) 删 `brain/pipeline.py`（**最后**，
它还带着 `combat_brain` 用得上的东西）。

每一步之后跑 `py -m routing_v2.tests.test_offline` + 一次 `run --flows mail`。
