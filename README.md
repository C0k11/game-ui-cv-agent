# 碧蓝档案日常助手

**Blue Archive** 全自动日常 + 实时战斗目标锁定，**纯本地运行，零云端**。

自研自动化管线，以 **OCR + 模板匹配 + 状态机** 为主干驱动日常全流程，辅以 YOLO 做战斗锁定；通过 MuMu 模拟器 DXcam 屏幕捕获 + PostMessage 注入实现零侵入操控。提供 **Windows 原生 WebView2 启动器** 一键出盒即用。

---

## 亮点

- **22 个日常技能** 串成一键刷日常（大厅→AP 溢出保护→活动刷本→咖啡厅→课程表→社团→MomoTalk→商店→…→主线推进）
- **可视化课程表头像区域编辑器**（画布拖拽/缩放/右键加框，Del 删除，Ctrl+D 复制，**JSON 一键 Copy/Paste 跨机器迁移配置**）
- **Event 刷本多轮扫荡**（`Event Rounds` + `AP Reserve`，借鉴 reference `activity_sweep_times` 的预算思路，见下文）
- **自训练 OCR 模型**（PP-OCRv4 碧蓝档案微调，词汇准确率 +20%）
- **实时战斗锁定**（240Hz DXcam + YOLOv8n + ByteTrack + 冻结预测，Win32 透明覆盖层）
- **异步轨迹写入**（每 tick 截图 + 元数据后台线程落盘，不阻塞主循环）
- **Windows 一键启动器**（.NET 8 + WebView2，免终端运行）

---

## 功能矩阵

| 模块 | 功能 | 识别栈 | 备注 |
|------|------|-------|------|
| **Lobby** | 弹窗清理、签到、公告 | OCR + 模板 | 自动处理更新/通知/TOUCH TO START |
| **AP 溢出保护** | AP≥900 优先刷本 | OCR 数值 | 防咖啡厅结算卡住 |
| **EventActivity** | 活动剧情、挑战、任务 | OCR + 状态机 | 自动识别 活動進行中 |
| **EventFarming** | 活动扫荡（Normal/Hard/特殊任务） | OCR + 状态机 | **支持 max_rounds 多轮 + AP 保留下限** |
| **Cafe** | 收益领取、邀请券、摸头 | 模板 (主) + YOLO (备) | happy_face 模板优先；1F 左→右、2F 右→左扫图 |
| **Schedule** | 收藏角色优先排课 | OCR + `AvatarMatcher` | 模板 + HSV 直方图匹配，画布可视化编辑区域 |
| **Club** | 领社团 AP | OCR | |
| **MomoTalk** | 自动回复未读消息 | OCR + 状态机 | 按未读排序，自动对话/故事跳过 |
| **Shop** | 日常免费 / 低价购买 | OCR + 状态机 | 检测购买完成/刷新按钮 |
| **Craft** | 领取 + 快速制造 | OCR + 状态机 | |
| **StoryCleanup** | 主线/小组/迷你剧情 | OCR + 状态机 | MENU→跳过流程，编队/战斗处理 |
| **Bounty** | 最高难度扫荡 | OCR + 状态机 | 三分支轮转（高架/鐵道/教室）+ 票券重检 |
| **Arena** | 领奖 + 自动对战 | OCR + 状态机 | 冷却等待、最优对手选择 |
| **JointFiringDrill / TotalAssault** | 自动参加 | OCR + 状态机 | |
| **Mail / DailyTasks / PassReward** | 一键领取 | OCR | |
| **ApPlanning** | 免费体力 + 购买策略 | OCR + 数值 | 可配置购买上限，禁止高级货币开关 |
| **HardFarming / CampaignPush** | 指定 / 兜底刷本 | OCR + 状态机 | |
| **BattleOverlay** | 实时头部框锁定 | YOLOv8n + ByteTrack | 240Hz DXcam + Win32 透明覆盖层 |

---

## 视觉识别栈

| 层级 | 组件 | 用途 | 延迟 |
|------|------|------|------|
| **L1 主干** | RapidOCR + **BA fine-tuned rec** | 全屏中/英/日文字识别 | ~50 ms |
| **L1 主干** | `cv2.matchTemplate` | 模板匹配（happy_face / 头像 / UI 图标） | <1 ms |
| **L1 主干** | HSV 像素分析 | 颜色状态（房间/按钮/勾选） | <1 ms |
| **L2 战斗** | YOLOv8n `battle_heads.pt` | 战斗角色头部 | ~2 ms |
| **L2 备用** | YOLOv8n `headpat.pt` | 咖啡厅摸头气泡（模板未命中时启用） | ~2 ms |
| **L3 跟踪** | ByteTrack + EMA | 目标跟踪 + 冻结预测 | <0.1 ms |

> **设计原则**：日常管线以 OCR + 模板 + 状态机为唯一主干，不依赖重型模型；YOLO 仅用于战斗锁定 + 咖啡厅备用。

### OCR 微调

PP-OCRv4 在碧蓝档案繁/简/英/日混合文字上微调：

| 指标 | 默认 PP-OCRv3 | BA fine-tuned | 提升 |
|------|-------------|---------------|------|
| 词汇精确匹配 | 35.8% | **55.8%** | **+20%** |
| 全样本精确匹配 | 19.2% | 20.8% | +1.6% |

五步管线在 `scripts/ocr_training/`：裁切 → 合成 → 训练 → 导 ONNX → 评估。产出 `data/ocr_model/ba_rec.onnx`，服务启动自动加载。

---

## EventFarming：借鉴 reference 的预算控制

借鉴 [reference](https://github.com/pur1fying/blue_archive_auto_script) `activity_sweep` 模块中 **"按 AP 预算扫荡 N 次，保留底线"** 的思路，加入两个非破坏性参数（默认行为不变）：

| 参数 | 默认 | 含义 |
|------|------|------|
| `event_max_rounds` | `1` | 每次 `EventFarming` / `ap_overflow` 技能执行内的扫荡循环次数；>1 时在 sweep 结果弹窗消失后自动重新选择当前关卡再扫荡 |
| `event_ap_reserve` | `0` | AP 下限：当前 AP ≤ 此值时立刻停止循环，保证不刷光（reference 的"-1=用完全部"可通过 `event_max_rounds=10 + event_ap_reserve=0` 近似） |

**与 reference 的差异**：reference 需要为每个活动手工维护 JSON stage 表（`src/explore_task_data/activities/<EventName>.json` 带 AP cost / 关卡列表），对当前活动覆盖精准，但需要频繁更新；我们选择 **自动检测 活動進行中 badge → 滚到最底扫荡 → 点 MAX** 的零配置路径，覆盖所有 夏莱 系活动与 Quest 型活动，代价是粒度只能到"每轮 MAX"。`event_max_rounds` 正是把这颗粒度从"1 次 MAX"扩展到"N 次 MAX + 保留 AP_reserve"，与 reference 的 `sweep_times = [-1, 0.5, 3]` 在语义上对齐。

Dashboard **Agent → AP Purchase Limit** 旁边的 `Event Rounds` / `AP Reserve` 两个输入框即可直接配置，保存到档案。

---

## 快速开始

### 环境要求

- **Windows 10/11**
- **Python 3.11+**（建议用 `py -3.11` 启动）
- **[MuMu Player 12](https://mumu.163.com/)** 运行碧蓝档案
- NVIDIA GPU（推荐 RTX 3060+；战斗锁定需要，日常管线仅需 CPU）

### 安装

```powershell
git clone https://github.com/C0k11/blue-archive-assistant.git
cd "blue-archive-assistant"
pip install -r requirements.txt
```

### 运行日常自动化

**方式一：Windows 一键启动器（推荐）**

下载 [Releases](https://github.com/C0k11/blue-archive-assistant/releases) 里的 `GameSecretaryApp.exe`（.NET 8 WebView2），双击即可。启动器自动拉起 uvicorn + 打开 Dashboard。

**方式二：终端启动**

```powershell
py -m uvicorn server.app:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000/dashboard.html
```

**方式三：脚本直跑（无 UI）**

```powershell
py mumu_runner.py
```

### 运行战斗锁定 Demo

```powershell
py scripts/battle_overlay_demo.py --fps 240 --conf 0.05
```

---

## Dashboard 主要页面

- **Home**：多档案切换、技能顺序编排、AP/Event 预算、收藏角色选择、Dry Run 开关
- **HUD**：实时状态（当前技能、子状态、tick 数、AP、最近动作原因）
- **Roster** 🆕：课程表头像识别区域可视化编辑器
  - 左键拖框 = 移动；拖角/边 = resize
  - 右键空白 = 新增；Del = 删除；Ctrl+D = 复制；Ctrl+S = 保存
  - **📋 Copy JSON / 📥 Paste JSON** = 跨机器/跨分辨率迁移配置
- **Annotate**：YOLO/OCR 标注中心（矩形、椭圆、自由笔刷，右键拖拽可靠）
- **Trajectories**：历史运行记录回看（每 tick 截图 + OCR + 动作）

---

## 战斗锁定：ByteTrack + 冻结预测

```
YOLO 检测 (conf ≥ 0.05)
    |
    +-- 高 conf (≥ 0.25) --> Stage 1: 匹配已有 tracks
    |                              └─ 未匹配 → 创建新 track
    |
    +-- 低 conf (< 0.25)  --> Stage 2: 救援未匹配 tracks（穿透 VFX）
                                   └─ 未匹配 → 丢弃

Track 生命周期：
    ├─ 匹配成功 → EMA 平滑更新 (α=0.85)
    ├─ 未匹配   → 冻结原地 (vx=vy=0)，conf × 0.92/帧
    └─ 连续 5 帧未匹配 → 删除

同 class NMS：center_dist < 1.0 的同类 track 合并
```

## 模型

| 模型 | 文件 | 训练数据 | 指标 | 用途 |
|------|------|---------|------|------|
| 战斗角色头部 | `battle_heads.pt` | 52 帧手标 + 增强 | mAP50 = 0.995 | BattleOverlay |
| 咖啡厅摸头气泡 | `headpat.pt` | 1808 cafe 帧（HSV 自动标注） | mAP50 = 0.96 | 模板失败备用 |
| OCR 识别 | `ba_rec.onnx` | 轨迹 OCR 裁切 + 合成 | 词汇 acc = 55.8% | 全管线文字识别 |

---

## 性能优化

- **异步轨迹写入**：`brain/pipeline.py` 使用 `Queue(maxsize=64)` + 后台线程落盘，每 tick 主循环不再阻塞 10–50 ms 磁盘 IO；JSON 改用紧凑分隔符 (-35% bytes)
- **头像匹配缓存**：`vision/avatar_matcher.py` 在 `match_avatar` 热路径上按 `(name, h, w)` 缓存 `cv2.resize` 结果 + 按 `(h, w)` 缓存圆形掩膜，避免 9 房间 × 4 格 × N 候选 每帧重算
- **OCR 结果缓存**：同帧多次 `find_text` 共用同一次 OCR 调用结果
- **YOLO 延迟加载**：仅战斗覆盖层/摸头备用路径按需加载

---

## 项目结构

```
ai-game-secretary/
├── brain/
│   ├── pipeline.py              # 技能调度器 + 全局拦截器 + 异步轨迹写入
│   └── skills/
│       ├── base.py              # ScreenState / BaseSkill / 公共弹窗处理
│       ├── lobby.py             # 大厅恢复/弹窗清理
│       ├── event_farming.py     # 活动刷体力（max_rounds / ap_reserve）
│       ├── event_activity.py    # 活动剧情/挑战入口
│       ├── ap_planning.py       # AP 规划/免费体力/购买策略
│       ├── cafe.py              # 咖啡厅（收益/邀请/模板摸头）
│       ├── schedule.py          # 课程表（OCR + AvatarMatcher）
│       ├── campaign_push.py     # 主线推进/兜底清体力
│       └── club / momo_talk / shop / craft / story_cleanup /
│           bounty / arena / jfd / total_assault / mail /
│           daily_tasks / pass_reward / farming
├── vision/
│   ├── engine.py                # OCR 引擎 + 截图缓存
│   ├── avatar_matcher.py        # 头像模板 + HSV 直方图（带 resize/mask 缓存）
│   ├── template_matcher.py      # 多尺度模板匹配
│   ├── yolo_detector.py         # YOLOv8 推理
│   ├── florence_vision.py       # Florence-2（可选实验性，非日常主干）
│   └── window.py                # 窗口/区域坐标工具
├── server/
│   ├── app.py                   # FastAPI 后端 + OCR 服务 + 所有 /api/v1/*
│   └── dashboard.html           # Web Dashboard（Home/HUD/Roster/Annotate/Trajectories）
├── scripts/
│   ├── box_tracker.py           # ByteTrack 跟踪器
│   ├── yolo_overlay.py          # Win32 透明覆盖层 (240Hz)
│   ├── battle_overlay_demo.py   # 战斗锁定 Demo
│   ├── collect_data.py          # DXcam 数据采集
│   ├── auto_label_*.py          # HSV 自动标注
│   ├── train_*.py               # YOLO 训练脚本
│   ├── ocr_training/            # OCR 微调五步管线
│   └── _test_*.py               # 回归测试（gitignored，本地录制轨迹）
├── windows_app/                 # .NET 8 WebView2 启动器
├── data/
│   ├── captures/                # 模板图片 + 角色头像库
│   ├── ocr_model/               # ba_rec.onnx
│   ├── models/                  # .pt / .onnx（gitignored）
│   ├── trajectories/            # 运行轨迹（gitignored）
│   ├── app_config.json          # 多档案配置
│   └── schedule_avatar_regions.json  # 课程表头像区域（Roster 页保存）
├── mumu_runner.py               # 无 UI 的直跑入口
├── launch.py                    # 启动器内部调用入口
├── requirements.txt
└── README.md
```

---

## 轨迹系统

每次 `DailyPipeline.start()` 在 `data/trajectories/run_<YYYYMMDD_HHMMSS>/` 创建目录，每 tick 异步落盘：
- `tick_NNNN.jpg`：截图
- `tick_NNNN.json`：OCR 结果、YOLO 结果、当前技能、子状态、动作、原因、时间戳

Dashboard **Trajectories** 页可直接回看历史运行，**Roster** 页从轨迹 JPG 中自动挑选课程表帧作为编辑底图。

---

## 训练与标注

```powershell
# DXcam 采集
py scripts/collect_data.py --interval 0.5

# 咖啡厅摸头气泡 HSV 自动标注 + 训练
py scripts/auto_label_headpat_v3.py
py scripts/train_headpat_yolo.py

# OCR 微调
cd scripts/ocr_training
py 1_crop_from_trajectories.py
py 2_synth_data.py
py 3_train.py
py 4_export_onnx.py
py 5_eval.py
```

---

## Roadmap

- [ ] 总力战 (Raid) 轴抄写自动化
- [ ] 大决战 (Grand Assault) 自动化
- [ ] 多模拟器支持（蓝叠 / 雷电 / LDPlayer）
- [ ] VFX 增强训练：叠加爆炸/闪光提升战斗遮挡召回率
- [ ] VLM 伪标签蒸馏 OCR
- [x] **Event 刷本预算 (reference activity_sweep_times 适配)** ✓
- [x] **课程表头像区域画布编辑器 + JSON 跨机器迁移** ✓
- [x] **异步轨迹写入 + 头像匹配缓存** ✓
- [x] Windows 一键启动器（.NET 8 + WebView2） ✓
- [x] OCR 微调：PP-OCRv4 碧蓝档案专用，词汇 +20% ✓
- [x] 标注中心：旋转框、椭圆框、自由笔刷 ✓

---

## Credits

- **[pur1fying/blue_archive_auto_script](https://github.com/pur1fying/blue_archive_auto_script)** (reference)：
  本项目 `EventFarming.max_rounds` / `ap_reserve` 预算控制受 reference `activity_sweep_times` / `activity_sweep_task_number` 的设计启发，用意配合我们的 **零配置自动检测** 路径给出可控的 AP 预算上限；`study/ref/` 下为仅供本地参考的源码阅读副本（`.gitignore` 已排除）。
- **[RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR)**：OCR 推理基座
- **[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)**：YOLOv8
- **[ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)**：跟踪器算法

---

## License

MIT
