# 私人碧蓝档案助手

**碧蓝档案 (Blue Archive)** 全自动日常助手 + 实时战斗目标锁定 — 纯本地运行，零云端依赖。

自研自动化管线，以 **OCR + 模板匹配 + 状态机** 为主干驱动日常全流程，辅以 YOLO 进行战斗锁定；通过 MuMu 模拟器 DXcam 屏幕捕获 + ADB 注入实现零侵入操控。

---

## 功能矩阵

| 模块 | 功能 | 主要识别方式 | 备注 |
|------|------|-------------|------|
| **大厅 (Lobby)** | 弹窗清理、公告关闭、通知确认 | OCR + 模板匹配 | 自动处理更新公告/签到/通知 |
| **AP 溢出保护** | 超上限时优先消耗体力 | OCR 数值解析 | |
| **活动剧情 (EventActivity)** | 活动入口、剧情推进、挑战 | OCR + 状态机 | 自动识别活动时效 |
| **活动刷体力 (EventFarming)** | Hard 扫荡、体力消耗 | OCR + 状态机 | 支持 Normal/Hard 自动切换 |
| **咖啡厅 (Cafe)** | 收益领取、邀请券、摸头 | 模板匹配 (主) + YOLO (备) | happy_face 模板优先；1F 左→右、2F 右→左扫图 |
| **课程表 (Schedule)** | 收藏角色优先排课 | OCR + 模板匹配 (头像) | AvatarMatcher 模板比对，无需 Florence |
| **社团 (Club)** | 领取社团 AP | OCR | |
| **MomoTalk** | 自动回复未读消息 | OCR + 状态机 | 按未读排序，自动对话/故事跳过 |
| **商店 (Shop)** | 每日免费 / 低价购买 | OCR + 状态机 | 自动检测购买完成/刷新按钮 |
| **制造 (Craft)** | 领取 + 快速制造 | OCR + 状态机 | |
| **剧情清理 (StoryCleanup)** | 主线/小组/迷你剧情 | OCR + 状态机 | MENU→跳过流程，编队/战斗处理 |
| **悬赏通缉 (Bounty)** | 最高难度扫荡 | OCR + 状态机 | 多分支轮转+票券重检 |
| **战术对抗赛 (Arena)** | 领奖 + 自动对战 | OCR + 状态机 | 冷却等待、最优对手选择 |
| **联合火力演习 (JFD)** | 自动参加 | OCR + 状态机 | |
| **总力战 (TotalAssault)** | 自动参加 | OCR + 状态机 | |
| **邮件 (Mail)** | 一键领取 | OCR | |
| **每日任务 (DailyTasks)** | 一键领取 | OCR | |
| **通行证 (PassReward)** | 领取奖励 | OCR | |
| **AP 规划 (APPlanning)** | 每日免费体力 + 购买策略 | OCR + 数值解析 | 可配置购买上限 |
| **Hard 刷取** | 指定关卡扫荡 | OCR + 状态机 | |
| **主线推进 (CampaignPush)** | 兜底清体力 | OCR + 状态机 | |
| **战斗锁定 (BattleOverlay)** | 实时头部框锁定 | YOLOv8n + ByteTrack | 240Hz DXcam 捕获 + Win32 透明覆盖层 |

---

## 视觉识别栈

| 层级 | 组件 | 用途 | 速度 | 角色 |
|------|------|------|------|------|
| **L1 主干** | RapidOCR | 全屏文字识别 (中/英/日) | ~50ms | 所有技能的文字判据 |
| **L1 主干** | cv2.matchTemplate | 模板匹配 (happy_face / 头像 / UI 图标) | <1ms | 咖啡厅摸头、课程表头像、红点检测 |
| **L1 主干** | HSV 像素分析 | 颜色状态判断 (房间/按钮/勾选) | <1ms | 课程表房间状态、头像绿勾/红心 |
| **L2 战斗** | YOLOv8n battle_heads.pt | 战斗角色头部锁定 | ~2ms | 仅战斗覆盖层使用 |
| **L2 备用** | YOLOv8n headpat.pt | 咖啡厅摸头气泡 (备用) | ~2ms | 仅当模板匹配无结果时启用 |
| **L3 跟踪** | ByteTrack | 目标跟踪 + EMA 平滑 | <0.1ms | 战斗覆盖层 |

> **设计原则**：日常管线以 OCR + 模板 + 状态机为唯一主干，不依赖重型模型；YOLO 仅在战斗锁定和咖啡厅备用路径中使用。

---

## 快速开始

### 环境要求

- Windows 10/11
- Python 3.11+
- [MuMu Player 12](https://mumu.163.com/) 运行碧蓝档案
- NVIDIA GPU（推荐 RTX 3060+；战斗锁定需要，日常管线仅需 CPU）

### 安装

```powershell
git clone https://github.com/C0k11/blue-archive-assistant.git
cd blue-archive-assistant
pip install -r requirements.txt
```

### 运行日常自动化

```powershell
# Web Dashboard（推荐）
py -m uvicorn server.app:app --host 127.0.0.1 --port 8000
# 打开 http://127.0.0.1:8000/dashboard.html

# 或直接 MuMu 运行
py mumu_runner.py
```

Dashboard 支持：多档案切换、技能顺序编排、收藏角色选择、实时 HUD 状态监控、AP 购买上限设置。

### 运行战斗锁定 Demo

```powershell
py scripts/battle_overlay_demo.py --fps 240 --conf 0.05
```

---

## 战斗锁定：ByteTrack + 冻结预测

```
YOLO 检测 (conf >= 0.05)
    |
    +-- 高 conf (>= 0.25) --> Stage 1: 匹配已有 tracks
    |                              | 未匹配 -> 创建新 track
    |
    +-- 低 conf (< 0.25)  --> Stage 2: 救援未匹配 tracks（穿透 VFX）
                                   | 未匹配 -> 丢弃

Track 生命周期：
    +-- 匹配成功 -> EMA 平滑更新 (alpha=0.85)
    +-- 未匹配   -> 冻结原地 (vx=vy=0)，conf x 0.92/帧
    +-- 连续 5 帧未匹配 -> 删除 track

同 class NMS：center_dist < 1.0 的同类 track 合并
```

## 模型

| 模型 | 文件 | 训练数据 | mAP50 | 用途 |
|------|------|----------|-------|------|
| 战斗角色头部 | `battle_heads.pt` | 52 帧 (手动标注 + 增强) | 0.995 | 战斗覆盖层 |
| 咖啡厅摸头气泡 | `headpat.pt` | 1808 cafe 帧 (HSV 自动标注) | 0.96 | 模板匹配备用 |

模型存放：`D:\Project\ml_cache\models\yolo\`

---

## 轨迹系统

每次运行自动记录到 `data/trajectories/`，每 tick 保存：
- 截图 (.jpg) + 元数据 (.json)
- OCR / 模板匹配检测结果
- 当前技能、子状态、执行动作与原因

## 训练与标注工具

Dashboard 内置标注中心 (Annotate 页签)：
- 数据集浏览、YOLO 格式标注框绘制
- 批量 OCR 扫描与结果编辑
- 支持自定义 class、框选/移动/缩放/旋转
- 矩形、椭圆、自由笔刷多边形三种标注形状
- Pointer Events 确保 Windows 右键拖拽可靠

```powershell
# DXcam 数据采集
py scripts/collect_data.py --interval 0.5

# 咖啡厅摸头 HSV 自动标注 + 训练
py scripts/auto_label_headpat_v3.py
py scripts/train_headpat_yolo.py
```

## 项目结构

```
ai-game-secretary/
├── brain/
│   ├── pipeline.py              # 技能调度器 + 全局拦截器
│   └── skills/
│       ├── base.py              # ScreenState, BaseSkill, 公共弹窗处理
│       ├── cafe.py              # 咖啡厅（收益/邀请/模板摸头）
│       ├── schedule.py          # 课程表（OCR + 模板头像匹配）
│       ├── event_activity.py    # 活动剧情/挑战入口
│       ├── event_farming.py     # 活动刷体力
│       ├── ap_planning.py       # AP 规划/免费体力
│       ├── campaign_push.py     # 主线推进/兜底清体力
│       ├── lobby.py             # 大厅恢复/弹窗清理
│       └── bounty / craft / momo_talk / story_cleanup / ...
├── vision/
│   ├── template_matcher.py      # 多尺度模板匹配（happy_face / UI 图标）
│   ├── avatar_matcher.py        # 头像模板 + 直方图比对
│   ├── yolo_detector.py         # YOLOv8 推理（仅战斗 + 备用）
│   └── florence_vision.py       # Florence-2（可选实验性，非日常主干）
├── scripts/
│   ├── box_tracker.py           # ByteTrack 跟踪器
│   ├── yolo_overlay.py          # Win32 透明覆盖层 (250Hz)
│   ├── battle_overlay_demo.py   # 战斗锁定 Demo
│   └── collect_data.py          # DXcam 数据采集
├── server/
│   └── app.py                   # FastAPI 后端 + DXcam 录制 + OCR 服务
├── dashboard.html               # Web 控制面板（助手/采集/标注三合一）
├── data/
│   ├── captures/                # 模板图片 + 角色头像库
│   ├── app_config.json          # 多档案配置
│   └── trajectories/            # 运行轨迹记录
├── mumu_runner.py               # MuMu 主运行入口
└── requirements.txt
```

## 未来计划

- [ ] 总力战 (Raid) 轴抄写自动化
- [ ] 大决战 (Grand Assault) 自动化
- [ ] VFX 增强训练：叠加爆炸/闪光特效提升战斗遮挡召回率
- [ ] 多模拟器支持 (蓝叠/雷电)
- [x] 标注中心：旋转框、椭圆框、自由笔刷多边形 ✓

## License

MIT
