# 私人碧蓝档案助手

**碧蓝档案 (Blue Archive)** 全自动日常助手 + 实时战斗目标锁定 — 纯本地运行，OCR + YOLO + Florence-2 + ByteTrack 驱动，无需云端 API。

在 MuMu 模拟器上通过 DXcam 屏幕捕获 + ADB 实现零侵入自动化：每日咖啡厅、课程表、社团、商店、制造、活动刷体力、悬赏通缉、战术大赛、邮件、每日任务、Hard 关卡等全流程一键执行。

---

## 核心能力

### 日常自动化
- **8 技能全流程闭环**：Cafe → Schedule → Shop → EventFarming → Bounty → Arena → Mail → DailyTasks
- **智能弹窗处理**：内嵌公告、退出确认、通知弹窗自动拦截
- **收藏角色管理**：课程表 / 咖啡厅邀请优先选择收藏角色
- **Florence-2 视觉问答**：头像匹配、按钮状态判断、课程表角色识别

### 实时战斗目标锁定 (NEW)
- **YOLOv8n + ByteTrack** 实时角色头部锁定
- **DXcam 240Hz 捕获** + 4090 nano 推理 (~2ms/帧)
- **ByteTrack 双阶段匹配**：高 conf 先匹配，低 conf 穿特效救援
- **冻结式预测**：VFX 遮挡时框原地等待，不漂移不乱飞
- **同 class NMS**：防止框裂变/重影
- **Win32 透明覆盖层**：250Hz 渲染，像素级对齐游戏窗口

### 视觉检测栈
| 组件 | 用途 | 速度 |
|------|------|------|
| RapidOCR | 全屏文字识别 (中/英/日) | ~50ms |
| YOLOv8n headpat.pt | 咖啡厅摸头气泡检测 | ~2ms |
| YOLOv8n battle_heads.pt | 战斗角色头部锁定 | ~2ms |
| Florence-2-large-ft | 头像匹配 / VQA / OVD | ~30ms |
| HSV 模板匹配 | 红点/黄点/货币图标 | <1ms |
| ByteTrack | 目标跟踪 + EMA 平滑 | <0.1ms |

---

## 快速开始

### 环境要求

- Windows 10/11
- Python 3.11+
- [MuMu Player 12](https://mumu.163.com/) 运行碧蓝档案
- NVIDIA GPU（推荐 RTX 3060+，RTX 4090 可跑满 240Hz）

### 安装

```powershell
git clone https://github.com/C0k11/blue-archive-assistant.git
cd blue-archive-assistant
pip install -r requirements.txt
```

### 运行日常自动化

```powershell
# 方式 1：MuMu 直接运行
py mumu_runner.py

# 方式 2：Web Dashboard
py -m uvicorn server.app:app --host 127.0.0.1 --port 8000
# 打开 http://127.0.0.1:8000/dashboard.html
```

### 运行战斗锁定 Demo

```powershell
# 打开碧蓝档案进入战斗，然后运行：
py scripts/battle_overlay_demo.py --fps 240 --conf 0.05

# 参数说明：
#   --fps 240    目标检测帧率（受 DXcam/GPU 限制，实际 ~45 FPS）
#   --conf 0.05  YOLO 置信度（低阈值喂给 ByteTrack 双阶段匹配）
```

---

## 模型

| 模型 | 文件 | 训练数据 | mAP50 |
|------|------|----------|-------|
| 咖啡厅摸头气泡 | `headpat.pt` | 1808 cafe 帧 (HSV 自动标注) | 0.96 |
| 战斗角色头部 | `battle_heads.pt` | 52 帧 (手动标注 + 增强) | 0.995 |

模型存放：`D:\Project\ml_cache\models\yolo\`

## 跟踪算法：ByteTrack + 冻结预测

```
YOLO 检测 (conf ≥ 0.05)
    │
    ├── 高 conf (≥ 0.25) ──→ Stage 1: 匹配已有 tracks
    │                              ↓ 未匹配 → 创建新 track
    │
    └── 低 conf (< 0.25) ──→ Stage 2: 救援未匹配 tracks（穿透 VFX）
                                   ↓ 未匹配 → 丢弃（不创建新 track）

Track 生命周期：
    ├── 匹配成功 → EMA 平滑更新 (α=0.85)
    ├── 未匹配   → 冻结原地 (vx=vy=0)，conf×0.92/帧
    └── 连续 5 帧未匹配 → 删除 track

同 class NMS：center_dist < 1.0 的同类 track 合并（杀死双黄蛋）
```

## 训练工具

```powershell
# DXcam 数据采集（前端录制）
# 在 Dashboard 点 "开始录制"，或：
py scripts/collect_data.py --interval 0.5

# 咖啡厅摸头 HSV 自动标注 + 训练
py scripts/auto_label_headpat_v3.py
py scripts/train_headpat_yolo.py

# 战斗角色标注（手动 YOLO 格式 .txt）+ 训练
py scripts/build_battle_dataset.py
# 然后用 ultralytics YOLO train
```

## 轨迹系统

每次运行自动记录到 `data/trajectories/`，每 tick 保存：
- 截图 (.jpg) + 元数据 (.json)
- OCR / YOLO / Florence 检测结果
- 当前技能、子状态、执行动作

## 项目结构

```
ai-game-secretary/
├── brain/
│   ├── pipeline.py              # 技能调度器 + 全局拦截器
│   └── skills/
│       ├── base.py              # ScreenState, BaseSkill
│       ├── cafe.py              # 咖啡厅（收益/邀请/摸头）
│       ├── schedule.py          # 课程表（Florence 头像匹配）
│       ├── event_farming.py     # 活动刷体力
│       ├── shop.py / bounty.py / arena.py / mail.py / daily_tasks.py
│       └── farming.py           # Hard 关卡
├── vision/
│   ├── yolo_detector.py         # YOLOv8 推理
│   ├── florence_vision.py       # Florence-2 VQA / OVD / 头像匹配
│   ├── template_matcher.py      # 多模板匹配（红点/黄点/货币）
│   └── avatar_matcher.py        # 头像特征匹配
├── scripts/
│   ├── box_tracker.py           # ByteTrack 跟踪器
│   ├── yolo_overlay.py          # Win32 透明覆盖层 (250Hz)
│   ├── battle_overlay_demo.py   # 战斗锁定 Demo (DXcam + YOLO)
│   ├── collect_data.py          # ADB 数据采集
│   └── auto_label_headpat_v3.py # HSV 自动标注
├── server/
│   └── app.py                   # FastAPI 后端 + DXcam 录制
├── mumu_runner.py               # MuMu 主运行入口
├── dashboard.html               # Web 控制面板
└── requirements.txt
```

## 未来计划

- [ ] **总力战 (Raid)** 轴抄写自动化
- [ ] **大决战 (Grand Assault)** 自动化
- [ ] **VFX 增强训练**：在训练集中叠加爆炸/闪光特效提升遮挡召回率
- [ ] **CSRT Fallback**：YOLO 连续丢失时切换 OpenCV 传统跟踪器

## License

MIT
