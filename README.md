# Game-UI CV Agent

A computer-vision agent that plays a mobile game's daily routine by looking at the
screen and clicking. No game files are modified, no packets are touched, nothing
leaves the machine.

The target game is Blue Archive running in an Android emulator. The perception and
control stack itself is not game-specific: it reads pixels and sends taps.

## 概述

本机运行的《蔚蓝档案》日常自动化助手。它看模拟器画面、模拟人的点击，把每天的例行
事务跑完：收咖啡厅、扫荡关卡、打竞技场、领邮件和任务奖励。活动期间可以自动推图，
按活动加成规划体力去向。不改游戏文件、不碰网络协议，全部在本地完成。

花钱 = 点确认成交。读不出余额就不点花钱键。进购买青辉石页、给购买键标框、
领免费包不是花钱。青辉石和 CAD 购买键程序不点，详见下面的付费安全一节。
大厅信用点「选择购买」默认关（`shop.credit_buy`）；战术大赛商店买饮料默认开
（`shop.arena_shop`）。两个都是前端开关，互不影响。

## 感知模型

三个 YOLO 检测器加一个数字 OCR，分工固定：页面识别和按钮判定全部来自检测器，
OCR 只读数字字段（体力、票数、货币），不参与任何判断。

| 用途 | 模型 | 职责 | 单帧延迟 |
|---|---|---|---|
| UI 导航 | YOLO26m v18 | 按钮、页签、弹窗、角标，驱动全部导航和点击 | imgsz=960 |
| 角色识别 | YOLO26x | 检测框加角色身份一次推理给出，含战斗技能卡的灰置、充能状态 | ~10 ms |
| 战斗单位 | YOLO26s | 我方、敌方、5 种 Boss、胜利横幅、HUD | ~12 ms |
| 数字读取 | PP-OCRv4，按 BA 字形微调 | 只读数字 | ~50 ms |

现役 UI 是 v18（`data/model_registry.json` 的 `ui.active`）。运行时解析权重
路径。回滚改一行 `active` 为 `v17`。推理必须 `imgsz=960`。弱类（购买灰 /
选择购买 / MAX）val=0，总 mAP 不代表它们，要 live 对账。

现役仍有 3 处 `tap_at` 过渡盲点：任务大厅入口 (0.944, 0.942)、大厅藏 UI
(0.40, 0.55)、空屏唤醒 (0.50, 0.50)。都是该类没盖住或没训好，进 v19
补训，训回删码。裸大厅入口 live 约 0.08-0.15，过不了门控 0.45。

## 数据

模型全部自行标注。素材来自 bot 实际跑线时录制的干净帧，用 Android 内部
`screencap` 抓取，画面里没有桌面层的调试元素。

- 素材池 82,590 帧，经过内容去重、同源泄漏筛查和逐类标注审计
- 已标注 52,750 帧，共 749,286 个框
- 类表 531 行（528 任务资讯 / 529 得星_1 占位 train=0 / 530 得星_2）
- 当前 UI 数据集：train 38,176 帧，val 1,507 帧（泄漏洗后）

## 付费安全

可能花掉付费货币的动作要连续通过几道独立的闸：

- 成交框的判据是结构性的（弹窗体内青辉石 / 同框对话框 / 双键框内步进器），货架页本身不是成交
- 余额读取失败就不点花钱键，不用默认值代替
- 未授权的购买键拒这一发，不停整轮
- 派发类操作按游戏日记账，同一天不会重复执行

## 快速开始

环境要求：

- Windows 10/11，Python 3.11+
- MuMu Player 12，游戏保持运行
- NVIDIA GPU。日常链路以 CPU 为主，有卡即可；战斗锁定建议 RTX 3060 以上

安装：

```bash
git clone https://github.com/C0k11/game-ui-cv-agent.git
cd game-ui-cv-agent
pip install -r requirements.txt
```

### 控制台

```bash
py -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/dashboard.html`。控制台承担三件事：查看 bot 运行
状态和日志、从模拟器采集新帧、给帧做标注。训练数据的采集和标注都在这里完成，
不需要别的工具。

### 跑日常

```bash
py -m routing_v2 run
```

按 flow 依次执行：咖啡厅收菜和摸头、体力扫荡、竞技场、邮件领取、任务和成就
奖励。每个玩法是 `routing_v2/flow/` 下的一个独立 flow，页面跳转由 UI 检测器
逐步确认，中途识别不到预期页面会停在原地报错，不会盲点。

### 跑战斗

```bash
py -u scripts/bot_play_quest.py 10 11 12
```

参数是要打的关卡编号。战斗链路走 scrcpy 视频流取帧，行为树决定出牌时机，
角色和技能卡状态由战斗检测器实时给出。

### 训练与上线

迭代一个模型的完整循环：

1. 控制台里采集新帧并标注
2. 构建数据集：`py -X utf8 scripts/build_ui_v2.py`
3. 训练：`py -X utf8 scripts/train_yolo26.py ui_yolo26m_v18`
4. 上线：冻结权重，把 `data/model_registry.json` 里对应条目的 `active` 指向新版本

评估和对比脚本在 `scripts/` 下，训练产物默认写到仓库外的缓存目录。

## 目录结构

    routing_v2/          当前架构：感知 / 状态 / 动作 / flow 四层
      percept/           抓帧、检测、读数
      state/             页面识别、类表词汇
      act/               动作、推进闸、金钱闸、记账
      flow/              每个玩法一个 flow
    brain/               上一代架构，仍承担战斗链路
    server/              FastAPI 后端 + 标注控制台
    scripts/             训练 / 建集 / 评估 / 采集工具
    data/
      model_registry.json   现役模型版本，单一真相源
      raw_images/           标注帧 + _classes.txt 主类表

模型权重、数据集和 HF 缓存放在仓库之外的独立目录（已 gitignore），路径由
`HF_HOME` 和 `model_registry.json` 配置。checkout 很小，不含二进制。

## License and Disclaimer

仅供个人学习使用。仓库不分发任何游戏文件，素材保留在你自己的模拟器安装里。
本项目与 Yostar、Nexon、Bilibili、NetEase 均无关联。
