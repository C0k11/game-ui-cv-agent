# Game-UI CV Agent

A computer-vision agent that plays a mobile game's daily routine by looking at the
screen and clicking. No game files are modified, no packets are touched, nothing
leaves the machine.

The target game is Blue Archive running in an Android emulator. The perception and
control stack itself is not game-specific: it reads pixels and sends taps.

## 概述

本机运行的《蔚蓝档案》日常自动化助手。它看模拟器画面、模拟人的点击，把每天的例行
事务跑完：收咖啡厅、扫荡关卡、打竞技场、领邮件和任务奖励。活动期间可以自动推图，
按活动加成规划体力去向；任務推图走「集中指挥」格子地图，按用户点名的关卡列表连推。
不改游戏文件、不碰网络协议，全部在本地完成。

花钱 = 点确认成交。读不出余额就不点花钱键。进购买青辉石页、给购买键标框、
领免费包不是花钱。青辉石和 CAD 购买键程序不点，详见下面的付费安全一节。
大厅信用点「选择购买」默认关（`shop.credit_buy`）；战术大赛商店买饮料默认开
（`shop.arena_shop`）。两个都是前端开关，互不影响。

## 感知模型

三个 YOLO 检测器加一个数字 OCR，分工固定：页面识别和按钮判定全部来自检测器，
OCR 只读数字字段（体力、票数、货币），不参与任何判断。

| 用途 | 模型 | 职责 | 备注 |
|---|---|---|---|
| UI 导航 | YOLO26m v20（v21 训练中） | 按钮、页签、弹窗、角标，驱动全部导航和点击 | imgsz=960 |
| 角色识别 | YOLO26x v6 | 检测框加角色身份一次推理给出，含战斗技能卡的灰置、充能状态 | ~10 ms |
| 战斗单位 | YOLO26s v11 | 我方、敌方、Boss、胜利横幅、HUD | ~12 ms |
| 数字读取 | PP-OCRv4，按 BA 字形微调 | 只读数字 | ~50 ms |

现役版本以 `data/model_registry.json` 的 `active` 为准（UI 为 `v20`），运行时解析
权重路径，回滚改一行 `active`。推理必须 `imgsz=960`。总 mAP 不代表弱类和 val=0
的类，这些只能 live 对账。

类表 `data/raw_images/_classes.txt`（546 行，479 现役 / 67 废案，与 v20 线上权重同表）
按行号索引，永不删行；废案只加 `_废弃N_` 前缀，检测层按名字屏蔽。要加新类时先复制
成 `_classes_next.txt` 追加并用它建集训练，新权重上线后再转正。每个类是什么、训练量多少、
被哪个 flow 引用，见自动生成的 `data/ui_cls_semantics.md`。

导航里只剩两处没有 cls 支撑的点击，都只在大厅收起 UI 变成空屏时点背景唤醒
（`routing_v2/flow/nav.py`），且必须写明理由；其余落点一律取检出框中心。

## 数据

模型全部自行标注。素材来自 bot 实际跑线时录制的干净帧，用 Android 内部
`screencap` 抓取，画面里没有桌面层的调试元素。

- 素材池 596 个目录、112,060 帧，已标注 66,438 帧、970,467 个框
- 2026-09-03 做过一次全池标注体检与修复：用人审金标池的像素中位模板逐框匹配（与模型无关，
  避免模型学会脏标后自证一致），结论是位置漂移不到 1%，真问题是框口径按池分裂；按金标口径
  吸附约 5,600 框、活动商店 只框字的转成整图标 1,780、改类 86、删幽灵与边条垃圾框 209，
  再按 v20 检出补漏标约 7,000（两态类用像素规则定状态）。全部改动带修前备份和逐条台账
- 当前 UI 数据集（v21，nc=546）：train 38,945 帧（去重后 38,741 唯一帧，<30 帧的类
  过采样到 30），val 11,052 帧；构建时按内容去重、剔除 train/val 同图、丢弃废案类的框
- val 按天或按独立录制 session 划分，与 train 不同源；新类没有独立 val 时只能 live 验

## 付费安全

可能花掉付费货币的动作要连续通过几道独立的闸：

- 成交框的判据是结构性的（弹窗体内青辉石 / 同框对话框 / 双键框内步进器），货架页本身不是成交
- 余额读取失败就不点花钱键，不用默认值代替
- 未授权的购买键拒这一发，不停整轮
- 派发类操作按游戏日记账，同一天不会重复执行
- 危险锚类（如走格子部署菜单的「解除」）只用来认局面，闸层拒绝任何指向它的点击

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

- `http://127.0.0.1:8000/v2/`：日常控制台。选要跑的 flow、配悬赏分支 / 学院 /
  活动关号 / 咖啡厅邀请对象 / 课程表找人，看运行日志；金钱项只读，前端放不宽
- `http://127.0.0.1:8000/dashboard.html`：标注控制台。从模拟器采集新帧、预标、
  逐帧标注，训练数据的采集和标注都在这里完成

`windows_app/` 是 WPF 桌面壳：拉起后端、内嵌控制台（WebView2）、可选悬浮层，
最小化到托盘。不装它也能用浏览器直接开上面两个地址。

### 跑日常

```bash
py -m routing_v2 probe          # 看一帧：页面身份 + 全部检出
py -m routing_v2 step --go      # 单步：看该点什么，--go 才真点并验到达
py -m routing_v2 run --auto     # 自主跑；不带 --auto 则每一发都要人放行
py -m routing_v2 run --flows event,bounty
py -m routing_v2 health         # cls 健康度 + 死判据审计
```

每个玩法是 `routing_v2/flow/` 下的一个独立 flow，页面身份由 UI 检测器逐帧确认，
点击带「点完该出现什么」的契约，未兑现前不发下一发；识别不到预期页面就停在原地
报告，不会盲点。当前登记的 flow：

| flow | 做什么 |
|---|---|
| free_pack / club / craft / shop / mail / daily_mission | 免费包、社团、快速制造、信用点商店与大赛商店、邮件、每日任务 |
| cafe / schedule | 咖啡厅收益、邀请、摸头；课程表按房间上课，可指定找人 |
| bounty / jfd | 悬赏通缉、学院交流会的票券扫荡，分支 / 学院由前端选 |
| event / event_shop | 活动首通与加成刷关、活动商店按产出推算买什么，台账按活动期记账 |
| arena | 战术大赛把票打完 |
| campaign | 任務推图：集中指挥走格子，按答案逐回合落子，含传送门、切区、部署侧多队感知 |
| story_mining / momotalk | 剧情挖矿、MomoTalk 好感度，默认关，前端开 |

### 跑战斗

```bash
py -u scripts/bot_play_quest.py 10 11 12
```

参数是要打的关卡编号。战斗链路走 scrcpy 视频流取帧，行为树决定出牌时机，
角色和技能卡状态由战斗检测器实时给出。

### 训练与上线

迭代一个 UI 模型的完整循环：

1. 控制台里采集新帧并标注；只在加新类时才复制出 `data/raw_images/_classes_next.txt` 追加
2. 构建数据集：`py -X utf8 scripts/build_ui_v2.py --clean`（加了新类时带 `--master _classes_next.txt`）
3. 训练：`py -X utf8 scripts/train_yolo26.py ui_yolo26m_v21`（超参照抄上一版，只变数据；
   中断后 `--resume`，显存踩线时 `--resume --batch 10`）
4. 重生成类语义表：`py -X utf8 scripts/gen_cls_corpus.py`
5. 上线：冻结权重，把 `data/model_registry.json` 里对应条目的 `active` 指向新版本，
   再把 `_classes_next.txt` 转正为 `_classes.txt`

评估和对比脚本在 `scripts/` 下，训练产物默认写到仓库外的缓存目录。

## 目录结构

    routing_v2/          当前架构：感知 / 状态 / 动作 / flow 四层
      percept/           抓帧、检测、读数
      state/             页面识别、类表词汇
      act/               动作、推进闸、金钱闸、记账
      flow/              每个玩法一个 flow
      app/               命令行、FastAPI 路由、日常控制台页面
      config/            配置 schema 与默认值
      tests/             离线回归（合成帧驱动，`py -X utf8 routing_v2/tests/test_offline.py`）
    brain/               上一代架构，仍承担战斗链路
    server/              FastAPI 后端 + 标注控制台
    windows_app/         WPF 桌面壳（后端拉起、内嵌控制台、悬浮层、托盘）
    scripts/             训练 / 建集 / 评估 / 采集 / 类表审计工具
    data/
      model_registry.json   现役模型版本，单一真相源
      ui_cls_semantics.md   类语义表（自动生成）
      raw_images/           标注帧 + _classes.txt 主类表
      grid_answers/         走格子答案（由 BAAH 开源解法转换）

模型权重、数据集和 HF 缓存放在仓库之外的独立目录（已 gitignore），路径由
`HF_HOME` 和 `model_registry.json` 配置。checkout 很小，不含二进制。

## License and Disclaimer

仅供个人学习使用。仓库不分发任何游戏文件，素材保留在你自己的模拟器安装里。
本项目与 Yostar、Nexon、Bilibili、NetEase 均无关联。
