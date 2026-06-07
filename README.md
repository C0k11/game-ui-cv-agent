# Blue Archive Daily Assistant

Fully-automated daily routine and a real-time battle head-lock for *Blue Archive* — running entirely on a local Windows machine. No cloud, no game modification.

Navigation is **vision-first**. Every button, tab, popup and badge the bot acts on is located by a trained **YOLO class, never a hardcoded screen coordinate**, so the same logic survives window-size and DPI changes. OCR is scoped to numeric fields only (AP / tickets / counts). An explicit state machine drives each skill through MuMu Player 12 via DXcam capture + PostMessage / ADB input.

## At a Glance

| | |
|---|---|
| **Platform** | Windows 10 / 11, NVIDIA GPU |
| **Game runtime** | MuMu Player 12 |
| **Daily** | `DailyRoutine` (10 sub-skills) + `CampaignSweep` (bounty / arena) |
| **Vision** | YOLO26m UI (451-cls) + YOLO26x avatar (252-cls) + YOLO26n emoticon + YOLOv8n battle head |
| **OCR** | PP-OCRv4 fine-tuned on BA glyphs — numeric fields only |
| **Battle lock** | DXcam + ByteTrack + Kalman predict-correct (lead-aim) |
| **Tooling** | WebView2 launcher + annotation / synth dashboard |

## How It Works

```mermaid
flowchart LR
    A[MuMu Player 12<br/>Blue Archive] -->|DXcam| B[Pipeline tick]
    B --> C{Skill state machine}
    C -->|nav / click| UI[YOLO26m UI<br/>451 cls]
    C -->|character ID| AV[YOLO26x avatar<br/>252 cls]
    C -->|numbers| O[PP-OCRv4]
    C -->|head-pat| EM[YOLO26n emoticon]
    UI --> D[Action decision]
    AV --> D
    O --> D
    EM --> D
    D -->|PostMessage / ADB| A
    B -.async.-> J[(Trajectory<br/>screenshot + meta)]
    J -.mine + label.-> K[Dashboard]
    K -.build + train.-> AV
```

## Vision Stack

| Tier | Model | Job | Latency |
|---|---|---|---|
| **UI** (primary) | YOLO26m `ui_v5` (451-cls) | every button / tab / popup / badge → drives all nav + clicks | ~6 ms |
| Avatar ID | YOLO26x `fused_avatar_v6` (252-cls) | bbox + character ID in one pass + in-battle skill-card recognition (incl. grayed-out / charging) | ~10 ms |
| Numeric OCR | PP-OCRv4 BA-tuned | AP / ticket / count digits only | ~50 ms |
| Head-pat | YOLO26n `emoticon` | cafe head-pat bubble | ~2 ms |
| Battle lock | YOLOv8n `battle_heads` | single-class head at 60+ FPS | ~2 ms |

Active versions are resolved at runtime from `data/model_registry.json`, so shipping a model is a one-line `active` bump — and rolling back is just as fast. Each detector infers at its training `imgsz` (960 for the UI / avatar models). `cv2.matchTemplate` / HSV survive as cheap fallbacks for a few stable glyphs.

**Why pure-YOLO (OCR demoted):** the pipeline used to navigate by OCR text + template match, which broke on font rendering, localization and resolution. Disabling OCR for navigation forced every click path through a trained class — navigation is now resolution- and scale-independent, and a miss is an honest "class not detected" instead of a silent mis-click.

## Daily Skills

`DailyRoutine` runs ten sub-skills in order; each finds its target by class and clicks the returned box. Money paths are gated **structurally** (right column), never by trust in a single detection.

| # | Skill | Does | Money / fallback guard |
|---|---|---|---|
| 1 | BuyPyroxene | claim daily free pack | confirm **only if `免费` present**, else cancel |
| 2 | Club | check-in for AP | card miss → red-dot offset |
| 3 | Craft | claim + queue craft | "finish-now" ticket dialog → cancel |
| 4 | Shop | affordable credit buys | bail if pyroxene tab; buy only if balance ≥ reserve |
| 5 | Cafe | income / invite / head-pat | NAV miss → bottom-bar extrapolation |
| 6 | Schedule | lesson dispatch, favorite-first | ticket OCR=0 → exit; pyroxene in dialog → cancel |
| 7 | MomoTalk | clear unread bond chats | — |
| 8 | StoryMining | mine unplayed story nodes | battle node (SORTIE/SQUAD) → back, spend no AP |
| 9 | Mail | claim all rewards | entry clicked only from lobby |
| 10 | DailyMission | claim dailies (runs last) | unlocked only after the rest finish |

`CampaignSweep` enters the mission hub once and delegates to bounty / arena (+ event when active). Global popups (rewards, level-up, exit / disconnect dialogs) are dismissed once in the pipeline interceptor by class — a backable modal via 取消 / X, never ESC (ESC could confirm the exit-game dialog).

## Battle Lock

A Kalman **predict → correct** tracker over YOLOv8n head detections, rendered to a Win32 layered overlay. Three things give it the external-grade feel:

- **One-Euro smoothing** — heavy when the target is slow (no jitter), light when fast (no lag).
- **Predictive lead-aim** — the box is drawn where the target *will* be (`position + velocity × end-to-end latency`), hiding the ~30–50 ms capture→render lag. Clamped so a noisy velocity spike can't fling it; off by default for static UI overlays.
- **Velocity coast + ByteTrack rescue** — an unmatched track glides on its decayed velocity through a VFX flash that tanks confidence; the low-conf second stage re-acquires the moment the head reappears.

## Dashboard

A FastAPI + WebView2 app for running the bot and iterating models:

- **Agent / HUD** — profile, skill order, AP / favorites, dry-run toggle; live pipeline state (current skill, sub-state, last action reason).
- **Capture** — DXcam capture with split routing to `train` runs or per-purpose held-out val pools, so rare-class samples are never stolen from training.
- **Annotate** — YOLO / OCR labeling hardened for long sessions: 50-step undo/redo, cross-frame paste, LRU prefetch for 0-latency paging, loss-proof saves, find-by-class, and model-assisted prefill (`YOLO预填` overlays the active model on a whole run so weak classes get their first samples).
- **Synth Templates** — visual per-context slot editor (axis-aligned rect or free 4-point quad), ref-crop preview, augmentation anchors, bbox modes, live preview — the heart of avatar / skill-card data generation.
- **Trajectories** — per-tick replay (screenshot, OCR, YOLO, action, reason).

## Models & Iteration

What actually moved the needle, learned across UI v1→v5 and avatar v1→v4:

- **Spatial augmentation** (mosaic / copy_paste / scale / translate / hsv_v) breaks position- and background-dependence — the core fix for the early overfit where a handful of frames blown up ~200× by oversample made the model memorize backgrounds instead of elements.
- **`fliplr / flipud / degrees = 0`** for UI — left/right is semantic (左切换 ↔ 右切换); flipping corrupts labels.
- **mixup / copy_paste are toxic for fine-grained 252-class ID** — they blend identity features; removed for the avatar model.
- **Synthetic compositing** pastes a rare element / character onto hundreds of real backgrounds — diversity that duplication cannot provide.
- **The small held-out val lies** — it carried zero instances of the weak classes it was meant to measure, and picked the wrong checkpoint. Models ship from a frozen `last.pt` after a real-frame check, not from val-mAP alone.
- **Warm-start** from the previous best preserves learned identity features and roughly halves wall-clock.

**Active:** UI `ui_yolo26m_v5` · avatar `fused_avatar_yolo26x_v6` · emoticon `v26n` · battle `battle_heads`.

**Avatar v6** (shipped) extends the 252-head detector to in-battle **EX skill cards** — the bottom-row character cards, including the grayed-out *insufficient-cost* state with its clock-wipe charge sweep — built from a template-synth pipeline with multi-background backgrounds and domain-accurate gray/sector augmentation. On held-out real frames: skill-card recall **0.85** (vs the prior cards-capable run's 0.56) while non-battle recognition (cafe / formation) holds at mAP50 **0.99**. The shipped checkpoint is the *real-val peak*, picked on a manual val set rather than the synth-inflated nominal best (later epochs overfit the synthetic cards).

**In progress:**

- **Unified detector** — one YOLO26x covering UI + avatar + emoticon + skill cards, to retire the multi-model split once it beats each domain specialist.
- **Battle skill-card AI** — the detector now *sees* the cards (incl. gray/charging); a combat policy that *reads* them (cost-aware skill rotation) is the next layer.

## Quick Start

### Requirements

Windows 10/11 · Python 3.11+ · [MuMu Player 12](https://mumu.163.com/) running Blue Archive · NVIDIA GPU (RTX 3060+ for the battle lock; the daily pipeline is CPU-bound).

### Install

```powershell
git clone https://github.com/C0k11/blue-archive-assistant.git
cd blue-archive-assistant
pip install -r requirements.txt
```

### Run

```powershell
# Launcher (recommended): download GameSecretaryApp.exe from Releases and double-click.
# Terminal:
py -m uvicorn server.app:app --host 127.0.0.1 --port 8000
# then open http://127.0.0.1:8000/dashboard.html

# Battle-lock demo (--lead-ms ≈ end-to-end latency to predict ahead):
py scripts/battle_overlay_demo.py --fps 240 --conf 0.05 --lead-ms 40
```

### Train / iterate a model

Build data in the dashboard (Capture + Annotate), then:

```powershell
py scripts/build_fused_avatar_dataset.py        # avatar / skill-card dataset (synth + manual + neg)
py scripts/train_yolo26.py fused_avatar_26x_v4  # train a registered config
py scripts/eval_fused_avatar_report.py          # per-frame HTML eval
# ship: freeze weights, bump the active version in data/model_registry.json
```

## Repository Layout

```
ai-game-secretary/
├── brain/
│   ├── pipeline.py          # interceptors, model-registry resolve, async trajectory writer
│   └── skills/              # one module per skill
├── vision/                  # OCR, avatar matcher, YOLO wrappers
├── server/                  # FastAPI app + dashboard.html
├── scripts/                 # train / build / eval / battle-overlay
│   ├── train_yolo26.py
│   ├── build_fused_avatar_dataset.py
│   ├── build_ui_v2.py
│   ├── battle_overlay_demo.py
│   └── ocr_training/
├── data/
│   ├── model_registry.json  # active model versions (single source of truth)
│   ├── synth_templates/     # per-context synth JSON
│   ├── raw_images/          # labeled frames + _classes.txt master
│   └── captures/角色头像/   # wiki portrait refs
└── windows_app/             # .NET 8 WebView2 launcher
```

Models, datasets and the HF cache live outside the repo under `D:/Project/ml_cache/` (gitignored).

## License & Disclaimer

Personal / educational use only. No game files are redistributed; assets stay in your own MuMu installation. Not affiliated with Yostar / Nexon / Bilibili / NetEase.
