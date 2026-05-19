# Blue Archive Daily Assistant

Fully automated daily routine and real-time battle target lock for *Blue Archive*. Runs entirely on your local machine with no cloud dependencies.

The automation pipeline is built around OCR, template matching, and explicit state machines, with a growing YOLO26 vision tier for cases where templates fail or scale poorly. It operates the game through MuMu Player 12 via DXcam screen capture and PostMessage / ADB input injection, requiring no game modifications. A Windows-native WebView2 launcher is provided for out-of-the-box use.

---

## Highlights

- Twenty-two composable daily skills covering the full routine (lobby cleanup, AP overflow protection, event farming, cafe, schedule, club, MomoTalk, shop, crafting, story cleanup, bounty, arena, joint firing drill, total assault, mail, daily tasks, pass rewards, AP planning, hard-mode farming, campaign push, and event boomerang).
- Visual schedule-avatar region editor on canvas: drag to move, resize by handles, right-click to add, Ctrl+C / Ctrl+V to copy a box and paste it elsewhere at the mouse position while preserving size, arrow keys to nudge, JSON export/import for cross-machine migration.
- Event farming budget control adapted from [reference](https://github.com/pur1fying/blue_archive_auto_script) `activity_sweep_times`: two parameters `event_max_rounds` and `event_ap_reserve` bound multi-round sweeping without per-event JSON stage tables.
- Custom OCR model fine-tuned on Blue Archive UI text (PP-OCRv4 base, Traditional/Simplified Chinese + English + Japanese mix), delivering a 20-point absolute gain in vocabulary accuracy.
- Multi-tier YOLO26 vision: emoticon (cafe head-pat bubble), static UI icons, avatar classifier (267 characters), and an in-training fused avatar detector that combines bbox localisation and character ID into a single forward pass.
- Real-time battle head lock at 240 Hz using DXcam capture, YOLOv8n, ByteTrack with freeze-and-predict rescue, rendered to a Win32 layered overlay window.
- Asynchronous trajectory writer: each tick's screenshot and metadata are enqueued and flushed on a background thread, so disk I/O never blocks the main perception loop.
- Annotation workspace with dedicated validation pools: dashboard's Capture page routes screenshots to `train` runs or to held-out `_val_<purpose>/frames/` pools, and the Annotate page groups datasets into Validation Pools / Recordings / Trajectories. Build scripts honour the pools so rare-class samples are never stolen from training.
- Windows launcher built on .NET 8 + WebView2 — double-click an `.exe` and the pipeline plus dashboard come up.

---

## Skill Matrix

| Skill | Function | Detection stack | Notes |
|------|----------|-----------------|-------|
| Lobby | Popup/announcement/notification cleanup, sign-in | OCR + templates | Handles update banners and `TOUCH TO START` |
| AP overflow guard | Dumps AP via event farming when AP >= 900 | OCR numeric parsing | Prevents cafe-settlement deadlock |
| EventActivity | Event story → mission → challenge → farming + shop | OCR + banner template + state machine | Auto-identifies event by period text (`auto_YYYYMMDD`) so each rotation gets its own progress bucket; story-tab smart-skip avoids redundant clicks; story-done falls through to mission instead of exiting; mission phase always direct-sortie (no quick-edit) so initial clears use the saved team — quick-edit + auto-bonus is reserved for the farming phase to update the sweep team for the rate-up stage; first-visible-node detection persists so resumed runs don't re-scan completed chapters |
| EventFarming | Normal / Hard / quest-type sweeps | OCR + state machine | `max_rounds` + `ap_reserve` budget |
| Cafe | Income collection, invitation tickets, head-pat | Template (primary) + YOLO26n emoticon (fallback) | `happy_face` template first; YOLO26n `emoticon` (mAP50 = 0.995) rescues misses; 1F left-to-right, 2F right-to-left scan |
| Schedule | Room assignment with favourite priority | OCR + `AvatarMatcher` (template + HSV); fused YOLO26m detector being trained | Region tuner on canvas; lobby-detect early exit; tuned `STAGE2_TOP_K=15` (≈32% faster, ≈18% more favourite hits than original); fused detector will replace AvatarMatcher once validated |
| Club | AP collection | OCR | |
| MomoTalk | Auto-reply to unread threads | OCR + state machine | Processes by unread count; auto-dialog / story skip |
| Shop | Free daily items and affordable purchases | OCR + state machine | Detects completion / refresh states |
| Craft | Claim finished items and queue quick craft | OCR + state machine | |
| StoryCleanup | Main / group / mini stories | OCR + state machine | Menu-driven skip with formation/battle handling |
| Bounty | Highest-difficulty sweep | OCR + state machine | Rotates three branches; ticket recheck |
| Arena | Reward claim and auto-battle | OCR + state machine | Cooldown wait, best-opponent selection |
| JointFiringDrill / TotalAssault | Auto-participation | OCR + state machine | |
| Mail / DailyTasks / PassReward | One-click claim | OCR | |
| ApPlanning | Free-AP and purchase strategy | OCR + numeric | Configurable purchase cap; premium-currency lockout |
| HardFarming / CampaignPush | Stage-specific / fallback sweeps | OCR + state machine | |
| BattleOverlay | Live head-box lock | YOLOv8n + ByteTrack | 240 Hz DXcam + Win32 transparent overlay |

---

## Vision Stack

| Tier | Component | Purpose | Latency |
|------|-----------|---------|---------|
| Primary | RapidOCR + fine-tuned recogniser | Full-screen Chinese / English / Japanese text | ~50 ms |
| Primary | `cv2.matchTemplate` | Template matching (`happy_face`, schedule avatars, UI icons) | <1 ms |
| Primary | HSV pixel analysis | Colour-state decisions (room occupancy, button enabled, checkmarks) | <1 ms |
| Battle | YOLOv8n `battle_heads.pt` | Combatant head detection | ~2 ms |
| Cafe fallback | YOLO26n `emoticon_yolo26n` | Head-pat bubble detection when templates miss | ~2 ms |
| Character classification | YOLO26n `avatar_cls_v2` (267 classes) | Crop-in / class-out classifier for cafe invite + schedule popup | ~5 ms |
| UI anchors (in progress) | YOLO26n `static_ui_v4` (143 classes) | Locate UI icons / popups / room cards / red-dot / coin badges for OCR-region anchoring | ~3 ms |
| One-shot avatar detect (in training) | YOLO26m `fused_avatar_yolo26m` (~235 classes) | Joint bbox + character ID, set to replace the 2-stage head_detector + classifier path | ~5 ms |
| Tracking | ByteTrack + EMA | Track association with freeze-and-predict rescue | <0.1 ms |

The daily pipeline favours templates and HSV decisions where they are cheap and exact; YOLO26 models are added selectively for sub-problems that templates cannot solve at scale (open-set character ID, sparse UI icons, multi-context avatar detection). The battle-lock path remains on YOLOv8n at 240 Hz.

### OCR fine-tuning

PP-OCRv4 was fine-tuned on Blue Archive mixed-script text (Traditional / Simplified Chinese, English, Japanese):

| Metric | Default PP-OCRv3 | BA fine-tuned | Delta |
|--------|------------------|---------------|-------|
| Vocabulary exact match | 35.8% | 55.8% | +20.0 |
| Full-sample exact match | 19.2% | 20.8% | +1.6 |

The five-step pipeline under `scripts/ocr_training/` crops from trajectories, synthesises augmented samples, trains, exports to ONNX, and evaluates. The output `data/ocr_model/ba_rec.onnx` is loaded automatically when the server starts.

---

## Event-farming Budget Control

The behaviour is adapted from the reference `activity_sweep` module's idea of "sweep N times with an AP floor", but without reference's per-event JSON stage tables. Two non-breaking parameters are added (defaults preserve legacy single-round behaviour):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `event_max_rounds` | `1` | Number of sweep cycles per `EventFarming` execution. When greater than one, after the sweep-result dialog clears, the stage is re-selected and swept again. |
| `event_ap_reserve` | `0` | Lower AP bound: once current AP falls to this value or below, the loop stops immediately. |

**Design contrast with reference.** reference maintains a per-event JSON file under `src/explore_task_data/activities/<EventName>.json` with explicit stage lists and AP costs, giving precise coverage of the current event at the price of frequent updates. This project takes a zero-configuration path instead: detect the active-event badge by OCR, scroll the stage list to the bottom, and press `MAX`. The trade-off is coarser granularity ("one MAX per round"). `event_max_rounds` extends that granularity to "N rounds of MAX with a reserve", which aligns semantically with reference's `sweep_times = [-1, 0.5, 3]`:

- `sweep_times = [-1]` (spend all AP) is approximated by `event_max_rounds = 10` and `event_ap_reserve = 0`.
- `sweep_times = [3]` maps to `event_max_rounds = 3` and `event_ap_reserve = 0`.
- `sweep_times = [0.5]` (half AP) maps to `event_max_rounds = 5` with `event_ap_reserve` set to roughly half of the starting AP.

Both parameters are exposed on the dashboard's Agent page next to `AP Purchase Limit` and are persisted per profile.

---

## Quick Start

### Requirements

- Windows 10 or 11
- Python 3.11 or later
- [MuMu Player 12](https://mumu.163.com/) running Blue Archive
- NVIDIA GPU (RTX 3060 or better recommended for battle lock; the daily pipeline runs on CPU)

### Install

```powershell
git clone https://github.com/C0k11/blue-archive-assistant.git
cd blue-archive-assistant
pip install -r requirements.txt
```

### Run the daily pipeline

Option 1 — Windows launcher (recommended): download `GameSecretaryApp.exe` from [Releases](https://github.com/C0k11/blue-archive-assistant/releases), double-click, and the launcher starts `uvicorn` and opens the dashboard in a WebView2 window.

Option 2 — Terminal:

```powershell
py -m uvicorn server.app:app --host 127.0.0.1 --port 8000
# then open http://127.0.0.1:8000/dashboard.html
```

Option 3 — Headless script:

```powershell
py mumu_runner.py
```

### Run the battle-lock demo

```powershell
py scripts/battle_overlay_demo.py --fps 240 --conf 0.05
```

---

## Dashboard

- **Home** — profile switching, skill ordering, AP / event budgets, favourite-character selection, dry-run toggle.
- **HUD** — live pipeline state (current skill, sub-state, tick count, AP reading, last action reason).
- **Roster** — canvas editor for the schedule avatar regions. Left-drag to move; drag a handle to resize; right-click empty space to add; `Del` to remove the selection; `Ctrl+C` / `Ctrl+V` to copy the selected box and paste it at the cursor while preserving width and height; `Ctrl+D` for in-place duplicate; arrow keys to nudge (hold `Shift` for a larger step); `Ctrl+S` to save. A collapsible JSON export/import is provided for cross-machine or cross-resolution migration.
- **Capture** — DXcam screen capture controls.  Two routing dropdowns:
  - **Split**: `Train (new run_*)` writes frames to a fresh `data/raw_images/run_<timestamp>/`, or `Val (held-out pool)` appends frames to the dedicated validation pool.
  - **Purpose** (visible only when Split = Val): `Fused Avatar` routes to `_val_fused/frames/`, `Static UI` routes to `_val_static_ui/frames/`.
  - Live destination hint updates as the dropdowns change so the user can verify where frames will land.
- **Annotate** — YOLO / OCR labelling workspace with rectangles, ellipses, and free-form brush polygons; right-drag to draw reliably on Windows via pointer events. Dataset dropdown is grouped into **⚖️ Validation Pools** (`_val_*` dirs), **📦 Recordings** (training data), and **🎬 Trajectories** (live ticks) for clear separation.
- **Trajectories** — replay of historical runs (screenshot, OCR, YOLO, action, reason) per tick.

---

## Battle Lock: ByteTrack with Freeze-and-Predict

```
YOLO detections (conf >= 0.05)
    |
    +-- high conf (>= 0.25) --> Stage 1: associate with existing tracks
    |                               unmatched -> create new track
    |
    +-- low  conf (< 0.25)  --> Stage 2: rescue unmatched tracks (VFX pass-through)
                                    unmatched -> discard

Track lifecycle:
    matched           -> EMA update (alpha = 0.85)
    unmatched         -> freeze in place (vx = vy = 0), conf *= 0.92 per frame
    5 frames unmatched-> delete

Intra-class NMS: merge same-class tracks with center_dist < 1.0
```

## Models

| Model | File | Training data | Metric | Role |
|-------|------|---------------|--------|------|
| Battle heads | `battle_heads.pt` | 52 hand-labelled frames + augmentation | mAP50 = 0.995 | BattleOverlay |
| Cafe head-pat bubble | `emoticon_yolo26n/best.pt` | 170 frames, YOLO26n, 1 class | mAP50 = 0.995 | Template fallback |
| Static UI elements | `static_ui_v4_yolo26n/best.pt` | 109 frames, 143 classes (UI icons, room cards, popups) | mAP50 = 0.339 (sparse-class limited) | UI anchor (in progress) |
| Avatar classifier v2 | `avatar_cls_v2_yolo26n/best.pt` | 3,295 crops, 267 classes, character-only classifier (no detection) | top1 = 95.91% on trajectory val | Used in current 2-stage Path-C; being replaced by fused detector |
| Fused avatar detector | `fused_avatar_yolo26m/best.pt` (in training) | 335 manual frames (5 UI contexts) + 486 synthetic composites + 79 negatives, 237 classes | TBD | One-shot avatar locate + classify, replaces 2-stage |
| OCR recogniser | `ba_rec.onnx` | Trajectory crops + synthesised text | vocabulary acc = 55.8% | Full pipeline |

---

## Performance Notes

- Trajectory writes are asynchronous: `brain/pipeline.py` drains a bounded `Queue(maxsize=64)` on a background thread, removing 10–50 ms of per-tick disk I/O from the main loop. JSON uses compact separators (about 35% smaller).
- Avatar matching caches resize results per `(name, h, w)` and circular masks per `(h, w)` in `vision/avatar_matcher.py`, avoiding redundant work across 9 rooms × 4 cells × N candidates per frame. The two-stage pipeline (cheap HSV histogram prefilter → masked `matchTemplate` on the top-K shortlist) defaults to `STAGE2_TOP_K=15`, which is both faster and more accurate than the original 40 because trimming look-alike non-favourite distractors lets genuine favourites win the open-set contest more often. `AVATAR_CROP_DIR` env var (or the `crop_dir=` constructor arg) swaps in alternative template sets without touching skill code.
- OCR results are cached within a single tick so multiple `find_text` calls in one pass share one OCR invocation.
- YOLO is lazily imported so the daily pipeline does not pay the load cost. Each model file is loaded on first use and held in process memory; the four active YOLO26 weights together fit comfortably under 200 MB of VRAM on an RTX 4090.
- Build scripts are intentionally destructive (`shutil.rmtree(OUT_ROOT)` on emit) — do not invoke them while a training run is reading the same dataset, or the worker `__getitem__` will raise `FileNotFoundError` and abort the run.

---

## Repository Layout

```
ai-game-secretary/
├── brain/
│   ├── pipeline.py              # Skill scheduler, global interceptors, async trajectory writer
│   └── skills/
│       ├── base.py              # ScreenState, BaseSkill, shared popup handling
│       ├── lobby.py
│       ├── event_farming.py     # max_rounds / ap_reserve budget control
│       ├── event_activity.py
│       ├── ap_planning.py
│       ├── cafe.py
│       ├── schedule.py          # AvatarMatcher-driven
│       ├── campaign_push.py
│       └── club, momo_talk, shop, craft, story_cleanup,
│           bounty, arena, jfd, total_assault, mail,
│           daily_tasks, pass_reward, farming
├── vision/
│   ├── engine.py                # OCR engine and per-tick screenshot cache
│   ├── avatar_matcher.py        # Template + HSV histogram matcher with resize/mask caches
│   ├── avatar_classifier.py     # YOLO26n avatar_cls_v2 wrapper with CN/EN bilingual logging
│   ├── template_matcher.py      # Multi-scale template matching
│   ├── event_banner_matcher.py  # Event lobby banner detection
│   ├── florence_vision.py       # Florence-2 vision helper (experimental)
│   ├── ocr_normalize.py         # OCR result normalisation (CN ↔ TW glyphs, full-width)
│   ├── io_utils.py              # Unicode-safe imread/imwrite
│   ├── yolo_detector.py         # YOLOv8/26 wrapper
│   └── window.py                # Window / region coordinate helpers
├── server/
│   ├── app.py                   # FastAPI backend, OCR service, capture worker with Split=train|val routing
│   └── dashboard.html           # Web dashboard (Home / HUD / Roster / Annotate / Trajectories / Capture)
├── scripts/
│   ├── box_tracker.py           # ByteTrack implementation
│   ├── yolo_overlay.py          # Win32 transparent overlay
│   ├── battle_overlay_demo.py
│   ├── collect_data.py          # DXcam capture utility (CLI alternative to dashboard)
│   ├── auto_label_*.py          # HSV auto-labelling tools (legacy emoticon flow)
│   ├── build_static_ui_dataset.py    # Build YOLO26 static UI dataset
│   ├── build_avatar_cls_v2_dataset.py # Build classifier dataset (crops + trajectory)
│   ├── build_head_detector_dataset.py # Build single-class avatar bbox dataset
│   ├── build_fused_avatar_dataset.py  # Build fused multi-class detector dataset (manual + synth + auto-negatives)
│   ├── trim_master_classes.py        # Master class registry trim (auto-backup)
│   ├── train_yolo26.py               # Unified YOLO26 training entry (multiple configs)
│   ├── train_headpat_yolo.py         # Legacy emoticon training
│   ├── eval_static_ui_report.py      # HTML eval: truth vs prediction overlay
│   ├── eval_avatar_cls_report.py     # HTML eval: per-tier classification
│   ├── eval_avatar_cls_schedule.py   # End-to-end Path C eval with --mode geom|head_det
│   ├── eval_avatar_cls_on_trajectories.py # Full trajectory audit
│   └── ocr_training/                 # Five-step OCR fine-tuning pipeline
├── windows_app/                 # .NET 8 WebView2 launcher
├── data/
│   ├── captures/                # Templates, character avatar refs (角色头像 + 角色头像_crop + 角色头像_crop_harvested_named)
│   ├── raw_images/              # Labelled frames; subdirs are datasets in dashboard:
│   │   ├── run_<timestamp>/     # Training data (any number of dirs)
│   │   ├── _val_fused/frames/   # Dedicated val pool for fused avatar detector
│   │   ├── _val_static_ui/frames/ # Dedicated val pool for static UI detector
│   │   └── _classes.txt         # Master class registry (UI 0..142 + characters 143..)
│   ├── trajectories/            # Live pipeline runs (tick_NNNN.jpg + .json)
│   ├── ocr_model/               # ba_rec.onnx
│   ├── ocr_training/            # OCR fine-tuning data
│   ├── student_name_map.json    # CN ↔ EN character name mapping
│   ├── avatar_cls_names_bilingual.json # 267-class bilingual labels
│   ├── app_config.json          # Profile store
│   ├── schedule_avatar_regions.json    # Schedule room regions
│   └── harvest_regions.json     # ROI for OCR harvest
├── mumu_runner.py               # Headless entry point
├── launch.py                    # Launcher helper
├── requirements.txt
└── README.md
```

Trained YOLO weights live outside the repo under `D:/Project/ml_cache/models/yolo/runs/`, and YOLO datasets under `D:/Project/ml_cache/models/yolo/dataset/`, to keep the source tree light.

---

## Trajectory System

Each call to `DailyPipeline.start()` creates `data/trajectories/run_<YYYYMMDD_HHMMSS>/` and writes asynchronously on every tick:

- `tick_NNNN.jpg` — screenshot
- `tick_NNNN.json` — OCR results, YOLO detections, current skill, sub-state, action, reason, and timestamp

The dashboard's Trajectories page replays historical runs. The Roster page automatically surfaces schedule-screen frames from trajectories as editing backgrounds.

---

## Training and Labelling

### Capture → Annotate → Build → Train (YOLO26 datasets)

The dashboard's Capture page now selects the destination split via two
dropdowns: **Split** (`Train` / `Val`) and **Purpose** (`Fused Avatar` /
`Static UI`).  Captured frames route automatically:

| Split | Destination | Used for |
|-------|-------------|----------|
| Train | `data/raw_images/run_<timestamp>/` | model training |
| Val (Fused Avatar) | `data/raw_images/_val_fused/frames/` | held-out validation for fused avatar detector |
| Val (Static UI) | `data/raw_images/_val_static_ui/frames/` | held-out validation for static UI detector |

The Annotate page groups datasets accordingly: **⚖️ Validation Pools**,
**📦 Recordings** (training data), and **🎬 Trajectories** (live ticks).
Class labels are the same across train and val — only the directory
location determines split membership.

### Dataset build pipeline

The build scripts read everything under `data/raw_images/` and emit a YOLO
dataset under `D:/Project/ml_cache/models/yolo/dataset/`:

```powershell
py scripts/build_fused_avatar_dataset.py     # detector for character avatars
py scripts/build_static_ui_dataset.py        # detector for UI icons / regions
py scripts/train_yolo26.py fused_avatar_26m  # train (config in train_yolo26.py)
py scripts/train_yolo26.py static_ui         # train
```

Split logic in both build scripts:

1. **Dedicated val pool (preferred)** — if `_val_<purpose>/frames/` exists
   with labelled `.txt` files, 100% of `run_*` frames go to train and the
   pool goes to val.  No samples are stolen from train.
2. **Stratified fallback** — when no val pool exists, build does a
   class-stratified split: rare classes (≤3 samples) are pinned to train so
   the model can still learn them, and more common classes get split 80/20.
3. **Auto-harvested negatives** — `find_negative_frames()` walks
   `data/trajectories/` and picks frames from contexts known to contain
   no character avatars (Mail / DailyTasks / CampaignSweep / Bounty enter /
   etc.), writing them with empty `.txt` labels.  Build mixes ~80 of these
   into train and ~20 into val automatically — no user labelling required.

### Older training pipelines

```powershell
# Capture data with DXcam (CLI variant)
py scripts/collect_data.py --interval 0.5

# Cafe head-pat: HSV auto-labelling + training (legacy emoticon flow)
py scripts/auto_label_headpat_v3.py
py scripts/train_headpat_yolo.py

# OCR fine-tuning
cd scripts/ocr_training
py 1_crop_from_trajectories.py
py 2_synth_data.py
py 3_train.py
py 4_export_onnx.py
py 5_eval.py
```

---

## Roadmap

Completed:

- Event-farming budget parameters adapted from reference `activity_sweep_times`.
- Roster canvas editor with Ctrl+C / Ctrl+V copy-paste and JSON migration.
- Asynchronous trajectory writer and avatar-matching caches.
- Windows launcher built on .NET 8 and WebView2.
- OCR fine-tuning on PP-OCRv4 (vocabulary +20 points).
- Annotation workspace with rotated boxes, ellipses, and free-form polygons.
- YOLO26n emoticon detector for cafe head-pat (mAP50 = 0.995).
- YOLO26n 267-class avatar classifier v2 (top1 = 95.91% on trajectory val).
- Dedicated validation-pool convention in build scripts (`_val_<purpose>/frames/`); dashboard Capture page routes screenshots to train or val pools automatically.
- Auto-harvested negative-sample integration: build pulls no-avatar frames from trajectory contexts (Mail/DailyTasks/Bounty/etc.) so the user never has to label backgrounds by hand.
- Class-stratified split fallback that pins rare classes (≤3 samples) to train, so single-sample classes are never lost to val.

In progress:

- Fused YOLO26m avatar detector (one-shot bbox + character ID) to replace the current 2-stage head_detector + classifier path. Training run uses ~335 manual frames across 5 UI contexts + 486 synthetic composites + 79 negatives, with 30 held-out user-labelled val frames.
- YOLO26n `static_ui_v4` improvement pass: targeted ~50-frame capture across top-bar HUD, shops, battle, campaign areas, squad editor, events, popups, and rewards to push the 127 sparse classes (<10 samples each) into the 10–30-sample range.
- Top-bar currency widgets: re-label icon-only currency classes (`体力`, `信用点`, `青辉石`, `课程表票`, etc.) as full `icon + digits` widgets to give YOLO-anchored OCR a guaranteed-overflow-safe ROI.

Planned:

- Iterative hard-negative mining: run trained fused detector on no-avatar contexts, collect false positives, retrain.
- Two-stage training (val-driven hyperparameters, then full-data retrain at the converged epoch count) to recover the 20% of samples held out for validation.
- TensorRT FP16 export for the fused detector to bring inference latency under 5 ms.
- Total Assault (Raid) script-based automation.
- Grand Assault automation.
- Support for additional emulators (BlueStacks, LDPlayer, NoxPlayer).
- VFX augmentation training to improve recall under battle occlusion.
- VLM-generated pseudo-labels for OCR distillation on rare glyphs.

---

## Credits

- [pur1fying/blue_archive_auto_script](https://github.com/pur1fying/blue_archive_auto_script) (reference). The `event_max_rounds` and `event_ap_reserve` budget parameters are inspired by reference's `activity_sweep_times` / `activity_sweep_task_number`. A local reading copy of the reference source lives under `study/ref/` and is excluded from version control.
- [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) — OCR inference backbone.
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8.
- [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) — tracking algorithm.

---

## License

MIT
