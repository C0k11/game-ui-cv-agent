# Blue Archive Daily Assistant

Fully automated daily routine and real-time battle target lock for *Blue Archive*. Runs entirely on your local machine with no cloud dependencies.

The automation pipeline is built around OCR, template matching, and explicit state machines, with YOLO reserved for real-time battle lock only. It operates the game through MuMu Player 12 via DXcam screen capture and PostMessage / ADB input injection, requiring no game modifications. A Windows-native WebView2 launcher is provided for out-of-the-box use.

---

## Highlights

- Twenty-two composable daily skills covering the full routine (lobby cleanup, AP overflow protection, event farming, cafe, schedule, club, MomoTalk, shop, crafting, story cleanup, bounty, arena, joint firing drill, total assault, mail, daily tasks, pass rewards, AP planning, hard-mode farming, campaign push, and event boomerang).
- Visual schedule-avatar region editor on canvas: drag to move, resize by handles, right-click to add, Ctrl+C / Ctrl+V to copy a box and paste it elsewhere at the mouse position while preserving size, arrow keys to nudge, JSON export/import for cross-machine migration.
- Event farming budget control adapted from [reference](https://github.com/pur1fying/blue_archive_auto_script) `activity_sweep_times`: two parameters `event_max_rounds` and `event_ap_reserve` bound multi-round sweeping without per-event JSON stage tables.
- Custom OCR model fine-tuned on Blue Archive UI text (PP-OCRv4 base, Traditional/Simplified Chinese + English + Japanese mix), delivering a 20-point absolute gain in vocabulary accuracy.
- Real-time battle head lock at 240 Hz using DXcam capture, YOLOv8n, ByteTrack with freeze-and-predict rescue, rendered to a Win32 layered overlay window.
- Asynchronous trajectory writer: each tick's screenshot and metadata are enqueued and flushed on a background thread, so disk I/O never blocks the main perception loop.
- Windows launcher built on .NET 8 + WebView2 — double-click an `.exe` and the pipeline plus dashboard come up.

---

## Skill Matrix

| Skill | Function | Detection stack | Notes |
|------|----------|-----------------|-------|
| Lobby | Popup/announcement/notification cleanup, sign-in | OCR + templates | Handles update banners and `TOUCH TO START` |
| AP overflow guard | Dumps AP via event farming when AP >= 900 | OCR numeric parsing | Prevents cafe-settlement deadlock |
| EventActivity | Event story → mission → challenge → farming + shop | OCR + banner template + state machine | Auto-identifies event by period text (`auto_YYYYMMDD`) so each rotation gets its own progress bucket; story-tab smart-skip avoids redundant clicks; story-done falls through to mission instead of exiting; mission phase always direct-sortie (no quick-edit) so initial clears use the saved team — quick-edit + auto-bonus is reserved for the farming phase to update the sweep team for the rate-up stage; first-visible-node detection persists so resumed runs don't re-scan completed chapters |
| EventFarming | Normal / Hard / quest-type sweeps | OCR + state machine | `max_rounds` + `ap_reserve` budget |
| Cafe | Income collection, invitation tickets, head-pat | Template (primary) + YOLO (fallback) | `happy_face` template first; 1F left-to-right, 2F right-to-left scan |
| Schedule | Room assignment with favourite priority | OCR + `AvatarMatcher` | Template + HSV histogram matching; region tuner on canvas; lobby-detect early exit; tuned `STAGE2_TOP_K=15` (≈32% faster, ≈18% more favourite hits than the original wide-shortlist sweep) |
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
| Primary | `cv2.matchTemplate` | Template matching (`happy_face`, avatars, UI icons) | <1 ms |
| Primary | HSV pixel analysis | Colour-state decisions (room occupancy, button enabled, checkmarks) | <1 ms |
| Battle | YOLOv8n `battle_heads.pt` | Combatant head detection | ~2 ms |
| Fallback | YOLOv8n `headpat.pt` | Cafe head-pat bubble when templates miss | ~2 ms |
| Tracking | ByteTrack + EMA | Track association with freeze-and-predict rescue | <0.1 ms |

The daily pipeline deliberately avoids heavy models; YOLO is used only for battle lock and the cafe fallback path.

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
- **Annotate** — YOLO / OCR labelling workspace with rectangles, ellipses, and free-form brush polygons; right-drag to draw reliably on Windows via pointer events.
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
| Cafe head-pat bubble | `headpat.pt` | 1,808 cafe frames (HSV auto-labelling) | mAP50 = 0.96 | Template fallback |
| OCR recogniser | `ba_rec.onnx` | Trajectory crops + synthesised text | vocabulary acc = 55.8% | Full pipeline |

---

## Performance Notes

- Trajectory writes are asynchronous: `brain/pipeline.py` drains a bounded `Queue(maxsize=64)` on a background thread, removing 10–50 ms of per-tick disk I/O from the main loop. JSON uses compact separators (about 35% smaller).
- Avatar matching caches resize results per `(name, h, w)` and circular masks per `(h, w)` in `vision/avatar_matcher.py`, avoiding redundant work across 9 rooms × 4 cells × N candidates per frame. The two-stage pipeline (cheap HSV histogram prefilter → masked `matchTemplate` on the top-K shortlist) defaults to `STAGE2_TOP_K=15`, which is both faster and more accurate than the original 40 because trimming look-alike non-favourite distractors lets genuine favourites win the open-set contest more often. `AVATAR_CROP_DIR` env var (or the `crop_dir=` constructor arg) swaps in alternative template sets without touching skill code.
- OCR results are cached within a single tick so multiple `find_text` calls in one pass share one OCR invocation.
- YOLO is lazily imported so the daily pipeline does not pay the load cost.

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
│   ├── template_matcher.py      # Multi-scale template matching
│   ├── yolo_detector.py         # YOLOv8 wrapper
│   └── window.py                # Window / region coordinate helpers
├── server/
│   ├── app.py                   # FastAPI backend and OCR service
│   └── dashboard.html           # Web dashboard (Home / HUD / Roster / Annotate / Trajectories)
├── scripts/
│   ├── box_tracker.py           # ByteTrack implementation
│   ├── yolo_overlay.py          # Win32 transparent overlay
│   ├── battle_overlay_demo.py
│   ├── collect_data.py          # DXcam capture utility
│   ├── auto_label_*.py          # HSV auto-labelling tools
│   ├── train_*.py               # YOLO training entry points
│   └── ocr_training/            # Five-step OCR fine-tuning pipeline
├── windows_app/                 # .NET 8 WebView2 launcher
├── data/
│   ├── captures/                # Template images and avatar library
│   ├── ocr_model/               # ba_rec.onnx
│   ├── app_config.json          # Profile store
│   └── schedule_avatar_regions.json
├── mumu_runner.py               # Headless entry point
├── launch.py                    # Launcher helper
├── requirements.txt
└── README.md
```

---

## Trajectory System

Each call to `DailyPipeline.start()` creates `data/trajectories/run_<YYYYMMDD_HHMMSS>/` and writes asynchronously on every tick:

- `tick_NNNN.jpg` — screenshot
- `tick_NNNN.json` — OCR results, YOLO detections, current skill, sub-state, action, reason, and timestamp

The dashboard's Trajectories page replays historical runs. The Roster page automatically surfaces schedule-screen frames from trajectories as editing backgrounds.

---

## Training and Labelling

```powershell
# Capture data with DXcam
py scripts/collect_data.py --interval 0.5

# Cafe head-pat: HSV auto-labelling + training
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

Planned:

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
