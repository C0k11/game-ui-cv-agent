import json
import os
import sys
import socket
import subprocess
import time
import hashlib
import threading
import ctypes
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_SERVER_STARTED_AT = time.time()

DASHBOARD_PATH = REPO_ROOT / "dashboard.html"
ANNOTATE_PATH = REPO_ROOT / "annotate.html"

CAPTURES_DIR = REPO_ROOT / "data" / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

RAW_IMAGES_DIR = REPO_ROOT / "data" / "raw_images"
RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

CHARACTERS_DIR = CAPTURES_DIR / "角色头像"
CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)

APP_CONFIG_PATH = REPO_ROOT / "data" / "app_config.json"

# DXcam capture state
_CAPTURE_LOCK = threading.Lock()
_CAPTURE_THREAD = None
_CAPTURE_RUNNING = False
_CAPTURE_STATUS = {"running": False, "frames": 0, "dataset": "", "error": ""}

OCR_CACHE_VERSION = 2

# ── Pipeline state ─────────────────────────────────────────────────────
_PIPELINE_LOCK = threading.Lock()
_PIPELINE = None          # brain.pipeline.DailyPipeline instance
_PIPELINE_THREAD = None   # background worker thread
_PIPELINE_RUNNING = False
_PIPELINE_STATUS = {"running": False, "error": "", "ticks": 0}
_LAST_PIPELINE_ERROR = ""
_DISPLAY_SYNC_HZ = 240.0
_INPUT_POLL_HZ = 8000.0
_TIMER_RES_ENABLED = False


def _enable_high_resolution_timer() -> None:
    global _TIMER_RES_ENABLED
    if _TIMER_RES_ENABLED:
        return
    try:
        ctypes.WinDLL("winmm", use_last_error=True).timeBeginPeriod(1)
        _TIMER_RES_ENABLED = True
    except Exception:
        pass


def _high_res_sleep(seconds: float) -> None:
    delay = float(seconds)
    if delay <= 0:
        return
    end_t = time.perf_counter() + delay
    if delay > 0.003:
        time.sleep(max(0.0, delay - 0.0015))
    while time.perf_counter() < end_t:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        if pid <= 0:
            return False
    except Exception:
        return False

    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
            kernel32.GetExitCodeProcess.restype = ctypes.c_bool
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            kernel32.CloseHandle.restype = ctypes.c_bool
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259

            h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            code = ctypes.c_ulong(0)
            ok = kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            try:
                kernel32.CloseHandle(h)
            except Exception:
                pass
            if not ok:
                return False
            return int(code.value) == STILL_ACTIVE
        except Exception:
            return False

    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _parent_watchdog() -> None:
    try:
        pid_str = os.environ.get("GAMESECRETARY_PARENT_PID") or "0"
        pid = int(pid_str)
    except Exception:
        pid = 0
    
    print(f"DEBUG: Watchdog init. Parent PID: {pid}", flush=True)
    
    if pid <= 0:
        return

    while True:
        try:
            if not _pid_alive(pid):
                print(f"DEBUG: Watchdog triggered. Parent {pid} gone. Exiting.", flush=True)
                try:
                    _stop_pipeline()
                except Exception:
                    pass
                try:
                    # Kill entire process group if possible
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgrp(), 9)
                    else:
                        os._exit(0)
                except Exception:
                    os._exit(0)
                break
        except Exception as e:
            print(f"DEBUG: Watchdog error: {e}", flush=True)
        time.sleep(2.0)


app = FastAPI()

# Print env vars for debugging
try:
    print(f"DEBUG: Backend starting. Env GAMESECRETARY_PARENT_PID={os.environ.get('GAMESECRETARY_PARENT_PID')}", flush=True)
    if os.environ.get("GAMESECRETARY_PARENT_PID"):
        t = threading.Thread(target=_parent_watchdog, daemon=True)
        t.start()
        print("DEBUG: Watchdog thread started", flush=True)
except Exception as e:
    print(f"DEBUG: Watchdog start failed: {e}", flush=True)


# ── Pipeline control ──────────────────────────────────────────────────

def _start_pipeline(*, payload: Dict[str, Any]) -> None:
    """Start the DailyPipeline in a background thread."""
    global _PIPELINE, _PIPELINE_THREAD, _PIPELINE_RUNNING, _PIPELINE_STATUS
    with _PIPELINE_LOCK:
        if _PIPELINE_RUNNING:
            return
        from brain.pipeline import DailyPipeline
        _PIPELINE = DailyPipeline()
        _PIPELINE.start()
        _PIPELINE_RUNNING = True
        _PIPELINE_STATUS = {"running": True, "error": "", "ticks": 0}

        window_title = str(payload.get("window_title") or "Blue Archive")
        step_sleep = float(payload.get("step_sleep_s") or 0.6)
        dry_run = bool(payload.get("dry_run") if payload.get("dry_run") is not None else True)

        _PIPELINE_THREAD = threading.Thread(
            target=_pipeline_worker,
            args=(window_title, step_sleep, dry_run),
            daemon=True,
        )
        _PIPELINE_THREAD.start()


def _stop_pipeline() -> None:
    """Stop the running pipeline."""
    global _PIPELINE, _PIPELINE_RUNNING
    with _PIPELINE_LOCK:
        _PIPELINE_RUNNING = False
        if _PIPELINE is not None:
            try:
                _PIPELINE.stop()
            except Exception:
                pass
            _PIPELINE = None
        _PIPELINE_STATUS["running"] = False


def _pipeline_worker(window_title: str, step_sleep: float, dry_run: bool) -> None:
    """Background thread: screenshot -> OCR -> skill decision -> execute action."""
    global _PIPELINE_RUNNING, _PIPELINE_STATUS
    try:
        from scripts.win_capture import (
            capture_client, find_window_by_title_substring,
            find_largest_visible_child,
        )
        import cv2
        import numpy as np

        _enable_high_resolution_timer()

        hwnd = find_window_by_title_substring(window_title)
        if not hwnd:
            _PIPELINE_STATUS["error"] = f"Window '{window_title}' not found"
            _PIPELINE_RUNNING = False
            _PIPELINE_STATUS["running"] = False
            return

        # For emulators (MuMu, etc.), capture from the render child window
        # so the toolbar/tabs are excluded and coordinates map correctly.
        render_hwnd = hwnd
        try:
            child = find_largest_visible_child(int(hwnd))
            if child:
                render_hwnd = int(child)
                _log_pipeline(f"Using render child hwnd={render_hwnd} (parent={hwnd})")
        except Exception:
            pass

        # Store top-level hwnd for SetForegroundWindow (child windows can't be foregrounded)
        _set_parent_hwnd(int(hwnd))

        # Start YOLO overlay on the render window
        _overlay = None
        try:
            from scripts.yolo_overlay import YoloOverlay
            _overlay = YoloOverlay(render_hwnd)
            _overlay.start()
            _log_pipeline(f"YOLO overlay started on render_hwnd={render_hwnd}")
        except Exception as e:
            _log_pipeline(f"YOLO overlay unavailable: {e}")

        _log_pipeline(f"Pipeline worker started. window='{window_title}' hwnd={hwnd} render={render_hwnd} sleep={step_sleep} dry_run={dry_run}")

        while _PIPELINE_RUNNING:
            pipe = None
            with _PIPELINE_LOCK:
                pipe = _PIPELINE
            if pipe is None or not pipe.is_running:
                break

            # 1. Capture screenshot from render child (PIL Image)
            try:
                pil_img = capture_client(render_hwnd)
            except OSError:
                # Handle invalid/stale hwnd — re-resolve
                render_hwnd = hwnd
                try:
                    new_hwnd = find_window_by_title_substring(window_title)
                    if new_hwnd:
                        hwnd = new_hwnd
                        child = find_largest_visible_child(int(hwnd))
                        render_hwnd = int(child) if child else int(hwnd)
                except Exception:
                    pass
                _high_res_sleep(0.25)
                continue

            if pil_img is None:
                _high_res_sleep(0.25)
                continue

            # Convert PIL -> numpy BGR for cv2
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Save to temp file for trajectory
            tmp_path = str(REPO_ROOT / "data" / "_pipeline_frame.jpg")
            cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # 2. Pipeline tick (use in-memory frame, skip re-reading file)
            action = pipe.tick_from_frame(frame, screenshot_path=tmp_path)
            action_type = action.get("action", "")
            reason = action.get("reason", "")
            _PIPELINE_STATUS["ticks"] = pipe._total_ticks
            _PIPELINE_STATUS["progress"] = pipe.progress

            _log_pipeline(f"tick={pipe._total_ticks} action={action_type} reason={reason}")

            # Update YOLO overlay with latest detections
            if _overlay and _overlay.is_alive:
                screen = pipe.last_screen
                if screen and screen.yolo_boxes:
                    _overlay.update(screen.yolo_boxes)
                else:
                    _overlay.update([])

            # 3. Execute action (unless dry_run)
            if not dry_run and action_type != "done":
                _execute_pipeline_action(action, render_hwnd, frame.shape[1], frame.shape[0])

            # 4. Sleep
            if action_type == "wait":
                wait_ms = action.get("duration_ms", 500)
                _high_res_sleep(max(1.0 / _DISPLAY_SYNC_HZ, wait_ms / 1000.0))
            else:
                _high_res_sleep(max(1.0 / _DISPLAY_SYNC_HZ, step_sleep))

    except Exception as e:
        _PIPELINE_STATUS["error"] = f"{type(e).__name__}: {e}"
        _log_pipeline(f"Pipeline worker error: {e}")
        traceback.print_exc()
    finally:
        if _overlay:
            try:
                _overlay.stop()
            except Exception:
                pass
        _PIPELINE_RUNNING = False
        _PIPELINE_STATUS["running"] = False
        _log_pipeline("Pipeline worker stopped.")


_dpi_set = False
_parent_hwnd: Optional[int] = None  # top-level window (for SetForegroundWindow)


def _set_parent_hwnd(hwnd: int) -> None:
    """Store the top-level parent hwnd for SetForegroundWindow."""
    global _parent_hwnd
    _parent_hwnd = hwnd


def _MAKELPARAM(x: int, y: int) -> int:
    """Pack two 16-bit values into a 32-bit LPARAM."""
    return (int(y) << 16) | (int(x) & 0xFFFF)


def _execute_pipeline_action(action: Dict[str, Any], hwnd: int, img_w: int, img_h: int) -> None:
    """Convert normalized pipeline action to real input.

    Uses PostMessage to the render child window for reliable emulator input.
    Falls back to SetCursorPos + mouse_event if PostMessage fails.
    """
    global _dpi_set
    import random
    from scripts.win_capture import get_client_rect_on_screen
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    _enable_high_resolution_timer()

    # Set DPI awareness once so coordinates are in physical pixels
    if not _dpi_set:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            pass
        _dpi_set = True

    action_type = action.get("action", "")

    # Bring top-level parent window to foreground
    fg_hwnd = _parent_hwnd if _parent_hwnd else hwnd
    try:
        user32.SetForegroundWindow(fg_hwnd)
    except Exception:
        pass
    _high_res_sleep(1.0 / _DISPLAY_SYNC_HZ)

    # Get client rect for coordinate conversion
    try:
        rect = get_client_rect_on_screen(hwnd)
        cw = rect.right - rect.left
        ch = rect.bottom - rect.top
    except Exception:
        rect = None
        cw = ch = 0

    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    WM_MOUSEMOVE = 0x0200
    MK_LBUTTON = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_WHEEL = 0x0800

    def _screen_xy(nx: float, ny: float):
        if not rect or cw <= 0 or ch <= 0:
            return None
        cx = int(nx * cw) + random.randint(-2, 2)
        cy = int(ny * ch) + random.randint(-2, 2)
        sx = rect.left + cx
        sy = rect.top + cy
        return cx, cy, sx, sy

    if action_type == "click":
        nx, ny = action.get("target", [0.5, 0.5])
        coords = _screen_xy(nx, ny)
        if coords:
            cx, cy, sx, sy = coords
            print(f"[Click] norm=({nx:.3f},{ny:.3f}) client=({cx},{cy}) cw={cw} ch={ch}")
            user32.SetCursorPos(sx, sy)
            _high_res_sleep(1.0 / _INPUT_POLL_HZ)
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            _high_res_sleep(max(1.0 / _INPUT_POLL_HZ, 1.0 / (_DISPLAY_SYNC_HZ * 2.0)))
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    elif action_type == "back":
        # Send Escape key via PostMessage
        WM_KEYDOWN = 0x0100
        WM_KEYUP = 0x0101
        VK_ESCAPE = 0x1B
        user32.PostMessageW(hwnd, WM_KEYDOWN, VK_ESCAPE, 0x00010001)
        _high_res_sleep(max(1.0 / _INPUT_POLL_HZ, 1.0 / _DISPLAY_SYNC_HZ))
        user32.PostMessageW(hwnd, WM_KEYUP, VK_ESCAPE, 0xC0010001)

    elif action_type == "swipe":
        frm = action.get("from", [0.5, 0.5])
        to = action.get("to", [0.5, 0.5])
        dur_ms = action.get("duration_ms", 400)
        coords1 = _screen_xy(frm[0], frm[1])
        coords2 = _screen_xy(to[0], to[1])
        if coords1 and coords2:
            cx1, cy1, sx1, sy1 = coords1
            cx2, cy2, sx2, sy2 = coords2
            steps = max(12, int((dur_ms / 1000.0) * _DISPLAY_SYNC_HZ))
            print(f"[Swipe] from=({frm[0]:.3f},{frm[1]:.3f}) to=({to[0]:.3f},{to[1]:.3f}) steps={steps}")

            user32.SetCursorPos(sx1, sy1)
            _high_res_sleep(1.0 / _INPUT_POLL_HZ)
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            for i in range(1, steps + 1):
                t = i / steps
                mx = int(sx1 + (sx2 - sx1) * t)
                my = int(sy1 + (sy2 - sy1) * t)
                user32.SetCursorPos(mx, my)
                _high_res_sleep(dur_ms / 1000.0 / steps)
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    elif action_type == "scroll":
        nx, ny = action.get("target", [0.5, 0.5])
        clicks = action.get("clicks", -3)
        coords = _screen_xy(nx, ny)
        if coords:
            _, _, sx, sy = coords
            user32.SetCursorPos(sx, sy)
            _high_res_sleep(1.0 / _DISPLAY_SYNC_HZ)
            user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(clicks * 120), 0)


def _log_pipeline(msg: str) -> None:
    """Append pipeline log message."""
    try:
        log_path = LOGS_DIR / "agent.out.log"
        with log_path.open("a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


# ── Utility functions ─────────────────────────────────────────────────

def _tail_text(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""

    data = data.replace(b"\x00", b"")
    parts = data.splitlines()[-max(1, lines) :]
    try:
        return b"\n".join(parts).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _write_state(name: str, state: Dict[str, Any]) -> None:
    try:
        (LOGS_DIR / name).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── Core routes ───────────────────────────────────────────────────────

@app.post("/api/v1/shutdown")
def shutdown_server() -> Dict[str, str]:
    def _do_exit():
        time.sleep(0.2)
        try:
            _stop_pipeline()
        except Exception:
            pass
        os._exit(0)
    threading.Thread(target=_do_exit, daemon=True).start()
    return {"status": "shutting_down"}


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return "<html><head><meta http-equiv='refresh' content='0; url=/dashboard.html'></head><body></body></html>"


@app.get("/dashboard.html")
def dashboard() -> FileResponse:
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(
        str(DASHBOARD_PATH),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/annotate.html")
def annotate() -> FileResponse:
    if not ANNOTATE_PATH.exists():
        raise HTTPException(status_code=404, detail="annotate.html not found")
    return FileResponse(
        str(ANNOTATE_PATH),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/v1/status")
def status() -> Dict[str, Any]:
    pipeline_running = _PIPELINE_RUNNING
    last_error = _PIPELINE_STATUS.get("error", "")
    if not pipeline_running and not last_error and _LAST_PIPELINE_ERROR:
        last_error = _LAST_PIPELINE_ERROR

    progress = None
    pipe = None
    with _PIPELINE_LOCK:
        pipe = _PIPELINE
    if pipe is not None:
        try:
            progress = pipe.progress
        except Exception:
            pass

    return {
        "server_pid": os.getpid(),
        "server_started_at": round(_SERVER_STARTED_AT, 3),
        "agent_running": pipeline_running,
        "agent_type": "pipeline",
        "pipeline_status": _PIPELINE_STATUS,
        "pipeline_progress": progress,
        "last_error": last_error,
    }


# ── Capture path helpers ──────────────────────────────────────────────

def _safe_capture_path(rel: str) -> Path:
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if not rel:
        raise HTTPException(status_code=400, detail="path is required")

    p = (CAPTURES_DIR / rel).resolve()
    base = CAPTURES_DIR.resolve()
    if not str(p).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid path")
    return p


def _ocr_cache_path(rel: str) -> Path:
    key = f"v{OCR_CACHE_VERSION}:{rel}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    cache_dir = CAPTURES_DIR / "_cache" / "ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}.json"


# ── Config / Favorites ────────────────────────────────────────────────

@app.get("/api/v1/config/favorites")
def get_favorites() -> Dict[str, Any]:
    if not APP_CONFIG_PATH.exists():
        return {"target_favorites": []}
    try:
        data = json.loads(APP_CONFIG_PATH.read_text("utf-8"))
        return {"target_favorites": data.get("target_favorites", [])}
    except Exception:
        return {"target_favorites": []}


@app.post("/api/v1/config/favorites")
def set_favorites(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = {}
    if APP_CONFIG_PATH.exists():
        try:
            data = json.loads(APP_CONFIG_PATH.read_text("utf-8"))
        except Exception:
            pass
    data["target_favorites"] = payload.get("target_favorites", [])
    APP_CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
    return {"status": "ok"}


# ── Characters ────────────────────────────────────────────────────────

@app.get("/api/v1/characters/avatars")
def list_character_avatars() -> Dict[str, Any]:
    if not CHARACTERS_DIR.exists():
        return {"avatars": []}
    avatars = sorted([p.name for p in CHARACTERS_DIR.glob("*.png")])
    return {"avatars": avatars}


# ── Captures ──────────────────────────────────────────────────────────

@app.get("/api/v1/captures/sessions")
def list_sessions() -> Dict[str, Any]:
    sessions: List[Dict[str, Any]] = []
    try:
        for d in sorted(CAPTURES_DIR.glob("session_*"), key=lambda x: x.name, reverse=True):
            if not d.is_dir():
                continue
            png_count = 0
            try:
                png_count = len(list(d.glob("*.png")))
            except Exception:
                png_count = 0
            sessions.append({"name": d.name, "png_count": png_count})
    except Exception:
        pass
    return {"sessions": sessions}


@app.get("/api/v1/captures/images")
def list_images(session: str = Query(...)) -> Dict[str, Any]:
    session = os.path.basename((session or "").strip())
    d = _safe_capture_path(session)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="session not found")

    images = sorted([p.name for p in d.glob("*.png")])
    return {"images": images}


@app.get("/api/v1/captures/image")
def get_image(path: str = Query(...)) -> FileResponse:
    p = _safe_capture_path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(p))


# ── Annotations ───────────────────────────────────────────────────────

def _labels_path_for_image(rel: str) -> Path:
    rel = rel.replace("\\", "/").lstrip("/")
    parts = rel.split("/")
    if parts and parts[0].startswith("session_"):
        return _safe_capture_path(parts[0]) / "labels.jsonl"
    return CAPTURES_DIR / "labels.jsonl"


@app.get("/api/v1/annotations")
def get_annotations(image: str = Query(...)) -> Dict[str, Any]:
    rel = (image or "").replace("\\", "/")
    p = _labels_path_for_image(rel)
    items: List[Dict[str, Any]] = []
    if p.exists():
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("image") == rel:
                    items.append(obj)
        except Exception:
            pass
    return {"image": rel, "items": items}


@app.post("/api/v1/annotations/add")
def add_annotation(payload: Dict[str, Any]) -> Dict[str, Any]:
    image = str(payload.get("image") or "").replace("\\", "/")
    label = str(payload.get("label") or "").strip()
    text = str(payload.get("text") or "").strip()
    bbox = payload.get("bbox")
    typ = str(payload.get("type") or "").strip() or "ocr"

    if not image or not label or bbox is None:
        raise HTTPException(status_code=400, detail="image/label/bbox required")

    _safe_capture_path(image)
    p = _labels_path_for_image(image)
    p.parent.mkdir(parents=True, exist_ok=True)

    rec = {
        "image": image,
        "label": label,
        "text": text,
        "bbox": bbox,
        "type": typ,
        "ts": time.time(),
    }
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to write label: {e}")

    return {"ok": True}


@app.get("/api/v1/annotations/index")
def annotations_index(session: str = Query(...)) -> Dict[str, Any]:
    session = os.path.basename((session or "").strip())
    if not session:
        raise HTTPException(status_code=400, detail="session is required")

    d = _safe_capture_path(session)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="session not found")

    labels_path = d / "labels.jsonl"
    labeled: Set[str] = set()
    if labels_path.exists():
        try:
            for line in labels_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                rel = str(obj.get("image") or "").replace("\\", "/")
                if rel.startswith(session + "/"):
                    labeled.add(rel.split("/", 1)[1])
        except Exception:
            pass

    return {"session": session, "labeled": sorted(labeled)}


@app.get("/api/v1/captures/next_unlabeled")
def next_unlabeled(
    session: str = Query(...),
    after: str = Query(""),
) -> Dict[str, Any]:
    session = os.path.basename((session or "").strip())
    if not session:
        raise HTTPException(status_code=400, detail="session is required")

    d = _safe_capture_path(session)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="session not found")

    after = os.path.basename((after or "").strip())
    images = sorted([p.name for p in d.glob("*.png")])

    labeled_res = annotations_index(session=session)
    labeled = set(labeled_res.get("labeled") or [])

    start_idx = 0
    if after and after in images:
        start_idx = images.index(after) + 1

    for name in images[start_idx:]:
        if name not in labeled:
            return {"session": session, "filename": name, "image": f"{session}/{name}"}

    for name in images[:start_idx]:
        if name not in labeled:
            return {"session": session, "filename": name, "image": f"{session}/{name}"}

    return {"session": session, "filename": None, "image": None}


# ── Dataset & Label Editor APIs ──────────────────────────────────────────

def _safe_dataset_path(name: str) -> Path:
    name = os.path.basename((name or "").strip())
    if not name:
        raise HTTPException(status_code=400, detail="dataset name required")
    p = (RAW_IMAGES_DIR / name).resolve()
    if not str(p).startswith(str(RAW_IMAGES_DIR.resolve())):
        raise HTTPException(status_code=400, detail="invalid dataset name")
    return p


@app.get("/api/v1/datasets")
def list_datasets() -> Dict[str, Any]:
    datasets = []
    for d in sorted(RAW_IMAGES_DIR.iterdir()):
        if not d.is_dir():
            continue
        # Count jpg files (may be in root or /frames subfolder)
        jpg_count = len(list(d.glob("*.jpg")))
        frames_dir = d / "frames"
        if frames_dir.is_dir():
            jpg_count += len(list(frames_dir.glob("*.jpg")))
        if jpg_count == 0:
            continue
        datasets.append({"name": d.name, "image_count": jpg_count})
    return {"datasets": datasets}


@app.get("/api/v1/datasets/images")
def list_dataset_images(dataset: str = Query(...)) -> Dict[str, Any]:
    d = _safe_dataset_path(dataset)
    if not d.exists():
        raise HTTPException(status_code=404, detail="dataset not found")
    # Images may be in root or /frames subfolder
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    images = sorted([p.name for p in img_dir.glob("*.jpg")])
    # Load classes
    classes_file = img_dir / "classes.txt"
    classes = []
    if classes_file.exists():
        classes = [c for c in classes_file.read_text(encoding="utf-8").strip().split("\n") if c.strip()]
    # Load labels per image
    items = []
    for img_name in images:
        label_path = img_dir / Path(img_name).with_suffix(".txt")
        labels = []
        if label_path.exists():
            for line in label_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append({
                        "cls": int(parts[0]),
                        "xc": float(parts[1]), "yc": float(parts[2]),
                        "w": float(parts[3]), "h": float(parts[4]),
                    })
        items.append({"img": img_name, "labels": labels})
    return {"dataset": dataset, "classes": classes, "images": items}


@app.get("/api/v1/datasets/image")
def get_dataset_image(dataset: str = Query(...), filename: str = Query(...)) -> FileResponse:
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    filename = os.path.basename(filename)
    p = img_dir / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(p), media_type="image/jpeg")


@app.post("/api/v1/datasets/save_labels")
def save_dataset_labels(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload.get("dataset") or "")
    img_name = str(payload.get("img") or "")
    label_text = str(payload.get("labels") or "")
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    label_path = img_dir / Path(os.path.basename(img_name)).with_suffix(".txt")
    label_path.write_text(label_text, encoding="utf-8")
    return {"ok": True}


@app.post("/api/v1/datasets/add_class")
def add_dataset_class(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload.get("dataset") or "")
    new_name = str(payload.get("name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="class name required")
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    classes_file = img_dir / "classes.txt"
    existing = []
    if classes_file.exists():
        existing = [c for c in classes_file.read_text(encoding="utf-8").strip().split("\n") if c.strip()]
    if new_name not in existing:
        existing.append(new_name)
        classes_file.write_text("\n".join(existing) + "\n", encoding="utf-8")
    return {"ok": True, "id": len(existing) - 1}


@app.post("/api/v1/datasets/delete_image")
def delete_dataset_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload.get("dataset") or "")
    img_name = str(payload.get("img") or "")
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    fname = os.path.basename(img_name)
    img_path = img_dir / fname
    label_path = img_dir / Path(fname).with_suffix(".txt")
    deleted = []
    if img_path.exists():
        img_path.unlink()
        deleted.append(fname)
    if label_path.exists():
        label_path.unlink()
        deleted.append(label_path.name)
    return {"ok": True, "deleted": deleted}


@app.post("/api/v1/datasets/florence_suggest")
def dataset_florence_suggest(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload.get("dataset") or "")
    img_name = str(payload.get("img") or "")
    labels = [str(x).strip() for x in (payload.get("labels") or []) if str(x).strip()]
    limit = max(1, min(int(payload.get("limit") or 24), 64))
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    img_path = img_dir / Path(os.path.basename(img_name))
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    if not labels:
        classes_file = img_dir / "classes.txt"
        if classes_file.exists():
            labels = [c.strip() for c in classes_file.read_text(encoding="utf-8").splitlines() if c.strip()]
    labels = labels[:24]
    if not labels:
        raise HTTPException(status_code=400, detail="no classes available for Florence suggestions")

    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = float((ix2 - ix1) * (iy2 - iy1))
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        denom = area_a + area_b - inter
        if denom <= 0:
            return 0.0
        return inter / denom

    try:
        import cv2

        img = cv2.imread(str(img_path))
        if img is None:
            raise HTTPException(status_code=400, detail="cannot read image")
        h, w = img.shape[:2]
        from vision.florence_vision import get_florence_vision

        raw = get_florence_vision().suggest_labels(str(img_path), labels)
        kept: List[Dict[str, Any]] = []
        for item in raw:
            bbox = item.get("bbox") or []
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if any(item.get("label") == old.get("label") and _iou(box, old.get("bbox") or [0, 0, 0, 0]) > 0.6 for old in kept):
                continue
            kept.append({
                "label": str(item.get("label") or ""),
                "score": float(item.get("score") or 1.0),
                "bbox": box,
            })
            if len(kept) >= limit:
                break

        suggestions = []
        for item in kept:
            x1, y1, x2, y2 = item["bbox"]
            suggestions.append({
                "label": item["label"],
                "score": item["score"],
                "x1": x1 / w,
                "y1": y1 / h,
                "x2": x2 / w,
                "y2": y2 / h,
                "xc": ((x1 + x2) / 2.0) / w,
                "yc": ((y1 + y2) / 2.0) / h,
                "w": (x2 - x1) / w,
                "h": (y2 - y1) / h,
                "bbox": [x1, y1, x2, y2],
            })
        return {"ok": True, "suggestions": suggestions, "count": len(suggestions), "labels_used": labels}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": str(e), "suggestions": [], "count": 0, "labels_used": labels}


# ── OCR API ───────────────────────────────────────────────────────────

_OCR_ENGINE = None
_OCR_LOCK = threading.Lock()

def _get_ocr():
    global _OCR_ENGINE
    with _OCR_LOCK:
        if _OCR_ENGINE is None:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        return _OCR_ENGINE

@app.get("/api/v1/datasets/ocr")
def dataset_ocr(dataset: str = Query(...), filename: str = Query(...)):
    """Run RapidOCR on a single image, return text boxes with pixel + normalized coords."""
    img_path = (RAW_IMAGES_DIR / dataset / filename).resolve()
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise HTTPException(400, "Cannot read image")
        ocr = _get_ocr()
        result, _ = ocr(img)
        boxes = []
        h, w = img.shape[:2]
        if result:
            for line in result:
                pts, text, conf = line
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                px1, py1 = int(min(xs)), int(min(ys))
                px2, py2 = int(max(xs)), int(max(ys))
                boxes.append({
                    "text": text,
                    "confidence": round(float(conf), 3),
                    "x1": px1 / w, "y1": py1 / h,
                    "x2": px2 / w, "y2": py2 / h,
                    "px1": px1, "py1": py1, "px2": px2, "py2": py2,
                })
        return {"ok": True, "boxes": boxes, "count": len(boxes),
                "image_w": w, "image_h": h}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": str(e), "boxes": [], "count": 0}


@app.post("/api/v1/datasets/open_folder")
def dataset_open_folder(payload: Dict[str, Any]):
    """Open the dataset folder in the system file explorer."""
    dataset = payload.get("dataset", "")
    ds_dir = RAW_IMAGES_DIR / dataset
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")
    subprocess.Popen(["explorer", str(ds_dir.resolve())])
    return {"ok": True, "path": str(ds_dir.resolve())}


@app.post("/api/v1/datasets/ocr/save")
def dataset_ocr_save(payload: Dict[str, Any]):
    """Save OCR results for an image to the dataset's ocr_results.json."""
    dataset = payload.get("dataset", "")
    filename = payload.get("filename", "")
    boxes = payload.get("boxes", [])
    image_w = payload.get("image_w", 0)
    image_h = payload.get("image_h", 0)
    ds_dir = RAW_IMAGES_DIR / dataset
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")
    ocr_path = ds_dir / "ocr_results.json"
    data = {}
    if ocr_path.exists():
        try:
            data = json.loads(ocr_path.read_text("utf-8"))
        except Exception:
            data = {}
    data[filename] = {
        "image_w": image_w, "image_h": image_h,
        "boxes": boxes,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ocr_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    total_images = len(data)
    total_texts = sum(len(v.get("boxes", [])) for v in data.values())
    return {"ok": True, "total_images": total_images, "total_texts": total_texts}


@app.post("/api/v1/datasets/ocr/batch")
def dataset_ocr_batch(payload: Dict[str, Any]):
    """Batch scan all images in a dataset with OCR, save results."""
    dataset = payload.get("dataset", "")
    ds_dir = RAW_IMAGES_DIR / dataset
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")
    import cv2
    ocr = _get_ocr()
    ocr_path = ds_dir / "ocr_results.json"
    data = {}
    if ocr_path.exists():
        try:
            data = json.loads(ocr_path.read_text("utf-8"))
        except Exception:
            data = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted(f for f in ds_dir.iterdir() if f.suffix.lower() in exts)
    scanned = 0
    for img_path in imgs:
        fname = img_path.name
        if fname in data:
            continue  # skip already scanned
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        result, _ = ocr(img)
        boxes = []
        if result:
            for line in result:
                pts, text, conf = line
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                px1, py1 = int(min(xs)), int(min(ys))
                px2, py2 = int(max(xs)), int(max(ys))
                boxes.append({
                    "text": text, "confidence": round(float(conf), 3),
                    "x1": px1 / w, "y1": py1 / h,
                    "x2": px2 / w, "y2": py2 / h,
                    "px1": px1, "py1": py1, "px2": px2, "py2": py2,
                })
        data[fname] = {
            "image_w": w, "image_h": h,
            "boxes": boxes,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        scanned += 1
    ocr_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    total_images = len(data)
    total_texts = sum(len(v.get("boxes", [])) for v in data.values())
    return {"ok": True, "scanned": scanned, "total_images": total_images,
            "total_texts": total_texts}


# ── Screen Capture APIs (BitBlt) ──────────────────────────────────────

def _capture_worker(dataset_name: str, interval: float, window_title: str):
    """Capture game window via BitBlt (same method as mumu_runner.py).

    Previous implementation used DXcam which had DPI scaling issues with
    MuMu emulator, causing cropped/misaligned screenshots. BitBlt captures
    the correct full game client area regardless of DPI settings.
    """
    global _CAPTURE_RUNNING, _CAPTURE_STATUS
    try:
        import cv2
        import numpy as np
        from scripts.win_capture import (
            find_window_by_title_substring,
            find_largest_visible_child,
            capture_client,
        )

        hwnd = find_window_by_title_substring(window_title)
        if not hwnd:
            _CAPTURE_STATUS["error"] = f"Window '{window_title}' not found"
            _CAPTURE_STATUS["running"] = False
            _CAPTURE_RUNNING = False
            return

        # For MuMu, the game renders in the largest visible child window
        child = find_largest_visible_child(hwnd)
        target_hwnd = child if child else hwnd

        out_dir = RAW_IMAGES_DIR / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _CAPTURE_STATUS["dataset"] = dataset_name
        _CAPTURE_STATUS["error"] = ""

        count = 0
        while _CAPTURE_RUNNING:
            t0 = time.time()
            try:
                pil_img = capture_client(target_hwnd)
                frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                fp = out_dir / f"frame_{count:06d}.jpg"
                cv2.imwrite(str(fp), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                count += 1
                _CAPTURE_STATUS["frames"] = count
            except Exception as cap_err:
                _CAPTURE_STATUS["error"] = f"capture error: {cap_err}"
                # Re-find window in case it was resized/moved
                hwnd = find_window_by_title_substring(window_title)
                if hwnd:
                    child = find_largest_visible_child(hwnd)
                    target_hwnd = child if child else hwnd
            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    except Exception as e:
        _CAPTURE_STATUS["error"] = str(e)
    finally:
        _CAPTURE_STATUS["running"] = False
        _CAPTURE_RUNNING = False


@app.post("/api/v1/capture/start")
def capture_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _CAPTURE_THREAD, _CAPTURE_RUNNING, _CAPTURE_STATUS
    with _CAPTURE_LOCK:
        if _CAPTURE_RUNNING:
            return {"ok": False, "error": "already running", "status": _CAPTURE_STATUS}
        interval = float(payload.get("interval", 0.5))
        window_title = str(payload.get("window_title", "MuMu"))
        from datetime import datetime
        ds_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        _CAPTURE_RUNNING = True
        _CAPTURE_STATUS = {"running": True, "frames": 0, "dataset": ds_name, "error": ""}
        _CAPTURE_THREAD = threading.Thread(
            target=_capture_worker,
            args=(ds_name, interval, window_title),
            daemon=True,
        )
        _CAPTURE_THREAD.start()
    return {"ok": True, "status": _CAPTURE_STATUS}


@app.post("/api/v1/capture/stop")
def capture_stop() -> Dict[str, Any]:
    global _CAPTURE_RUNNING
    _CAPTURE_RUNNING = False
    return {"ok": True, "status": _CAPTURE_STATUS}


@app.get("/api/v1/capture/status")
def capture_status_api() -> Dict[str, Any]:
    return {"status": _CAPTURE_STATUS}


# ── Game launch ───────────────────────────────────────────────────────

def _maybe_launch_game(exe_path: str, wait_seconds: float = 5.0) -> str:
    """Launch game exe if not already running. Returns a short status message."""
    exe_path = (exe_path or "").strip()
    if not exe_path:
        return "no_exe_path"
    p = Path(exe_path)
    if not p.is_file():
        return f"exe_not_found: {exe_path}"
    exe_name = p.name.lower()
    # Check if the process is already running
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {p.name}", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        if exe_name in result.stdout.lower():
            return "already_running"
    except Exception:
        pass  # tasklist failed – try launching anyway
    # Launch the game
    try:
        subprocess.Popen(
            [str(p)],
            cwd=str(p.parent),
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    except Exception as e:
        return f"launch_error: {e}"
    # Give the game a moment to create its window
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    return "launched"


# ── Agent Start / Stop ────────────────────────────────────────────────

@app.post("/api/v1/start")
def api_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _LAST_PIPELINE_ERROR
    _LAST_PIPELINE_ERROR = ""
    # --- clear logs ---
    try:
        for fn in ("agent.out.log", "agent.err.log"):
            try:
                (LOGS_DIR / fn).write_text("", encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass
    # --- auto-launch game ---
    _game_launch_status = ""
    try:
        _game_launch_status = _maybe_launch_game(str(payload.get("game_exe_path") or ""))
    except Exception:
        _game_launch_status = f"exception: {traceback.format_exc(limit=5)}"
    try:
        log_path = LOGS_DIR / "agent.out.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[game_launch] {_game_launch_status}\n")
    except Exception:
        pass
    # --- stop any existing pipeline ---
    try:
        if _PIPELINE_RUNNING:
            _stop_pipeline()
    except Exception:
        pass
    # --- start new pipeline ---
    try:
        _start_pipeline(payload=payload)
    except Exception:
        _LAST_PIPELINE_ERROR = traceback.format_exc(limit=20)
    return status()


@app.post("/api/v1/stop")
def api_stop() -> Dict[str, Any]:
    _stop_pipeline()
    return status()


# ── Logs ──────────────────────────────────────────────────────────────

@app.get("/api/v1/logs")
def api_logs(
    name: str = Query(...),
    lines: int = Query(200, ge=1, le=2000),
) -> PlainTextResponse:
    safe = os.path.basename(name)
    path = LOGS_DIR / safe
    return PlainTextResponse(_tail_text(path, lines))
