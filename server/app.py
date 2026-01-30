import json
import os
import socket
import subprocess
import time
import hashlib
import threading
import base64
import tempfile
import ctypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
import requests
from PIL import Image

from config import (
    HF_CACHE_DIR,
    LOCAL_VLM_DEVICE,
    LOCAL_VLM_MAX_NEW_TOKENS,
    LOCAL_VLM_MODEL,
    LOCAL_VLM_MODELS_DIR,
    MODELS_DIR,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_SERVER_STARTED_AT = time.time()

DASHBOARD_PATH = REPO_ROOT / "dashboard.html"
ANNOTATE_PATH = REPO_ROOT / "annotate.html"

CAPTURES_DIR = REPO_ROOT / "data" / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

OCR_CACHE_VERSION = 2
VLM_OCR_CACHE_VERSION = 1
LOCAL_VLM_OCR_CACHE_VERSION = 1

_VISION_LOCK = threading.Lock()
_VISION = None

_VLM_AGENT_LOCK = threading.Lock()
_VLM_AGENT = None


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
        pid = int(os.environ.get("GAMESECRETARY_PARENT_PID") or 0)
    except Exception:
        pid = 0
    if pid <= 0:
        return

    log_path = LOGS_DIR / "watchdog.log"
    def _log(msg):
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        except:
            pass

    _log(f"Watchdog started monitoring parent PID={pid}")

    while True:
        try:
            if not _pid_alive(pid):
                _log(f"Parent PID={pid} is dead. Terminating self.")
                try:
                    _stop_vlm_agent()
                except Exception as e:
                    _log(f"Error stopping agent: {e}")
                
                # Force kill self
                try:
                    os._exit(0)
                except Exception:
                    pass
                break
        except Exception as e:
            _log(f"Watchdog loop error: {e}")
        time.sleep(2.0)


app = FastAPI()

try:
    if os.environ.get("GAMESECRETARY_PARENT_PID"):
        threading.Thread(target=_parent_watchdog, daemon=True).start()
except Exception:
    pass


def _get_vision():
    global _VISION
    with _VISION_LOCK:
        if _VISION is None:
            from vision.florence_vision import FlorenceVision

            _VISION = FlorenceVision()
        return _VISION


def _get_vlm_agent():
    global _VLM_AGENT
    with _VLM_AGENT_LOCK:
        return _VLM_AGENT


def _start_vlm_agent(*, payload: Dict[str, Any]) -> None:
    global _VLM_AGENT
    with _VLM_AGENT_LOCK:
        from agent.vlm_policy import VlmPolicyAgent, VlmPolicyConfig

        cfg = VlmPolicyConfig(
            window_title=str(payload.get("window_title") or "Blue Archive"),
            goal=str(payload.get("goal") or "Keep the game running safely."),
            steps=int(payload.get("steps") or 0),
            dry_run=bool(payload.get("dry_run") if payload.get("dry_run") is not None else True),
            step_sleep_s=float(payload.get("step_sleep_s") or 0.6),
            exploration_click=bool(payload.get("exploration_click") or False),
            forbid_premium_currency=bool(payload.get("forbid_premium_currency") if payload.get("forbid_premium_currency") is not None else True),
        )

        _VLM_AGENT = VlmPolicyAgent(cfg)
        _VLM_AGENT.start()


def _stop_vlm_agent() -> None:
    global _VLM_AGENT
    with _VLM_AGENT_LOCK:
        if _VLM_AGENT is None:
            return
        try:
            _VLM_AGENT.stop()
        except Exception:
            pass
        _VLM_AGENT = None


def _tcp_listening(host: str, port: int, timeout_s: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


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


_ollama_proc: Optional[subprocess.Popen] = None
_agent_proc: Optional[subprocess.Popen] = None


def ensure_ollama(*, host: str, port: int, models_dir: str, auto_pull: bool, model_tag: str) -> None:
    global _ollama_proc

    if _tcp_listening(host, port):
        if auto_pull and model_tag:
            _ollama_pull(models_dir=models_dir, model_tag=model_tag)
        return

    out_log = LOGS_DIR / "ollama.out.log"
    err_log = LOGS_DIR / "ollama.err.log"
    out_f = open(out_log, "ab", buffering=0)
    err_f = open(err_log, "ab", buffering=0)

    env = os.environ.copy()
    if models_dir:
        env["OLLAMA_MODELS"] = models_dir

    try:
        _ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=out_f,
            stderr=err_f,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="Ollama not found. Please install Ollama and ensure `ollama` is in PATH.") from e

    _write_state(
        "ollama.state.json",
        {
            "pid": _ollama_proc.pid,
            "host": host,
            "port": port,
            "models_dir": models_dir,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "started_by_dashboard_server",
        },
    )

    t0 = time.time()
    while time.time() - t0 < 30:
        if _tcp_listening(host, port):
            break
        if _ollama_proc.poll() is not None:
            raise HTTPException(status_code=500, detail="Ollama exited early. See logs/ollama.err.log")
        time.sleep(0.3)

    if not _tcp_listening(host, port):
        raise HTTPException(status_code=500, detail="Ollama did not open port 11434 in time. See logs/ollama.*.log")

    if auto_pull and model_tag:
        _ollama_pull(models_dir=models_dir, model_tag=model_tag)


@app.post("/api/v1/shutdown")
def shutdown_server() -> Dict[str, str]:
    def _do_exit():
        time.sleep(0.2)
        try:
            _stop_vlm_agent()
        except Exception:
            pass
        os._exit(0)
    threading.Thread(target=_do_exit, daemon=True).start()
    return {"status": "shutting_down"}


@app.post("/api/v1/vlm/ensure")
def ensure_vlm(payload: Dict[str, Any]) -> Dict[str, Any]:
    base_url = str(payload.get("base_url") or "http://127.0.0.1:11434").strip()
    models_dir = str(payload.get("models_dir") or os.environ.get("OLLAMA_MODELS") or r"D:\\Project\\ml_cache\\models").strip()
    model_tag = str(payload.get("model") or payload.get("model_tag") or os.environ.get("VLM_MODEL") or "").strip()
    auto_pull = bool(payload.get("auto_pull") if payload.get("auto_pull") is not None else False)

    try:
        u = urlparse(base_url)
        host = u.hostname or "127.0.0.1"
        port = int(u.port or 11434)
    except Exception:
        host = "127.0.0.1"
        port = 11434

    ensure_ollama(host=host, port=port, models_dir=models_dir, auto_pull=auto_pull, model_tag=model_tag)
    return {"ok": True, "base_url": f"http://{host}:{port}", "models_dir": models_dir, "model": model_tag, "auto_pull": auto_pull}


@app.get("/api/v1/vlm/tags")
def vlm_tags(base_url: str = Query("")) -> Dict[str, Any]:
    bu = (base_url or os.environ.get("VLM_BASE_URL") or os.environ.get("LLM_BASE_URL") or "http://127.0.0.1:11434").strip()
    url = bu.rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to query ollama tags: {e}")


def _ollama_pull(*, models_dir: str, model_tag: str) -> None:
    if not model_tag:
        return

    out_log = LOGS_DIR / "ollama_pull.out.log"
    err_log = LOGS_DIR / "ollama_pull.err.log"

    env = os.environ.copy()
    if models_dir:
        env["OLLAMA_MODELS"] = models_dir

    with open(out_log, "ab", buffering=0) as out_f, open(err_log, "ab", buffering=0) as err_f:
        p = subprocess.run(
            ["ollama", "pull", model_tag],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=out_f,
            stderr=err_f,
        )
        if p.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ollama pull failed (exit={p.returncode}). See logs/ollama_pull.err.log")


def start_agent(
    *,
    llm_base_url: str,
    llm_model: str,
    adb_serial: str,
    steps: int,
    od_queries: Optional[List[str]],
    ollama_models_dir: str,
) -> None:
    global _agent_proc

    if _agent_proc is not None and _agent_proc.poll() is None:
        return

    out_log = LOGS_DIR / "agent.out.log"
    err_log = LOGS_DIR / "agent.err.log"
    out_f = open(out_log, "ab", buffering=0)
    err_f = open(err_log, "ab", buffering=0)

    env = os.environ.copy()
    env["LLM_BASE_URL"] = llm_base_url
    env["LLM_MODEL"] = llm_model
    if adb_serial:
        env["ADB_SERIAL"] = adb_serial
    if ollama_models_dir:
        env["OLLAMA_MODELS"] = ollama_models_dir

    args: List[str] = ["py", "main.py", "--llm-base-url", llm_base_url, "--llm-model", llm_model]
    if steps > 0:
        args += ["--steps", str(steps)]
    if od_queries:
        for q in od_queries:
            args += ["--od", q]

    _agent_proc = subprocess.Popen(
        args,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=out_f,
        stderr=err_f,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )

    _write_state(
        "agent.state.json",
        {
            "pid": _agent_proc.pid,
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "adb_serial": adb_serial,
            "od_queries": od_queries or [],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "started_by_dashboard_server",
        },
    )


def stop_agent() -> None:
    global _agent_proc
    if _agent_proc is None:
        return
    try:
        if _agent_proc.poll() is None:
            _agent_proc.kill()
    except Exception:
        pass
    _agent_proc = None


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
    agent = _get_vlm_agent()
    agent_running = bool(agent is not None and getattr(agent, "is_running")())
    last_action = None
    last_error = ""
    agent_cfg = None
    try:
        if agent is not None:
            last_action = getattr(agent, "last_action")
            last_error = getattr(agent, "last_error")
            cfg = getattr(agent, "cfg", None)
            if cfg is not None:
                agent_cfg = {
                    "window_title": getattr(cfg, "window_title", None),
                    "goal": getattr(cfg, "goal", None),
                    "steps": getattr(cfg, "steps", None),
                    "dry_run": getattr(cfg, "dry_run", None),
                    "step_sleep_s": getattr(cfg, "step_sleep_s", None),
                    "exploration_click": getattr(cfg, "exploration_click", None),
                    "forbid_premium_currency": getattr(cfg, "forbid_premium_currency", None),
                    "model": getattr(cfg, "model", None),
                }
    except Exception:
        pass
    return {
        "server_pid": os.getpid(),
        "server_started_at": round(_SERVER_STARTED_AT, 3),
        "ollama_listening": False,
        "agent_running": agent_running,
        "agent_pid": None,
        "agent_type": "vlm_policy",
        "agent_cfg": agent_cfg,
        "last_action": last_action,
        "last_error": last_error,
    }


@app.get("/api/v1/debug/vision")
def debug_vision() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    try:
        import transformers

        data["transformers_version"] = getattr(transformers, "__version__", None)
        try:
            from transformers import PreTrainedModel

            data["pretrainedmodel_has__supports_sdpa"] = hasattr(PreTrainedModel, "_supports_sdpa")
        except Exception as e:
            data["pretrainedmodel_has__supports_sdpa"] = f"error: {e}"

        try:
            from transformers.generation.utils import GenerationMixin

            data["generationmixin_has__supports_sdpa"] = hasattr(GenerationMixin, "_supports_sdpa")
        except Exception as e:
            data["generationmixin_has__supports_sdpa"] = f"error: {e}"
    except Exception as e:
        data["transformers_import_error"] = str(e)

    data["vision_loaded"] = _VISION is not None
    return data


@app.post("/api/v1/local_vlm/warmup")
def local_vlm_warmup(
    model: str = Query(""),
    models_dir: str = Query(""),
    hf_home: str = Query(""),
    device: str = Query(""),
    max_new_tokens: int = Query(64),
) -> Dict[str, Any]:
    m = (model or os.environ.get("LOCAL_VLM_MODEL") or LOCAL_VLM_MODEL or "").strip()
    if not m:
        raise HTTPException(status_code=400, detail="model is required (query model=... or env LOCAL_VLM_MODEL)")

    md = (
        models_dir
        or os.environ.get("LOCAL_VLM_MODELS_DIR")
        or LOCAL_VLM_MODELS_DIR
        or os.environ.get("MODELS_DIR")
        or MODELS_DIR
        or r"D:\\Project\\ml_cache\\models\\vlm"
    ).strip()
    hh = (hf_home or os.environ.get("HF_HOME") or HF_CACHE_DIR or r"D:\\Project\\ml_cache\\huggingface").strip()
    dev = (device or os.environ.get("LOCAL_VLM_DEVICE") or LOCAL_VLM_DEVICE or "cuda").strip()

    mnt = int(max_new_tokens) if max_new_tokens else 64
    if mnt < 16:
        mnt = 16

    t0 = time.time()
    from vision.local_vlm_runtime import get_local_vlm

    engine = get_local_vlm(model=m, models_dir=md, hf_home=hh, device=dev)

    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="vlm_warmup_", suffix=".png")
        os.close(fd)
        im = Image.new("RGB", (32, 32), (255, 255, 255))
        im.save(tmp_path)

        prompt = "Return JSON only: {\"items\": []}."
        res = engine.ocr(image_path=tmp_path, prompt=prompt, max_new_tokens=mnt)
        raw = str(res.get("raw") or "")
    finally:
        try:
            if tmp_path:
                os.remove(tmp_path)
        except Exception:
            pass

    return {
        "ok": True,
        "model": m,
        "device": dev,
        "models_dir": md,
        "hf_home": hh,
        "max_new_tokens": mnt,
        "elapsed_s": round(time.time() - t0, 3),
        "raw_head": raw.strip().replace("\n", " ")[:200],
    }


@app.on_event("startup")
def _startup_local_vlm_warmup() -> None:
    if (os.environ.get("LOCAL_VLM_WARMUP") or "").strip() != "1":
        return
    try:
        local_vlm_warmup()
    except Exception as e:
        try:
            print(f"local_vlm warmup failed: {e}")
        except Exception:
            pass


@app.get("/api/v1/local_vlm/ocr")
def local_vlm_ocr(
    path: str = Query(...),
    model: str = Query(""),
    models_dir: str = Query(""),
    hf_home: str = Query(""),
    device: str = Query(""),
    max_new_tokens: int = Query(2048),
    strict: bool = Query(False),
    force: bool = Query(False),
    debug: bool = Query(False),
) -> Dict[str, Any]:
    rel = (path or "").replace("\\", "/")
    img = _safe_capture_path(rel)
    if not img.exists() or not img.is_file():
        raise HTTPException(status_code=404, detail="image not found")

    m = (model or os.environ.get("LOCAL_VLM_MODEL") or LOCAL_VLM_MODEL or "").strip()
    if not m:
        raise HTTPException(status_code=400, detail="model is required (query model=... or env LOCAL_VLM_MODEL)")

    md = (
        models_dir
        or os.environ.get("LOCAL_VLM_MODELS_DIR")
        or LOCAL_VLM_MODELS_DIR
        or os.environ.get("MODELS_DIR")
        or MODELS_DIR
        or r"D:\\Project\\ml_cache\\models\\vlm"
    ).strip()
    hh = (hf_home or os.environ.get("HF_HOME") or HF_CACHE_DIR or r"D:\\Project\\ml_cache\\huggingface").strip()
    dev = (device or os.environ.get("LOCAL_VLM_DEVICE") or LOCAL_VLM_DEVICE or "cuda").strip()
    mnt = int(max_new_tokens) if max_new_tokens else int(LOCAL_VLM_MAX_NEW_TOKENS)
    if mnt < 64:
        mnt = 64

    cache_path = _local_vlm_ocr_cache_path(rel, m)
    if (not debug) and (not force) and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        try:
            with Image.open(img) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0

        prompt = (
            "You are an OCR engine. Extract all visible text regions from the image and return JSON only. "
            "Return format: {\"items\": [{\"label\": <text>, \"bbox\": [x1,y1,x2,y2]}]}. "
            "bbox must be pixel coordinates in the original image. "
            f"Image size: width={w}, height={h}. "
            "Do not include any extra keys. Do not wrap in markdown."
        )

        from vision.local_vlm_runtime import get_local_vlm

        engine = get_local_vlm(model=m, models_dir=md, hf_home=hh, device=dev)
        res = engine.ocr(image_path=str(img), prompt=prompt, max_new_tokens=mnt)
        raw = str(res.get("raw") or "")
        if not raw.strip():
            parsed = None
            parse_error = "empty content from local_vlm"
        else:
            try:
                parsed = _parse_json_content(raw)
                parse_error = ""
            except Exception as e:
                parsed = None
                parse_error = str(e)

        items = []
        if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
            items = parsed.get("items")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"local_vlm ocr failed: {e}")

    data: Dict[str, Any] = {"image": rel, "model": m, "items": items}
    if debug:
        data["debug"] = {
            "raw": raw[:2000],
            "parse_error": parse_error,
            "models_dir": md,
            "hf_home": hh,
            "device": dev,
            "max_new_tokens": mnt,
        }

    if parse_error:
        data["error"] = {"type": "parse_error", "message": parse_error, "raw_head": raw.strip().replace("\n", " ")[:400]}
        if (not debug) and strict:
            raise HTTPException(status_code=500, detail=f"local_vlm output is not valid json: {parse_error}")

    try:
        if (not debug) and (not parse_error):
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return data


@app.get("/api/v1/vlm/ocr")
def vlm_ocr(
    path: str = Query(...),
    model: str = Query(""),
    base_url: str = Query(""),
    timeout_s: int = Query(1800),
    keep_alive: str = Query("10m"),
    num_ctx: int = Query(8192),
    num_predict: int = Query(2048),
    num_gpu: int = Query(99),
    strict: bool = Query(False),
    force: bool = Query(False),
    debug: bool = Query(False),
) -> Dict[str, Any]:
    rel = (path or "").replace("\\", "/")
    img = _safe_capture_path(rel)
    if not img.exists() or not img.is_file():
        raise HTTPException(status_code=404, detail="image not found")

    m = (model or os.environ.get("VLM_MODEL") or os.environ.get("LLM_MODEL") or "").strip()
    if not m:
        raise HTTPException(status_code=400, detail="model is required (query model=... or env VLM_MODEL)")

    bu = (base_url or os.environ.get("VLM_BASE_URL") or os.environ.get("LLM_BASE_URL") or "http://127.0.0.1:11434").strip()

    cache_path = _vlm_ocr_cache_path(rel, m)
    if (not debug) and (not force) and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        ts = int(timeout_s) if timeout_s else 1800
        if ts < 30:
            ts = 30
        nc = int(num_ctx) if num_ctx else 8192
        if nc < 512:
            nc = 512

        np = int(num_predict) if num_predict else 2048
        if np < 64:
            np = 64

        ng = int(num_gpu) if num_gpu else 0
        if ng < 0:
            ng = 0

        try:
            res = _ollama_vlm_ocr(
                image_path=img,
                model=m,
                base_url=bu,
                timeout_s=ts,
                keep_alive=keep_alive,
                num_ctx=nc,
                num_predict=np,
                num_gpu=ng,
            )
        except requests.RequestException as e:
            if ng > 0:
                try:
                    res = _ollama_vlm_ocr(
                        image_path=img,
                        model=m,
                        base_url=bu,
                        timeout_s=ts,
                        keep_alive=keep_alive,
                        num_ctx=nc,
                        num_predict=np,
                        num_gpu=0,
                    )
                    res["fallback"] = {"num_gpu": 0, "request_error": str(e)}
                except requests.RequestException as e2:
                    if debug:
                        return {
                            "image": rel,
                            "model": m,
                            "items": [],
                            "error": {"type": "ollama_request_failed", "message": str(e)},
                            "fallback": {"num_gpu": 0, "error": str(e2)},
                        }
                    raise HTTPException(status_code=500, detail=f"ollama request failed: {e}; fallback(num_gpu=0) failed: {e2}")
            else:
                if debug:
                    return {
                        "image": rel,
                        "model": m,
                        "items": [],
                        "error": {"type": "ollama_request_failed", "message": str(e)},
                    }
                raise

        if res.get("parse_error") and (ng > 0) and ("empty content" in str(res.get("parse_error"))):
            res2 = _ollama_vlm_ocr(
                image_path=img,
                model=m,
                base_url=bu,
                timeout_s=ts,
                keep_alive=keep_alive,
                num_ctx=nc,
                num_predict=np,
                num_gpu=0,
            )
            if not res2.get("parse_error"):
                res = res2
            else:
                res["fallback"] = {"num_gpu": 0, "parse_error": str(res2.get("parse_error") or "")}

        parsed = res.get("parsed") or {}
        items = parsed.get("items")
        if not isinstance(items, list):
            items = []
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"ollama request failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vlm ocr failed: {e}")

    data: Dict[str, Any] = {"image": rel, "model": m, "items": items}
    if debug:
        data["debug"] = res

    if res.get("parse_error"):
        raw = str(res.get("raw") or "")
        head = raw.strip().replace("\n", " ")[:400]
        data["error"] = {"type": "parse_error", "message": str(res.get("parse_error")), "raw_head": head}
        if (not debug) and strict:
            raise HTTPException(status_code=500, detail=f"vlm output is not valid json: {res.get('parse_error')}; raw_head={head}")

    try:
        if (not debug) and (not res.get("parse_error")):
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return data


@app.post("/api/v1/vision/reload")
def reload_vision() -> Dict[str, Any]:
    global _VISION
    try:
        import importlib

        import vision.florence_vision as florence_vision

        importlib.reload(florence_vision)
    except Exception:
        pass

    with _VISION_LOCK:
        _VISION = None

    return {"ok": True}


def _safe_capture_path(rel: str) -> Path:
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if not rel:
        raise HTTPException(status_code=400, detail="path is required")

    p = (CAPTURES_DIR / rel).resolve()
    base = CAPTURES_DIR.resolve()
    if not str(p).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid path")
    return p


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


def _ocr_cache_path(rel: str) -> Path:
    key = f"v{OCR_CACHE_VERSION}:{rel}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    cache_dir = CAPTURES_DIR / "_cache" / "ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}.json"


def _vlm_ocr_cache_path(rel: str, model: str) -> Path:
    key = f"v{VLM_OCR_CACHE_VERSION}:{model}:{rel}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    cache_dir = CAPTURES_DIR / "_cache" / "vlm_ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}.json"


def _local_vlm_ocr_cache_path(rel: str, model: str) -> Path:
    key = f"v{LOCAL_VLM_OCR_CACHE_VERSION}:{model}:{rel}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    cache_dir = CAPTURES_DIR / "_cache" / "local_vlm_ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}.json"


def _parse_json_content(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    s = (content or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            if "\n" in s:
                s = s.split("\n", 1)[1]
            s = s.strip()

    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        return json.loads(s[l : r + 1])

    raise json.JSONDecodeError("Unable to parse JSON", content, 0)


def _ollama_vlm_ocr(
    *,
    image_path: Path,
    model: str,
    base_url: str,
    timeout_s: int,
    keep_alive: str,
    num_ctx: int,
    num_predict: int,
    num_gpu: int,
) -> Dict[str, Any]:
    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception:
        w, h = 0, 0

    img_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = (
        "You are an OCR engine. Extract all visible text regions from the image and return JSON only. "
        "Return format: {\"items\": [{\"label\": <text>, \"bbox\": [x1,y1,x2,y2]}]}. "
        "bbox must be pixel coordinates in the original image. "
        f"Image size: width={w}, height={h}. "
        "Do not include any extra keys. Do not wrap in markdown."
    )

    base = base_url.rstrip("/")
    url = base + "/api/chat"
    def _call_chat(use_format_json: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "stream": False,
            "keep_alive": keep_alive,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            "options": {
                "temperature": 0.0,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "num_gpu": num_gpu,
            },
        }
        if use_format_json:
            payload["format"] = "json"

        resp = requests.post(url, json=payload, timeout=(10, timeout_s))
        resp.raise_for_status()
        return resp.json()

    def _call_generate() -> Dict[str, Any]:
        url2 = base + "/api/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "stream": False,
            "keep_alive": keep_alive,
            "prompt": prompt,
            "images": [img_b64],
            "options": {
                "temperature": 0.0,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "num_gpu": num_gpu,
            },
        }
        resp = requests.post(url2, json=payload, timeout=(10, timeout_s))
        resp.raise_for_status()
        return resp.json()

    data = _call_chat(False)
    msg = data.get("message") if isinstance(data.get("message"), dict) else {}
    content = (msg.get("content") or "") if isinstance(msg, dict) else ""
    if not content.strip():
        data2 = _call_chat(True)
        msg2 = data2.get("message") if isinstance(data2.get("message"), dict) else {}
        content2 = (msg2.get("content") or "") if isinstance(msg2, dict) else ""
        if content2.strip():
            data = data2
            msg = msg2
            content = content2

    if not content.strip():
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if isinstance(tool_calls, list) and tool_calls:
            tc0 = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
            fn = tc0.get("function") if isinstance(tc0.get("function"), dict) else {}
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, dict):
                return {
                    "raw": "",
                    "parsed": args,
                    "parse_error": "",
                    "ollama": {"model": data.get("model"), "done_reason": data.get("done_reason"), "message": msg},
                }
            if isinstance(args, str) and args.strip():
                try:
                    parsed = _parse_json_content(args)
                    return {
                        "raw": args,
                        "parsed": parsed,
                        "parse_error": "",
                        "ollama": {"model": data.get("model"), "done_reason": data.get("done_reason"), "message": msg},
                    }
                except Exception:
                    pass

        try:
            data3 = _call_generate()
            content3 = data3.get("response") or ""
            if isinstance(content3, str) and content3.strip():
                data = data3
                content = content3
        except Exception:
            pass

    if not content.strip():
        return {
            "raw": "",
            "parsed": None,
            "parse_error": f"empty content from ollama: keys={list(data.keys())}",
            "ollama": {"model": data.get("model"), "done_reason": data.get("done_reason"), "message": msg},
        }

    try:
        parsed = _parse_json_content(content)
        return {
            "raw": content,
            "parsed": parsed,
            "parse_error": "",
            "ollama": {"model": data.get("model"), "done_reason": data.get("done_reason"), "message": msg},
        }
    except Exception as e:
        return {
            "raw": content,
            "parsed": None,
            "parse_error": str(e),
            "ollama": {"model": data.get("model"), "done_reason": data.get("done_reason"), "message": msg},
        }


@app.get("/api/v1/vision/ocr")
def vision_ocr(path: str = Query(...), force: bool = Query(False), debug: bool = Query(False)) -> Dict[str, Any]:
    rel = (path or "").replace("\\", "/")
    img = _safe_capture_path(rel)
    if not img.exists() or not img.is_file():
        raise HTTPException(status_code=404, detail="image not found")

    cache_path = _ocr_cache_path(rel)
    if (not debug) and (not force) and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        v = _get_vision()
        items = v.analyze_screen(str(img), od_queries=None, enable_ocr=True)
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"vision dependencies missing: {e}. Please install requirements.txt (einops, timm).",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vision ocr failed: {e}")

    data: Dict[str, Any] = {"image": rel, "items": items}
    if debug:
        try:
            data["debug"] = v.ocr_debug(str(img))
        except Exception as e:
            data["debug_error"] = str(e)

    try:
        if not debug:
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return data


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


@app.post("/api/v1/vision/ocr_index/build")
def build_ocr_index(
    session: str = Query(...),
    compute_missing: bool = Query(False),
) -> Dict[str, Any]:
    session = os.path.basename((session or "").strip())
    if not session:
        raise HTTPException(status_code=400, detail="session is required")

    d = _safe_capture_path(session)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="session not found")

    out_path = d / "ocr_index.jsonl"
    images = sorted([p.name for p in d.glob("*.png")])

    written = 0
    missing = 0
    errors = 0

    v = None
    if compute_missing:
        v = _get_vision()

    with out_path.open("w", encoding="utf-8") as f:
        for name in images:
            rel = f"{session}/{name}".replace("\\", "/")
            cache_path = _ocr_cache_path(rel)
            data: Optional[Dict[str, Any]] = None

            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    data = None

            if data is None:
                missing += 1
                if not compute_missing:
                    continue
                try:
                    items = v.analyze_screen(str(_safe_capture_path(rel)), od_queries=None, enable_ocr=True)
                    data = {"image": rel, "items": items}
                    try:
                        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass
                except Exception:
                    errors += 1
                    continue

            rec = {"image": rel, "items": data.get("items") or []}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    return {
        "ok": True,
        "session": session,
        "out": str(out_path.relative_to(CAPTURES_DIR).as_posix()),
        "total_images": len(images),
        "written": written,
        "missing": missing,
        "errors": errors,
    }


@app.post("/api/v1/local_vlm/ocr_index/build")
def build_local_vlm_ocr_index(
    session: str = Query(...),
    model: str = Query(""),
    compute_missing: bool = Query(False),
) -> Dict[str, Any]:
    session = os.path.basename((session or "").strip())
    if not session:
        raise HTTPException(status_code=400, detail="session is required")

    d = _safe_capture_path(session)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="session not found")

    m = (model or os.environ.get("LOCAL_VLM_MODEL") or LOCAL_VLM_MODEL or "").strip()
    if compute_missing and not m:
        raise HTTPException(status_code=400, detail="model is required when compute_missing=1")

    out_path = d / "local_vlm_ocr_index.jsonl"
    images = sorted([p.name for p in d.glob("*.png")])

    written = 0
    missing = 0
    errors = 0

    with out_path.open("w", encoding="utf-8") as f:
        for name in images:
            rel = f"{session}/{name}".replace("\\", "/")
            cache_path = _local_vlm_ocr_cache_path(rel, m or "") if m else None
            data: Optional[Dict[str, Any]] = None

            if cache_path is not None and cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    data = None

            if data is None:
                missing += 1
                if not compute_missing:
                    continue
                try:
                    data = local_vlm_ocr(path=rel, model=m, debug=False, force=True)
                except Exception:
                    errors += 1
                    continue

            rec = {"image": rel, "model": data.get("model") or m, "items": data.get("items") or []}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    return {
        "ok": True,
        "session": session,
        "out": str(out_path.relative_to(CAPTURES_DIR).as_posix()),
        "total_images": len(images),
        "written": written,
        "missing": missing,
        "errors": errors,
    }


@app.post("/api/v1/vlm/ocr_index/build")
def build_vlm_ocr_index(
    session: str = Query(...),
    model: str = Query(""),
    base_url: str = Query(""),
    compute_missing: bool = Query(False),
) -> Dict[str, Any]:
    session = os.path.basename((session or "").strip())
    if not session:
        raise HTTPException(status_code=400, detail="session is required")

    d = _safe_capture_path(session)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="session not found")

    m = (model or os.environ.get("VLM_MODEL") or os.environ.get("LLM_MODEL") or "").strip()
    if compute_missing and not m:
        raise HTTPException(status_code=400, detail="model is required when compute_missing=1")

    bu = (base_url or os.environ.get("VLM_BASE_URL") or os.environ.get("LLM_BASE_URL") or "http://127.0.0.1:11434").strip()

    out_path = d / "vlm_ocr_index.jsonl"
    images = sorted([p.name for p in d.glob("*.png")])

    written = 0
    missing = 0
    errors = 0

    with out_path.open("w", encoding="utf-8") as f:
        for name in images:
            rel = f"{session}/{name}".replace("\\", "/")
            cache_path = _vlm_ocr_cache_path(rel, m or "") if m else None
            data: Optional[Dict[str, Any]] = None

            if cache_path is not None and cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    data = None

            if data is None:
                missing += 1
                if not compute_missing:
                    continue
                try:
                    res = _ollama_vlm_ocr(image_path=_safe_capture_path(rel), model=m, base_url=bu, timeout_s=600)
                    parsed = res.get("parsed") or {}
                    items = parsed.get("items")
                    if not isinstance(items, list):
                        items = []
                    data = {"image": rel, "model": m, "items": items}
                    try:
                        if cache_path is not None:
                            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass
                except Exception:
                    errors += 1
                    continue

            rec = {"image": rel, "model": data.get("model") or m, "items": data.get("items") or []}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    return {
        "ok": True,
        "session": session,
        "out": str(out_path.relative_to(CAPTURES_DIR).as_posix()),
        "total_images": len(images),
        "written": written,
        "missing": missing,
        "errors": errors,
    }


@app.post("/api/v1/start")
def api_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        agent = _get_vlm_agent()
        if agent is not None and getattr(agent, "is_running")():
            _stop_vlm_agent()
    except Exception:
        pass
    _start_vlm_agent(payload=payload)
    return status()


@app.post("/api/v1/stop")
def api_stop() -> Dict[str, Any]:
    _stop_vlm_agent()
    return status()


@app.get("/api/v1/logs")
def api_logs(
    name: str = Query(...),
    lines: int = Query(200, ge=1, le=2000),
) -> PlainTextResponse:
    safe = os.path.basename(name)
    path = LOGS_DIR / safe
    return PlainTextResponse(_tail_text(path, lines))
