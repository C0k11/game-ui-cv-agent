from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


try:
    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        setattr(PreTrainedModel, "_supports_sdpa", False)
except Exception:
    pass

try:
    from transformers.generation.utils import GenerationMixin

    if not hasattr(GenerationMixin, "_supports_sdpa"):
        setattr(GenerationMixin, "_supports_sdpa", False)
except Exception:
    pass


def _set_hf_cache_dir(cache_dir: str) -> None:
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    try:
        os.environ.pop("TRANSFORMERS_CACHE", None)
    except Exception:
        pass


def _to_xyxy(bbox: Any) -> Optional[List[int]]:
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
        x1, y1, x2, y2 = bbox
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    if isinstance(bbox, (list, tuple)) and len(bbox) == 8 and all(isinstance(x, (int, float)) for x in bbox):
        xs = [float(bbox[i]) for i in (0, 2, 4, 6)]
        ys = [float(bbox[i]) for i in (1, 3, 5, 7)]
        return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]
    if isinstance(bbox, (list, tuple)) and bbox and isinstance(bbox[0], (list, tuple)):
        xs: List[float] = []
        ys: List[float] = []
        for pt in bbox:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
        if xs and ys:
            return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]
    return None


def _unwrap_task_dict(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return parsed
    if "bboxes" in parsed or "labels" in parsed or "quad_boxes" in parsed or "boxes" in parsed:
        return parsed
    if len(parsed) == 1:
        (_, v) = next(iter(parsed.items()))
        return v
    for v in parsed.values():
        if isinstance(v, dict) and ("bboxes" in v or "labels" in v or "quad_boxes" in v or "boxes" in v):
            return v
    return parsed


def _extract_items(parsed: Any) -> List[Dict[str, Any]]:
    if parsed is None:
        return []
    parsed = _unwrap_task_dict(parsed)
    if isinstance(parsed, dict):
        labels = parsed.get("labels") or parsed.get("texts") or parsed.get("text") or parsed.get("words") or []
        bboxes = parsed.get("bboxes") or parsed.get("boxes") or parsed.get("quad_boxes") or []
        if isinstance(labels, str):
            labels = [labels]
        items: List[Dict[str, Any]] = []
        for i in range(min(len(labels), len(bboxes))):
            xyxy = _to_xyxy(bboxes[i])
            if xyxy is None:
                continue
            items.append({"label": str(labels[i]), "bbox": xyxy})
        return items
    if isinstance(parsed, list):
        items: List[Dict[str, Any]] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label") or entry.get("text") or entry.get("word")
            bbox = entry.get("bbox") or entry.get("box")
            xyxy = _to_xyxy(bbox)
            if label is None or xyxy is None:
                continue
            items.append({"label": str(label), "bbox": xyxy})
        return items
    return []


def _to_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, os.PathLike, Path)):
        return Image.open(str(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr[:, :, ::-1]).convert("RGB")
        if arr.ndim == 3 and arr.shape[2] == 4:
            return Image.fromarray(arr[:, :, [2, 1, 0, 3]], mode="RGBA").convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


@dataclass
class FlorenceConfig:
    model_id: str = os.environ.get("FLORENCE_MODEL_ID", "microsoft/Florence-2-large-ft")
    device: str = os.environ.get("FLORENCE_DEVICE", "cuda")
    dtype: torch.dtype = torch.float16
    cache_dir: str = os.environ.get("FLORENCE_CACHE_DIR", r"D:\Project\ml_cache\models")
    max_new_tokens: int = int(os.environ.get("FLORENCE_MAX_NEW_TOKENS", "256"))
    num_beams: int = int(os.environ.get("FLORENCE_NUM_BEAMS", "1"))


class FlorenceVision:
    def __init__(self, cfg: Optional[FlorenceConfig] = None):
        self.cfg = cfg or FlorenceConfig()
        if str(self.cfg.device).lower().startswith("cuda") and not torch.cuda.is_available():
            self.cfg.device = "cpu"
        if str(self.cfg.device).lower() == "cpu" and self.cfg.dtype in (torch.float16, torch.bfloat16):
            self.cfg.dtype = torch.float32
        _set_hf_cache_dir(self.cfg.cache_dir)
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=True,
            cache_dir=self.cfg.cache_dir,
        )
        load_attempts = [
            {
                "trust_remote_code": True,
                "dtype": self.cfg.dtype,
                "attn_implementation": "eager",
                "cache_dir": self.cfg.cache_dir,
            },
            {
                "trust_remote_code": True,
                "torch_dtype": self.cfg.dtype,
                "attn_implementation": "eager",
                "cache_dir": self.cfg.cache_dir,
            },
            {
                "trust_remote_code": True,
                "torch_dtype": self.cfg.dtype,
                "cache_dir": self.cfg.cache_dir,
            },
        ]
        last_err: Optional[Exception] = None
        self.model = None
        for kwargs in load_attempts:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, **kwargs)
                last_err = None
                break
            except TypeError as e:
                last_err = e
        if self.model is None:
            if last_err is not None:
                raise last_err
            raise RuntimeError("Failed to load Florence model")
        try:
            if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
                self.model.generation_config.use_cache = False
        except Exception:
            pass
        try:
            if hasattr(self.model, "config") and self.model.config is not None:
                self.model.config.use_cache = False
        except Exception:
            pass
        if not hasattr(self.model, "_supports_sdpa"):
            try:
                setattr(self.model, "_supports_sdpa", False)
            except Exception:
                pass
        self.model.to(self.cfg.device)
        self.model.eval()

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict[str, torch.Tensor]:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        moved: Dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k == "pixel_values" and v.is_floating_point() and str(self.cfg.device).lower().startswith("cuda"):
                moved[k] = v.to(self.cfg.device, dtype=self.cfg.dtype)
            else:
                moved[k] = v.to(self.cfg.device)
        return moved

    def _generate_raw(self, image: Any, prompt: str, *, max_new_tokens: Optional[int] = None, num_beams: Optional[int] = None) -> str:
        pil = _to_pil_rgb(image)
        inputs = self._prepare_inputs(pil, prompt)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs.get("input_ids"),
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                do_sample=False,
                num_beams=num_beams or self.cfg.num_beams,
                use_cache=False,
            )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    def run_task(self, image: Any, task_prompt: str) -> Any:
        pil = _to_pil_rgb(image)
        generated_text = self._generate_raw(pil, task_prompt)
        return self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=pil.size,
        )

    def generate_text(self, image: Any, prompt: str, *, max_new_tokens: int = 32, num_beams: int = 1) -> str:
        text = self._generate_raw(image, prompt, max_new_tokens=max_new_tokens, num_beams=num_beams)
        return text.strip()

    def answer_yes_no(self, image: Any, prompt: str, *, default: bool = False) -> bool:
        text = self.generate_text(image, prompt, max_new_tokens=16, num_beams=1)
        low = text.lower().strip()
        if not low:
            return default
        negative_hits = ["no", "not", "disabled", "grey", "gray", "dark", "cooldown", "unavailable"]
        positive_hits = ["yes", "enabled", "clickable", "available", "ready"]
        neg = any(tok in low for tok in negative_hits)
        pos = any(tok in low for tok in positive_hits)
        if pos and not neg:
            return True
        if neg and not pos:
            return False
        first = re.sub(r"[^a-z]+", " ", low).strip().split()
        if first:
            if first[0] in ("yes", "enabled", "clickable", "available", "ready"):
                return True
            if first[0] in ("no", "disabled", "grey", "gray", "dark", "cooldown", "unavailable"):
                return False
        return default

    def classify_button_enabled(self, image: Any, *, hint: str = "button", default: bool = True) -> bool:
        prompt = f"Is the main {hint} in this crop enabled and clickable, or disabled or greyed out? Answer only ENABLED or DISABLED."
        text = self.generate_text(image, prompt, max_new_tokens=12, num_beams=1).lower()
        if "enabled" in text:
            return True
        if "disabled" in text or "grey" in text or "gray" in text or "dark" in text:
            return False
        return default

    def detect_open_vocabulary(self, image: Any, queries: Sequence[str]) -> List[Dict[str, Any]]:
        pil = _to_pil_rgb(image)
        results: List[Dict[str, Any]] = []
        for q in queries:
            query = str(q).strip()
            if not query:
                continue
            parsed = self.run_task(pil, f"<OPEN_VOCABULARY_DETECTION> {query}")
            items = _extract_items(parsed)
            for it in items:
                it.setdefault("type", "od")
                it.setdefault("query", query)
                it.setdefault("label", str(it.get("label") or query))
                it.setdefault("score", 1.0)
            results.extend(items)
        return results

    def suggest_labels(self, image: Any, labels: Sequence[str]) -> List[Dict[str, Any]]:
        queries = [str(x).strip() for x in labels if str(x).strip()]
        hits = self.detect_open_vocabulary(image, queries)
        ranked: List[Dict[str, Any]] = []
        for hit in hits:
            bbox = _to_xyxy(hit.get("bbox"))
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            ranked.append({
                "label": str(hit.get("query") or hit.get("label") or ""),
                "bbox": [x1, y1, x2, y2],
                "score": float(hit.get("score") or 1.0),
                "query": str(hit.get("query") or hit.get("label") or ""),
            })
        ranked.sort(key=lambda x: ((x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1])), reverse=True)
        return ranked

    def analyze_screen(self, screenshot_path: str, od_queries: Optional[Sequence[str]] = None, enable_ocr: bool = True) -> List[Dict[str, Any]]:
        image = _to_pil_rgb(screenshot_path)
        results: List[Dict[str, Any]] = []
        if od_queries:
            results.extend(self.detect_open_vocabulary(image, od_queries))
        if enable_ocr:
            parsed = self.run_task(image, "<OCR_WITH_REGION>")
            items = _extract_items(parsed)
            for it in items:
                it.setdefault("type", "ocr")
            results.extend(items)
        return results


class FlorenceReferenceMatcher:
    def __init__(self, vision: FlorenceVision, reference_dir: str):
        self.vision = vision
        self.reference_dir = Path(reference_dir)
        self._cache: Dict[str, Image.Image] = {}

    def _load_reference(self, name: str) -> Optional[Image.Image]:
        if name in self._cache:
            return self._cache[name]
        path = self.reference_dir / f"{name}.png"
        if not path.exists():
            path = self.reference_dir / name
        if not path.exists():
            return None
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        crop = img.crop((int(w * 0.125), 0, int(w * 0.875), int(h * 0.65)))
        cw, ch = crop.size
        side = min(cw, ch)
        left = max(0, (cw - side) // 2)
        top = max(0, (ch - side) // 2)
        crop = crop.crop((left, top, left + side, top + side)).convert("RGB")
        crop = crop.resize((192, 192), Image.Resampling.LANCZOS)
        self._cache[name] = crop
        return crop

    def _prepare_candidate(self, image: Any) -> Image.Image:
        crop = _to_pil_rgb(image)
        w, h = crop.size
        side = min(w, h)
        left = max(0, (w - side) // 2)
        top = max(0, (h - side) // 2)
        crop = crop.crop((left, top, left + side, top + side))
        return crop.resize((192, 192), Image.Resampling.LANCZOS)

    def _compose_pair(self, ref_img: Image.Image, cand_img: Image.Image) -> Image.Image:
        canvas = Image.new("RGB", (416, 208), (245, 245, 245))
        canvas.paste(ref_img, (8, 8))
        canvas.paste(cand_img, (216, 8))
        return canvas

    def match_candidate(self, candidate_image: Any, candidate_names: Sequence[str]) -> Tuple[Optional[str], float]:
        cand = self._prepare_candidate(candidate_image)
        best_name: Optional[str] = None
        best_score = -1.0
        for name in candidate_names:
            ref = self._load_reference(str(name))
            if ref is None:
                continue
            pair = self._compose_pair(ref, cand)
            prompt = f"The left portrait is the reference image of {name} from Blue Archive. Does the right portrait show the same student? Answer only YES or NO."
            same = self.vision.answer_yes_no(pair, prompt, default=False)
            score = 1.0 if same else 0.0
            if score > best_score:
                best_name = str(name)
                best_score = score
            if same:
                return str(name), 1.0
        return best_name, best_score


_FLORENCE_LOCK = threading.Lock()
_FLORENCE: Optional[FlorenceVision] = None
_FLORENCE_KEY = ""
_FLORENCE_MATCHERS: Dict[str, FlorenceReferenceMatcher] = {}


def get_florence_vision(cfg: Optional[FlorenceConfig] = None) -> FlorenceVision:
    global _FLORENCE, _FLORENCE_KEY
    cfg = cfg or FlorenceConfig()
    key = f"{cfg.model_id}|{cfg.device}|{cfg.cache_dir}|{cfg.dtype}|{cfg.max_new_tokens}|{cfg.num_beams}"
    with _FLORENCE_LOCK:
        if _FLORENCE is None or _FLORENCE_KEY != key:
            _FLORENCE = FlorenceVision(cfg)
            _FLORENCE_KEY = key
        return _FLORENCE


def get_florence_reference_matcher(reference_dir: str, cfg: Optional[FlorenceConfig] = None) -> FlorenceReferenceMatcher:
    ref_key = str(Path(reference_dir).resolve())
    with _FLORENCE_LOCK:
        matcher = _FLORENCE_MATCHERS.get(ref_key)
        if matcher is None:
            matcher = FlorenceReferenceMatcher(get_florence_vision(cfg), ref_key)
            _FLORENCE_MATCHERS[ref_key] = matcher
        return matcher
