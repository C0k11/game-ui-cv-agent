import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)


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

    if isinstance(bbox, (list, tuple)) and len(bbox) > 0 and isinstance(bbox[0], (list, tuple)):
        xs: List[float] = []
        ys: List[float] = []
        for pt in bbox:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
        if not xs or not ys:
            return None
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
        if isinstance(v, dict) and (
            "bboxes" in v or "labels" in v or "quad_boxes" in v or "boxes" in v
        ):
            return v

    return parsed


def _extract_items(parsed: Any) -> List[Dict[str, Any]]:
    if parsed is None:
        return []

    parsed = _unwrap_task_dict(parsed)

    if isinstance(parsed, dict):
        labels = (
            parsed.get("labels")
            or parsed.get("texts")
            or parsed.get("text")
            or parsed.get("words")
            or []
        )
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


@dataclass
class FlorenceConfig:
    model_id: str = os.environ.get("FLORENCE_MODEL_ID", "microsoft/Florence-2-large")
    device: str = os.environ.get("FLORENCE_DEVICE", "cuda")
    dtype: torch.dtype = torch.float16
    cache_dir: str = os.environ.get("HF_HOME", r"D:\\Project\\ml_cache\\huggingface")
    max_new_tokens: int = int(os.environ.get("FLORENCE_MAX_NEW_TOKENS", "256"))
    num_beams: int = int(os.environ.get("FLORENCE_NUM_BEAMS", "1"))


class FlorenceVision:
    def __init__(self, cfg: Optional[FlorenceConfig] = None):
        self.cfg = cfg or FlorenceConfig()

        if str(self.cfg.device).lower().startswith("cuda") and not torch.cuda.is_available():
            self.cfg.device = "cpu"

        if str(self.cfg.device).lower().startswith("cuda"):
            if "FLORENCE_MAX_NEW_TOKENS" not in os.environ and self.cfg.max_new_tokens <= 256:
                self.cfg.max_new_tokens = 768
            if "FLORENCE_NUM_BEAMS" not in os.environ and self.cfg.num_beams <= 1:
                self.cfg.num_beams = 3

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
        for kwargs in load_attempts:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, **kwargs)
                last_err = None
                break
            except TypeError as e:
                last_err = e

        if last_err is not None:
            raise last_err

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
                try:
                    setattr(self.model.__class__, "_supports_sdpa", False)
                except Exception:
                    pass
        self.model.to(self.cfg.device)
        self.model.eval()

    def _run_task(self, image: Image.Image, task_prompt: str) -> Any:
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
        moved: Dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k == "pixel_values" and v.is_floating_point() and str(self.cfg.device).lower().startswith("cuda"):
                moved[k] = v.to(self.cfg.device, dtype=self.cfg.dtype)
            else:
                moved[k] = v.to(self.cfg.device)
        inputs = moved

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs.get("input_ids"),
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                num_beams=self.cfg.num_beams,
                use_cache=False,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=image.size,
        )
        return parsed

    def ocr_debug(self, screenshot_path: str) -> Dict[str, Any]:
        image = Image.open(screenshot_path).convert("RGB")
        task_prompt = "<OCR_WITH_REGION>"

        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
        moved: Dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k == "pixel_values" and v.is_floating_point() and str(self.cfg.device).lower().startswith("cuda"):
                moved[k] = v.to(self.cfg.device, dtype=self.cfg.dtype)
            else:
                moved[k] = v.to(self.cfg.device)
        inputs = moved

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs.get("input_ids"),
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                num_beams=self.cfg.num_beams,
                use_cache=False,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=image.size,
        )
        return {
            "generated_text": generated_text,
            "parsed": parsed,
        }

    def analyze_screen(
        self,
        screenshot_path: str,
        od_queries: Optional[Sequence[str]] = None,
        enable_ocr: bool = True,
    ) -> List[Dict[str, Any]]:
        image = Image.open(screenshot_path).convert("RGB")

        results: List[Dict[str, Any]] = []

        if od_queries:
            for q in od_queries:
                prompt = f"<OPEN_VOCABULARY_DETECTION> {q}"
                parsed = self._run_task(image=image, task_prompt=prompt)
                items = _extract_items(parsed)
                for it in items:
                    it.setdefault("type", "od")
                    it.setdefault("query", str(q))
                results.extend(items)

        if enable_ocr:
            prompt = "<OCR_WITH_REGION>"
            parsed = self._run_task(image=image, task_prompt=prompt)
            items = _extract_items(parsed)
            for it in items:
                it.setdefault("type", "ocr")
            results.extend(items)

        return results

    def analyze_screen_json(
        self,
        screenshot_path: str,
        od_queries: Optional[Sequence[str]] = None,
        enable_ocr: bool = True,
        ensure_ascii: bool = False,
    ) -> str:
        data = self.analyze_screen(
            screenshot_path=screenshot_path,
            od_queries=od_queries,
            enable_ocr=enable_ocr,
        )
        return json.dumps(data, ensure_ascii=ensure_ascii)
