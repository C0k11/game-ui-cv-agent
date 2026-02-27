import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from action.adb import AdbConfig, AdbDevice
from brain.openai_client import OpenAIChatClient, OpenAIChatConfig
from brain.prompting import build_messages
from vision.florence_vision import FlorenceConfig, FlorenceVision


def _center(bbox: Sequence[int]) -> Sequence[int]:
    x1, y1, x2, y2 = bbox
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


@dataclass
class AgentConfig:
    screenshot_path: str = "./_screen.png"
    od_queries: Optional[Sequence[str]] = None
    step_sleep_s: float = 0.6


class GameAgent:
    def __init__(
        self,
        vision: FlorenceVision,
        llm: OpenAIChatClient,
        device: AdbDevice,
        cfg: Optional[AgentConfig] = None,
    ):
        self.vision = vision
        self.llm = llm
        self.device = device
        self.cfg = cfg or AgentConfig()

    def step(self) -> Dict[str, Any]:
        self.device.screenshot(self.cfg.screenshot_path)
        items = self.vision.analyze_screen(
            screenshot_path=self.cfg.screenshot_path,
            od_queries=self.cfg.od_queries,
            enable_ocr=True,
        )
        messages = build_messages(items)
        action = self.llm.chat(messages)
        self._execute(action, items)
        return action

    def _execute(self, action: Dict[str, Any], items: Sequence[Dict[str, Any]]) -> None:
        a = str(action.get("action", "")).lower().strip()
        if a == "wait":
            ms = int(action.get("duration_ms", 800))
            time.sleep(ms / 1000.0)
            return

        if a == "click":
            target = action.get("target")
            if isinstance(target, (list, tuple)) and len(target) == 2:
                self.device.tap(int(target[0]), int(target[1]))
                time.sleep(self.cfg.step_sleep_s)
                return

            bbox = action.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y = _center([int(x) for x in bbox])
                self.device.tap(int(x), int(y))
                time.sleep(self.cfg.step_sleep_s)
                return

            raise ValueError("click action requires target [x,y] or bbox [x1,y1,x2,y2]")

        if a == "swipe":
            p1 = action.get("from")
            p2 = action.get("to")
            d = int(action.get("duration_ms", 500))
            if not (isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)) and len(p1) == 2 and len(p2) == 2):
                raise ValueError("swipe action requires from/to")
            self.device.swipe(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), duration_ms=d)
            time.sleep(self.cfg.step_sleep_s)
            return

        if a == "back":
            self.device.back()
            time.sleep(self.cfg.step_sleep_s)
            return

        raise ValueError(f"unknown action: {a}")


def build_default_agent(
    *,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: Optional[str],
    adb_serial: Optional[str],
    hf_cache_dir: str,
    od_queries: Optional[Sequence[str]],
) -> GameAgent:
    os.environ.setdefault("HF_HOME", hf_cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache_dir)

    vision = FlorenceVision(FlorenceConfig(cache_dir=hf_cache_dir))
    llm = OpenAIChatClient(OpenAIChatConfig(base_url=llm_base_url, model=llm_model, api_key=llm_api_key))
    device = AdbDevice(AdbConfig(serial=adb_serial))
    agent = GameAgent(vision=vision, llm=llm, device=device, cfg=AgentConfig(od_queries=od_queries))
    return agent
