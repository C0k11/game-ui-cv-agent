import os


HF_CACHE_DIR = os.environ.get("HF_HOME", r"D:\\Project\\ml_cache\\huggingface")
MODELS_DIR = os.environ.get("MODELS_DIR", r"D:\\Project\\ml_cache\\models")

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    os.path.join(MODELS_DIR, "Qwen2.5-14B-Instruct-AWQ"),
)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:14b-instruct")
LLM_API_KEY = os.environ.get("LLM_API_KEY")

LOCAL_VLM_MODEL = os.environ.get("LOCAL_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
LOCAL_VLM_MODELS_DIR = os.environ.get("LOCAL_VLM_MODELS_DIR", os.path.join(MODELS_DIR, "vlm"))
LOCAL_VLM_DEVICE = os.environ.get("LOCAL_VLM_DEVICE", "cuda")
LOCAL_VLM_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_VLM_MAX_NEW_TOKENS", "256"))

YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    os.path.join(MODELS_DIR, "yolo", "best.pt"),
)

ADB_SERIAL = os.environ.get("ADB_SERIAL")
