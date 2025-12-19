import os
from pathlib import Path

def load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                if key.strip() not in os.environ:
                    os.environ[key.strip()] = value.strip()

load_env()

PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MIC_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "0"))
CAMERA_DEVICE = os.getenv("CAMERA_DEVICE", "/dev/video0")
AUDIO_PLAYER = os.getenv("AUDIO_PLAYER", "paplay")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-flash")
MOVE_COMPENSATION = float(os.getenv("MOVE_COMPENSATION", "1.3"))
TURN_COMPENSATION = float(os.getenv("TURN_COMPENSATION", "2.2"))
DEFAULT_LINEAR_SPEED = float(os.getenv("DEFAULT_LINEAR_SPEED", "0.2"))
DEFAULT_ANGULAR_SPEED = float(os.getenv("DEFAULT_ANGULAR_SPEED", "0.8"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
MAX_RECORD_SECONDS = int(os.getenv("MAX_RECORD_SECONDS", "20"))
SILENCE_TIMEOUT = float(os.getenv("SILENCE_TIMEOUT", "1.5"))
ENERGY_THRESHOLD = float(os.getenv("ENERGY_THRESHOLD", "500"))
USE_LLM_INTENT = os.getenv("USE_LLM_INTENT", "1") == "1"
