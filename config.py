from pathlib import Path

ROOT = Path(__file__).resolve().parent

KNOWN_FACES_DIR = ROOT / "known_faces"
LOGS_DIR = ROOT / "logs"
DB_PATH = ROOT / "embeddings.pkl"

# Cosine-similarity threshold for declaring a match (0..1, higher = stricter).
# ArcFace embeddings: ~0.40-0.55 is a typical operating range. Tune on your data.
MATCH_THRESHOLD = 0.45

# Don't re-alert on the same person more often than this (seconds).
ALERT_COOLDOWN_SEC = 5.0

CAMERA_INDEX = 0

# Process every Nth frame (1 = every frame). Bump on slow hardware.
FRAME_SKIP = 1

# InsightFace model pack. buffalo_sc is the smallest, fits Pi 5 comfortably.
# Options: buffalo_l (best accuracy, heavy), buffalo_s, buffalo_sc.
INSIGHTFACE_MODEL = "buffalo_sc"

# Detector input size. Smaller = faster, but worse at small/distant faces (drones).
DET_SIZE = (640, 640)
DET_THRESHOLD = 0.5

# Drop face boxes whose shorter side is below this (pixels).
MIN_FACE_SIZE = 30
