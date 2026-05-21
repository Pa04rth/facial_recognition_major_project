"""Inference core. Used by recognize.py (local debug) and pi_server.py (server)."""
import time
from dataclasses import dataclass
import cv2
import numpy as np

import config
from db import EmbeddingDB
from enroll import build_face_analysis, embed_image


@dataclass
class Match:
    name: str
    score: float
    bbox: tuple[int, int, int, int]
    crop_bgr: np.ndarray
    ts: float


@dataclass
class ProcessedFrame:
    annotated: np.ndarray
    matches: list[Match]
    n_faces: int


class Engine:
    """Detect, recognize, annotate. Owns the FaceAnalysis pipeline and the DB."""

    def __init__(self, db: EmbeddingDB | None = None):
        self.app = build_face_analysis()
        self.db = db if db is not None else EmbeddingDB(config.DB_PATH)
        self._last_alert: dict[str, float] = {}

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        faces = self.app.get(frame)
        now = time.time()
        matches: list[Match] = []
        out = frame

        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            if min(x2 - x1, y2 - y1) < config.MIN_FACE_SIZE:
                continue

            name, score = self.db.match(f.normed_embedding, config.MATCH_THRESHOLD)
            label = f"{name} {score:.2f}" if name else f"Unknown {score:.2f}"
            color = (0, 200, 0) if name else (0, 0, 255)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name and (now - self._last_alert.get(name, 0) >= config.ALERT_COOLDOWN_SEC):
                self._last_alert[name] = now
                crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
                if crop.size > 0:
                    matches.append(Match(
                        name=name, score=score, bbox=(x1, y1, x2, y2),
                        crop_bgr=crop, ts=now,
                    ))

        return ProcessedFrame(annotated=out, matches=matches, n_faces=len(faces))

    def enroll_image(self, name: str, image_bgr: np.ndarray) -> bool:
        """Embed an image and add to the live DB. Returns True if a face was found."""
        emb = embed_image(self.app, image_bgr)
        if emb is None:
            return False
        self.db.add(name, emb)
        return True
