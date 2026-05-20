"""Live face recognition (Phase 1: laptop webcam).

Phase 2 swaps the camera open for picamera2 on Pi 5.
Phase 3 layers in MAVLink GPS tagging + RTSP streaming to ground station.

Run:
  python recognize.py
Quit:
  q or Esc
"""
import csv
import time
from datetime import datetime
import cv2
from insightface.app import FaceAnalysis

import config
from db import EmbeddingDB


def open_camera():
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {config.CAMERA_INDEX}")
    return cap


def main():
    config.LOGS_DIR.mkdir(exist_ok=True)
    crops_dir = config.LOGS_DIR / "crops"
    crops_dir.mkdir(exist_ok=True)

    db = EmbeddingDB(config.DB_PATH)
    if len(db.names) == 0:
        raise SystemExit("Embedding DB is empty. Run `python enroll.py` first.")
    print(f"Loaded {len(db.names)} enrolled embedding(s): "
          f"{sorted(set(db.names))}")

    app = FaceAnalysis(
        name=config.INSIGHTFACE_MODEL,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=config.DET_SIZE, det_thresh=config.DET_THRESHOLD)

    log_path = config.LOGS_DIR / f"detections_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["timestamp", "name", "score", "x1", "y1", "x2", "y2", "crop_file"])

    last_alert: dict[str, float] = {}
    cap = open_camera()
    frame_idx = 0
    fps_t0 = time.time()
    fps_n = 0
    fps_display = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame_idx += 1
            if frame_idx % config.FRAME_SKIP != 0:
                continue

            faces = app.get(frame)
            now = time.time()

            for f in faces:
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                if min(x2 - x1, y2 - y1) < config.MIN_FACE_SIZE:
                    continue

                name, score = db.match(f.normed_embedding, config.MATCH_THRESHOLD)
                label = f"{name} {score:.2f}" if name else f"Unknown {score:.2f}"
                color = (0, 200, 0) if name else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if name and (now - last_alert.get(name, 0) >= config.ALERT_COOLDOWN_SEC):
                    last_alert[name] = now
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    crop_file = crops_dir / f"{name}_{ts}.jpg"
                    if crop.size > 0:
                        cv2.imwrite(str(crop_file), crop)
                    log.writerow([
                        datetime.now().isoformat(), name, f"{score:.4f}",
                        x1, y1, x2, y2, crop_file.name,
                    ])
                    log_file.flush()
                    print(f"[MATCH] {name} score={score:.3f}")

            fps_n += 1
            if now - fps_t0 >= 1.0:
                fps_display = fps_n / (now - fps_t0)
                fps_t0 = now
                fps_n = 0
            cv2.putText(frame, f"{fps_display:.1f} FPS  faces={len(faces)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("face-recog", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
