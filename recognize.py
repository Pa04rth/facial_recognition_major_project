"""Live face recognition (Phase 1: laptop webcam, local display).

For headless/server use on the Pi, see pi_server.py.

Run:
  python recognize.py
Quit:
  q or Esc
"""
import csv
import time
from datetime import datetime
import cv2

import config
from db import EmbeddingDB
from engine import Engine


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
    print(f"Loaded {len(db.names)} enrolled embedding(s): {sorted(set(db.names))}")

    engine = Engine(db=db)

    log_path = config.LOGS_DIR / f"detections_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["timestamp", "name", "score", "x1", "y1", "x2", "y2", "crop_file"])

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

            result = engine.process(frame)
            now = time.time()

            for m in result.matches:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                crop_file = crops_dir / f"{m.name}_{ts}.jpg"
                cv2.imwrite(str(crop_file), m.crop_bgr)
                log.writerow([
                    datetime.now().isoformat(), m.name, f"{m.score:.4f}",
                    *m.bbox, crop_file.name,
                ])
                log_file.flush()
                print(f"[MATCH] {m.name} score={m.score:.3f}")

            fps_n += 1
            if now - fps_t0 >= 1.0:
                fps_display = fps_n / (now - fps_t0)
                fps_t0 = now
                fps_n = 0
            cv2.putText(result.annotated,
                        f"{fps_display:.1f} FPS  faces={result.n_faces}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("face-recog", result.annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
