"""FastAPI server for headless deployment (Pi 5 or laptop dev).

Runs the inference loop in a background thread; serves MJPEG video, a
WebSocket event stream, and enrollment endpoints over HTTP.

Token auth: bearer token from $PI_AUTH_TOKEN. If unset, a random one is
generated on startup and printed once.

  python pi_server.py
  cloudflared tunnel --url http://localhost:8000   # optional, public URL
"""
import asyncio
import base64
import csv
import os
import secrets
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import cv2
from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException, UploadFile,
    WebSocket, WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse
import numpy as np

import config
from db import EmbeddingDB
from engine import Engine

SERVER_ID = os.environ.get("PI_SERVER_ID") or f"pi-{uuid.uuid4().hex[:8]}"
VERSION = "0.1.0"
HOST = os.environ.get("PI_HOST", "0.0.0.0")
PORT = int(os.environ.get("PI_PORT", "8000"))

AUTH_TOKEN = os.environ.get("PI_AUTH_TOKEN") or secrets.token_hex(16)

CROPS_DIR = config.LOGS_DIR / "crops"


class LatestFrame:
    """Single-slot frame holder. Producer overwrites, consumers always get newest."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._event = threading.Event()
        self._seq = 0

    def set(self, jpeg: bytes):
        with self._lock:
            self._jpeg = jpeg
            self._seq += 1
            self._event.set()

    def wait_for_next(self, last_seq: int, timeout: float = 5.0) -> tuple[bytes, int] | None:
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._jpeg is not None and self._seq != last_seq:
                    return self._jpeg, self._seq
                self._event.clear()
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            self._event.wait(timeout=remaining)


latest_frame = LatestFrame()
event_subscribers: set[asyncio.Queue] = set()
main_loop: asyncio.AbstractEventLoop | None = None

engine_lock = threading.Lock()
engine: Engine | None = None
inference_stats = {"fps": 0.0, "frames": 0, "matches": 0, "started_at": time.time()}


def open_camera():
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {config.CAMERA_INDEX}")
    return cap


def broadcast_event(payload: dict):
    """Called from the inference thread; schedules send on the asyncio loop."""
    if main_loop is None:
        return
    for q in list(event_subscribers):
        try:
            main_loop.call_soon_threadsafe(q.put_nowait, payload)
        except RuntimeError:
            pass


def inference_loop():
    config.LOGS_DIR.mkdir(exist_ok=True)
    CROPS_DIR.mkdir(exist_ok=True)
    csv_path = config.LOGS_DIR / f"detections_{datetime.now():%Y%m%d_%H%M%S}.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "name", "score", "x1", "y1", "x2", "y2", "crop_file"])

    cap = open_camera()
    fps_t0 = time.time()
    fps_n = 0
    fps_display = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            with engine_lock:
                result = engine.process(frame)

            for m in result.matches:
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                crop_file = CROPS_DIR / f"{m.name}_{ts_str}.jpg"
                cv2.imwrite(str(crop_file), m.crop_bgr)
                csv_writer.writerow([
                    datetime.now().isoformat(), m.name, f"{m.score:.4f}",
                    *m.bbox, crop_file.name,
                ])
                csv_file.flush()
                inference_stats["matches"] += 1

                ok_enc, crop_jpg = cv2.imencode(".jpg", m.crop_bgr,
                                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                crop_b64 = base64.b64encode(crop_jpg.tobytes()).decode() if ok_enc else ""
                broadcast_event({
                    "ts": m.ts,
                    "iso": datetime.now().isoformat(),
                    "name": m.name,
                    "score": round(m.score, 4),
                    "bbox": list(m.bbox),
                    "crop_file": crop_file.name,
                    "crop_b64": crop_b64,
                })

            fps_n += 1
            inference_stats["frames"] += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps_display = fps_n / (now - fps_t0)
                fps_t0 = now
                fps_n = 0
                inference_stats["fps"] = round(fps_display, 1)

            cv2.putText(result.annotated,
                        f"{fps_display:.1f} FPS  faces={result.n_faces}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            ok_enc, jpg = cv2.imencode(".jpg", result.annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok_enc:
                latest_frame.set(jpg.tobytes())
    finally:
        cap.release()
        csv_file.close()


def require_token(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if not secrets.compare_digest(authorization[7:], AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="invalid token")
    return True


app = FastAPI(title="face-recog Pi server", version=VERSION)


@app.on_event("startup")
async def on_startup():
    global engine, main_loop
    main_loop = asyncio.get_running_loop()
    db = EmbeddingDB(config.DB_PATH)
    engine = Engine(db=db)
    threading.Thread(target=inference_loop, daemon=True, name="inference").start()
    print(f"\n{'=' * 60}")
    print(f"  server_id : {SERVER_ID}")
    print(f"  listening : http://{HOST}:{PORT}")
    print(f"  token     : {AUTH_TOKEN}")
    print(f"  enrolled  : {len(db.names)} embedding(s) -> {sorted(set(db.names)) or '[]'}")
    print(f"  tunnel    : run `cloudflared tunnel --url http://localhost:{PORT}` separately")
    print(f"{'=' * 60}\n")


@app.get("/health")
async def health():
    return {"ok": True, "server_id": SERVER_ID, "version": VERSION}


@app.get("/handshake")
async def handshake(_: bool = Depends(require_token)):
    return {
        "server_id": SERVER_ID,
        "version": VERSION,
        "capabilities": {
            "fps": inference_stats["fps"],
            "det_size": list(config.DET_SIZE),
            "model": config.INSIGHTFACE_MODEL,
            "match_threshold": config.MATCH_THRESHOLD,
        },
        "stream_path": "/video.mjpg",
        "events_path": "/events",
        "stats": inference_stats,
    }


def mjpeg_generator():
    last_seq = -1
    boundary = b"--frame\r\n"
    while True:
        got = latest_frame.wait_for_next(last_seq, timeout=10.0)
        if got is None:
            continue
        jpeg, last_seq = got
        yield boundary + b"Content-Type: image/jpeg\r\n" \
              + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() \
              + jpeg + b"\r\n"


@app.get("/video.mjpg")
async def video_stream(_: bool = Depends(require_token)):
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/events")
async def events_ws(ws: WebSocket):
    token = ws.query_params.get("token") or ws.headers.get("authorization", "")[7:]
    if not token or not secrets.compare_digest(token, AUTH_TOKEN):
        await ws.close(code=1008)
        return
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    event_subscribers.add(q)
    try:
        while True:
            payload = await q.get()
            await ws.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        event_subscribers.discard(q)


@app.get("/crops")
async def list_crops(_: bool = Depends(require_token)):
    out = []
    if CROPS_DIR.exists():
        for p in sorted(CROPS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                name = p.stem.split("_")[0]
                out.append({"name": name, "file": p.name,
                            "ts": p.stat().st_mtime, "size": p.stat().st_size})
    return out


@app.get("/crops/{filename}")
async def get_crop(filename: str, _: bool = Depends(require_token)):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="bad filename")
    path = CROPS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404)
    return Response(content=path.read_bytes(), media_type="image/jpeg")


@app.get("/enrolled")
async def list_enrolled(_: bool = Depends(require_token)):
    counts = engine.db.counts_by_name() if engine else {}
    return [{"name": k, "n_photos": v} for k, v in sorted(counts.items())]


@app.post("/enroll")
async def enroll(
    name: str = Form(...),
    photos: list[UploadFile] = File(...),
    _: bool = Depends(require_token),
):
    name = name.strip()
    if not name or any(c in name for c in r'/\:*?"<>|'):
        raise HTTPException(status_code=400, detail="invalid name")

    person_dir = config.KNOWN_FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped: list[str] = []
    for f in photos:
        data = await f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            skipped.append(f"{f.filename}: cannot decode")
            continue

        out_path = person_dir / f"{int(time.time() * 1000)}_{f.filename}"
        out_path.write_bytes(data)

        with engine_lock:
            ok = engine.enroll_image(name, img)
        if ok:
            added += 1
        else:
            skipped.append(f"{f.filename}: no face found")

    if added:
        engine.db.save()

    return {
        "name": name,
        "added": added,
        "skipped": skipped,
        "total_in_db": len(engine.db.names),
    }


@app.delete("/enrolled/{name}")
async def remove_enrolled(name: str, _: bool = Depends(require_token)):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid name")
    with engine_lock:
        removed = engine.db.remove(name)
        if removed:
            engine.db.save()
    person_dir = config.KNOWN_FACES_DIR / name
    if person_dir.exists() and person_dir.is_dir():
        shutil.rmtree(person_dir, ignore_errors=True)
    return {"name": name, "removed_embeddings": removed,
            "total_in_db": len(engine.db.names)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pi_server:app", host=HOST, port=PORT, log_level="info")
