"""Dashboard backend (runs on the laptop).

Serves the HTML UI, holds the Pi credentials so the browser never sees the
bearer token, proxies the MJPEG stream and event WebSocket, and archives
match crops to dashboard/captures/<name>/.

  python dashboard/app.py
"""
import asyncio
import json
from pathlib import Path

import httpx
from fastapi import (
    FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from pi_client import PiClient

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
CAPTURES_DIR = ROOT / "captures"
CONFIG_PATH = ROOT / "config.json"

CAPTURES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="face-recog dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

pi: PiClient | None = None
event_subscribers: set[asyncio.Queue] = set()


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_config(data: dict):
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


async def fan_out_event(payload: dict):
    for q in list(event_subscribers):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


@app.on_event("startup")
async def auto_reconnect():
    cfg = load_config()
    if cfg.get("pi_url") and cfg.get("token"):
        try:
            await _connect(cfg["pi_url"], cfg["token"], persist=False)
            print(f"[dashboard] auto-reconnected to {cfg['pi_url']}")
        except Exception as e:
            print(f"[dashboard] auto-reconnect failed: {e}")


async def _connect(pi_url: str, token: str, persist: bool = True):
    global pi
    if pi is not None:
        await pi.stop()
        pi = None
    client = PiClient(pi_url=pi_url, token=token,
                      captures_dir=CAPTURES_DIR,
                      event_callback=fan_out_event)
    info = await client.handshake()
    client.start_event_subscriber()
    pi = client
    if persist:
        save_config({"pi_url": pi_url, "token": token})
    return info


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/connect")
async def api_connect(payload: dict):
    pi_url = (payload.get("pi_url") or "").strip()
    token = (payload.get("token") or "").strip()
    if not pi_url or not token:
        raise HTTPException(status_code=400, detail="pi_url and token required")
    try:
        info = await _connect(pi_url, token)
    except PermissionError:
        raise HTTPException(status_code=401, detail="invalid token")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"pi unreachable: {e!r}")
    return info


@app.post("/api/disconnect")
async def api_disconnect():
    global pi
    if pi is not None:
        await pi.stop()
        pi = None
    return {"ok": True}


@app.get("/api/status")
async def api_status():
    if pi is None:
        return {"connected": False}
    return {
        "connected": pi.connected,
        "pi_url": pi.pi_url,
        "server_info": pi.server_info,
    }


@app.get("/api/stream")
async def api_stream():
    if pi is None or pi.server_info is None:
        raise HTTPException(status_code=503, detail="not connected")

    async def gen():
        try:
            async for chunk in pi.stream_mjpeg():
                yield chunk
        except Exception:
            return

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/api/events")
async def api_events(ws: WebSocket):
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


@app.get("/api/captures")
async def api_captures():
    out: dict[str, list[dict]] = {}
    for person_dir in sorted(CAPTURES_DIR.iterdir() if CAPTURES_DIR.exists() else []):
        if not person_dir.is_dir():
            continue
        items = []
        for p in sorted(person_dir.iterdir(),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append({
                    "file": p.name,
                    "ts": p.stat().st_mtime,
                    "url": f"/api/captures/{person_dir.name}/{p.name}",
                })
        out[person_dir.name] = items
    return out


@app.get("/api/captures/{name}/{file}")
async def api_capture_file(name: str, file: str):
    if "/" in file or ".." in file or ".." in name:
        raise HTTPException(status_code=400)
    path = CAPTURES_DIR / name / file
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/enrolled")
async def api_enrolled():
    if pi is None:
        raise HTTPException(status_code=503, detail="not connected")
    return await pi.list_enrolled()


@app.post("/api/enroll")
async def api_enroll(
    name: str = Form(...),
    photos: list[UploadFile] = File(...),
):
    if pi is None:
        raise HTTPException(status_code=503, detail="not connected")
    files = []
    for f in photos:
        files.append((f.filename or "photo.jpg", await f.read(),
                      f.content_type or "image/jpeg"))
    try:
        result = await pi.enroll(name=name, files=files)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"pi error: {e!r}")
    return result


@app.delete("/api/enrolled/{name}")
async def api_remove_enrolled(name: str):
    if pi is None:
        raise HTTPException(status_code=503, detail="not connected")
    return await pi.remove_enrolled(name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, log_level="info")
