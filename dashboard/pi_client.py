"""Client-side wrapper for the Pi server: handshake + MJPEG proxy + WS subscriber."""
import asyncio
import base64
import json
import time
from pathlib import Path

import httpx
import websockets


class PiClient:
    """Connects to a Pi server. Holds the token. Streams video and events.

    `event_callback` is called from the subscriber task with each decoded event
    dict from the Pi. The callback is responsible for archiving and rebroadcast.
    """

    def __init__(self, pi_url: str, token: str, captures_dir: Path,
                 event_callback):
        self.pi_url = pi_url.rstrip("/")
        self.token = token
        self.captures_dir = captures_dir
        self.event_callback = event_callback
        self.server_info: dict | None = None
        self._ws_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.connected = False

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    async def handshake(self) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{self.pi_url}/handshake", headers=self.headers)
        if r.status_code == 401:
            raise PermissionError("invalid token")
        r.raise_for_status()
        self.server_info = r.json()
        return self.server_info

    async def list_enrolled(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{self.pi_url}/enrolled", headers=self.headers)
        r.raise_for_status()
        return r.json()

    async def enroll(self, name: str, files: list[tuple[str, bytes, str]]) -> dict:
        """files: list of (filename, bytes, content_type)."""
        data = {"name": name}
        files_payload = [("photos", (fn, b, ct)) for fn, b, ct in files]
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(f"{self.pi_url}/enroll",
                             headers=self.headers,
                             data=data, files=files_payload)
        r.raise_for_status()
        return r.json()

    async def remove_enrolled(self, name: str) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.delete(f"{self.pi_url}/enrolled/{name}",
                               headers=self.headers)
        r.raise_for_status()
        return r.json()

    async def stream_mjpeg(self):
        """Async generator yielding raw bytes from Pi's MJPEG stream."""
        url = f"{self.pi_url}{self.server_info['stream_path']}"
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("GET", url, headers=self.headers) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    yield chunk

    def start_event_subscriber(self):
        if self._ws_task is None or self._ws_task.done():
            self._stop.clear()
            self._ws_task = asyncio.create_task(self._ws_loop())

    async def stop(self):
        self._stop.set()
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
        self.connected = False

    async def _ws_loop(self):
        events_path = self.server_info["events_path"]
        scheme = "wss" if self.pi_url.startswith("https") else "ws"
        host = self.pi_url.split("://", 1)[1]
        url = f"{scheme}://{host}{events_path}?token={self.token}"

        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self.connected = True
                    backoff = 1.0
                    async for raw in ws:
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        await self._archive_crop(payload)
                        try:
                            await self.event_callback(payload)
                        except Exception as e:
                            print(f"[pi_client] event_callback error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                print(f"[pi_client] WS error: {e!r}, reconnecting in {backoff:.0f}s")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                    break
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 30.0)

    async def _archive_crop(self, payload: dict):
        name = payload.get("name", "unknown")
        b64 = payload.get("crop_b64") or ""
        if not b64:
            return
        try:
            raw = base64.b64decode(b64)
        except Exception:
            return
        person_dir = self.captures_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)
        ts = int(payload.get("ts", time.time()) * 1000)
        out = person_dir / f"{ts}.jpg"
        out.write_bytes(raw)
        payload["archive_path"] = f"{name}/{out.name}"
