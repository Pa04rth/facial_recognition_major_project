# Drone-Based Face Recognition for Search & Rescue

A real-time, multi-face recognition system designed to identify missing
persons or persons-of-interest from a Raspberry Pi 5 + Camera Module 3
mounted on a drone. Built with InsightFace (ArcFace embeddings) over an
SCRFD detector, running on ONNX Runtime, with a browser dashboard that
connects to the Pi over a public tunnel.

This evolved from a college-gate LBPH project. The use case changed (gate →
drone over a crowd / disaster zone), so the algorithm and architecture
changed with it.

---

## Why not LBPH (the original approach)?

LBPH (Local Binary Pattern Histograms) was adequate at a college gate where
faces are roughly frontal, well-lit, and ~1 m from the camera. For a drone:

- Faces are small, off-angle, partially occluded, and motion-blurred.
- Lighting varies wildly (open sky, dust, shadows, smoke).
- LBPH's histogram comparison degrades past ~15° pose deviation.
- It does not scale: matching against N missing persons means N model
  trainings or one large model with poor inter-class separation.

The fix: **deep-learning face embeddings**. Each detected face is mapped to
a 512-dimensional vector by ArcFace. Identifying a person is then just a
cosine similarity lookup against a database of enrolled embeddings. This
handles pose and lighting far better and scales to thousands of identities.

---

## System Architecture

```
+--------------------+      Cloudflare Tunnel       +-------------------+
|     Pi 5           | <--------------------------> |     Laptop        |
|                    |  https://*.trycloudflare.com |                   |
|  pi_server.py      |                              |  dashboard/app.py |
|  - inference loop  |  GET  /handshake (auth)      |  - handshake      |
|  - FastAPI/uvicorn |  GET  /video.mjpg            |  - MJPEG proxy    |
|  - bearer-token    |  WS   /events                |  - WS client      |
|  - hot-enroll DB   |  POST /enroll  (upload)      |  - crop archive   |
|                    |  GET  /enrolled              |  - serves HTML/JS |
|  + cloudflared     |  DEL  /enrolled/{name}       |        |          |
|    (separate)      |                              |        v          |
+--------------------+                              |  Browser UI       |
                                                    +-------------------+
```

The inference pipeline inside `pi_server.py`:

```
[ Camera ] → [ SCRFD detector + landmarks ] → [ align 112x112 ]
           → [ ArcFace embedder (512-d) ]   → [ cosine match vs DB ]
           → [ per-name cooldown ]           → [ annotated MJPEG + event ]
```

Multiple simultaneous faces are handled by design: the detector returns N
boxes per frame, and embedding + matching is run independently for each.
Adding a new missing person is "upload photos via the dashboard" — no
retraining, no restart.

**Why the dashboard has a backend.** A browser `<img src=".../video.mjpg">`
cannot attach an `Authorization` header, so we can't point it directly at
an authenticated Pi endpoint. The dashboard backend on the laptop holds the
bearer token, proxies the MJPEG stream and event WebSocket, and writes crop
archives to disk. The browser never sees the Pi token.

---

## Technical Details

| Component         | Choice                          | Why                                                  |
|-------------------|---------------------------------|------------------------------------------------------|
| Detector          | SCRFD-500MF (via InsightFace)   | Strong on small/tilted faces; ships with model pack  |
| Recognizer        | MobileFaceNet (ArcFace, 512-d)  | Small enough for Pi 5 CPU; trained on 600k identities |
| Model pack        | `buffalo_sc`                    | Smallest InsightFace pack (~14 MB)                   |
| Inference runtime | ONNX Runtime (CPU)              | No CUDA, no PyTorch — clean Pi deployment            |
| Matching          | Brute-force cosine vs. matrix   | Optimal for 1-20 enrolled persons; no index overhead |
| Embedding storage | Pickled `(names, embeddings)`   | Trivial to inspect, rebuild, version                 |
| Hot-enroll safety | `threading.RLock` around DB     | HTTP and inference threads share the DB              |
| API framework     | FastAPI + uvicorn               | Async, MJPEG/WebSocket-friendly, light dependency    |
| Stream protocol   | `multipart/x-mixed-replace` MJPEG | Works in any browser via plain `<img>`; no codec   |
| Public tunnel     | Cloudflare Tunnel (or ngrok)    | No port-forwarding; HTTPS by default                 |
| Capture (laptop)  | `cv2.VideoCapture` (DSHOW)      | Standard Windows path                                |
| Capture (Pi 5)    | `picamera2` (libcamera)         | Required for Camera Module 3 on Pi 5 (Phase 2)       |

**Cosine-similarity threshold (`MATCH_THRESHOLD`):** start at `0.45`. Raise
toward `0.55` if you see false matches between similar-looking people; lower
toward `0.35` if real matches are being missed. ArcFace embeddings are
L2-normalized so the inner product *is* cosine similarity.

**Per-identity cooldown (`ALERT_COOLDOWN_SEC`, default 5 s):** prevents the
log from filling with one row per frame while a known face sits in view.

---

## Pi endpoints

All require `Authorization: Bearer <token>` except `/health`.

| Method | Path                 | Purpose                                                  |
|--------|----------------------|----------------------------------------------------------|
| GET    | `/health`            | Liveness probe (no auth).                                |
| GET    | `/handshake`         | `{server_id, version, capabilities, stream_path, events_path}` |
| GET    | `/video.mjpg`        | Annotated MJPEG stream.                                  |
| WS     | `/events?token=...`  | JSON match events as they fire.                          |
| GET    | `/enrolled`          | Current search list `[{name, n_photos}, ...]`.           |
| POST   | `/enroll`            | `multipart/form-data` (`name`, `photos`). Hot-merges.    |
| DELETE | `/enrolled/{name}`   | Remove person from DB and their `known_faces/<name>/`.   |
| GET    | `/crops`             | List of locally-saved crop files on the Pi.              |
| GET    | `/crops/{file}`      | A specific crop JPEG.                                    |

---

## Quickstart

Tested on Windows 11 with Python 3.12. The same scripts run on the Pi 5
(Phase 2 just swaps the camera capture line — see below).

### 1. Clone and set up the virtual environment

```powershell
git clone <your-fork-url> face_recog
cd face_recog
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r pi_requirements.txt
pip install httpx==0.27.2
```

`requirements.txt` is the inference stack (OpenCV, InsightFace, ONNX
Runtime). `pi_requirements.txt` adds the server stack (FastAPI, uvicorn,
WebSockets). `httpx` is the dashboard's Pi-facing HTTP client.

### 2. Set your auth token

Copy [.env.example](.env.example) to `.env` and edit `PI_AUTH_TOKEN` to any
string you want. This is the bearer token the dashboard will use to talk
to the Pi server.

```powershell
copy .env.example .env
notepad .env
```

`.env` is gitignored, so the token never gets committed.

### 3. Run the Pi server (Terminal 1)

```powershell
.venv\Scripts\Activate.ps1
python pi_server.py
```

uvicorn binds on `:8000`. The startup banner prints the `server_id` and
token. The first run downloads the InsightFace model pack (~14 MB) into
`~/.insightface/models/buffalo_sc/`.

### 4. Run the dashboard (Terminal 2)

```powershell
.venv\Scripts\Activate.ps1
python dashboard\app.py
```

Binds `127.0.0.1:5000`.

### 5. Open the dashboard in a browser

Go to <http://localhost:5000>. In the header, enter:

- **Pi URL:** `http://localhost:8000` (for local testing) — paste **without
  quotes**.
- **Token:** the value of `PI_AUTH_TOKEN` from your `.env` — paste exactly,
  no quotes.

Click **Connect**. You should see live annotated video in the main pane.

### 6. Enroll a person to search for

In the **Search List** section, type a name (e.g. `jane_doe`), pick 2–3
clear photos with `Choose photos…`, click **Upload & Enroll**. Within a
second the chip appears in the search list and the next time that person
enters the camera frame they'll be drawn with a green box and their name.

Crops of every match are saved to `dashboard\captures\<name>\` and shown in
the **Found People** gallery at the bottom of the page.

### 7. Make the Pi reachable from anywhere (optional)

In a third terminal — on the Pi or laptop, whichever is running
`pi_server.py`:

```powershell
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` prints `https://<random-words>.trycloudflare.com`. In the
dashboard, **Disconnect**, replace the Pi URL with that tunnel URL, keep
the same token, and **Connect** again. The flow is identical; only the
network path changes.

> `trycloudflare.com` URLs rotate on every `cloudflared` restart. For a
> stable URL, register a named tunnel (free with a Cloudflare account + a
> domain) or use Tailscale.

---

## Local-only mode (no dashboard)

If you just want to see the annotated feed on the same machine without
running the server, the original CLI still works:

```powershell
python enroll.py        # one-time: bulk-enroll from known_faces\<name>\*.jpg
python recognize.py     # webcam + cv2.imshow window
```

`recognize.py` and `pi_server.py` share the same `Engine` class — same
detector, same recognizer, same DB.

---

## Project Layout

```
face_recog\
  README.md
  .env.example          # copy to .env and edit PI_AUTH_TOKEN
  .gitignore            # excludes .env, embeddings.pkl, captures/, logs/
  requirements.txt      # inference stack
  pi_requirements.txt   # server stack (FastAPI, uvicorn, websockets)

  config.py             # thresholds, paths, model name
  db.py                 # EmbeddingDB (thread-safe via RLock)
  enroll.py             # CLI bulk-enroll + reusable embed_image() helper
  engine.py             # Engine: detect → match → annotate, shared core
  recognize.py          # local webcam loop (cv2.imshow)
  pi_server.py          # FastAPI server (Pi side) — uses Engine

  known_faces\          # input: <person_name>\*.jpg (or upload via UI)
  logs\
    detections_*.csv    # one row per matched detection (cooldown applied)
    crops\              # face crops at match time

  dashboard\
    app.py              # FastAPI on laptop, proxy + crop archiver
    pi_client.py        # handshake, MJPEG stream, WS subscriber
    requirements.txt    # dashboard-specific (just for portability)
    static\
      index.html
      app.js
      style.css
    captures\           # archived crops, <name>\<ms_timestamp>.jpg
    config.json         # auto-saved: last pi_url + token
```

---

## Configuration

All tunable in [config.py](config.py):

| Setting              | Default      | Effect                                                |
|----------------------|--------------|-------------------------------------------------------|
| `MATCH_THRESHOLD`    | `0.45`       | Cosine similarity needed to declare a match           |
| `ALERT_COOLDOWN_SEC` | `5.0`        | Minimum seconds between repeat alerts per person      |
| `CAMERA_INDEX`       | `0`          | Webcam index for `cv2.VideoCapture`                   |
| `FRAME_SKIP`         | `1`          | Process every Nth frame (raise on slow hardware)      |
| `INSIGHTFACE_MODEL`  | `buffalo_sc` | Model pack: `buffalo_sc` / `buffalo_s` / `buffalo_l`  |
| `DET_SIZE`           | `(640, 640)` | Detector input resolution                             |
| `DET_THRESHOLD`      | `0.5`        | Detection confidence floor                            |
| `MIN_FACE_SIZE`      | `30` px      | Drop boxes smaller than this on the shorter side      |

Server-side environment vars (set in [.env](.env.example)):

| Var               | Default        | Effect                                            |
|-------------------|----------------|---------------------------------------------------|
| `PI_AUTH_TOKEN`   | random hex     | Bearer token required for all auth'd endpoints    |
| `PI_HOST`         | `0.0.0.0`      | Bind address for uvicorn                          |
| `PI_PORT`         | `8000`         | Bind port                                         |
| `PI_SERVER_ID`    | `pi-<random>`  | Friendly identifier shown in the dashboard header |

---

## Phase 2 — Raspberry Pi 5 port

Only the camera-capture line in `pi_server.py` needs to change. On Pi 5
with Camera Module 3, replace `cv2.VideoCapture(...)` with a `picamera2`
capture loop. The rest of the code (Engine, DB, FastAPI server, dashboard)
is unchanged.

- Keep `buffalo_sc`; expect ~5–10 FPS at 640×640 on Pi 5 CPU.
- Tune `FRAME_SKIP` and/or `DET_SIZE` to hit the latency budget.
- Add active cooling — sustained inference pushes the SoC near 80% CPU.
- Install `cloudflared` from the Cloudflare APT repo and run it as a
  systemd unit so the tunnel comes up on boot.

## Phase 3 — Drone integration

- Pull GPS / altitude / heading from the flight controller over MAVLink
  (`pymavlink` / DroneKit). Tag every match event with `(lat, lon, alt, hdg)`
  before pushing to the WebSocket.
- Optionally publish a MAVLink `STATUSTEXT` when a match fires so a
  ground-station operator gets pinged in QGroundControl / Mission Planner.
- Power budget: ~5–7 W for Pi 5 + Camera Module 3. Plan thermals and
  battery draw before strapping it to the airframe.

---

## Troubleshooting

| Symptom                                       | Likely cause / fix                                                  |
|-----------------------------------------------|---------------------------------------------------------------------|
| Dashboard says "invalid token"                | Token mismatch — copy the value from `.env` exactly, no quotes.     |
| Dashboard says "pi unreachable"               | Pi server crashed (check Terminal 1) or URL wrong.                  |
| Video pane stays blank after Connect          | Browser may have cached a failed stream; hit refresh.               |
| Upload says "no face found" for a good photo  | Face too small or angled. Use a clearer frontal shot.               |
| All known faces show as "Unknown"             | `MATCH_THRESHOLD` is too high; try `0.35` and re-tune up.           |
| Two different people get the same name        | `MATCH_THRESHOLD` is too low; raise toward `0.55`.                  |
| Low FPS                                       | Raise `FRAME_SKIP` to `2` or `3`; shrink `DET_SIZE` to `(480, 480)`. |
| Model download stalled on first run           | Re-run the script; InsightFace resumes from the partial download.   |
| `cv2.VideoCapture` fails on Pi 5              | Use `picamera2` instead — see Phase 2 notes.                        |
| Cloudflare URL stops working after a few hours| Free `trycloudflare.com` URLs are ephemeral; restart `cloudflared` or use a named tunnel. |

---

## Outputs

**Pi side** — `logs\detections_YYYYMMDD_HHMMSS.csv`, one row per match
event after cooldown:

```
timestamp, name, score, x1, y1, x2, y2, crop_file
2026-05-20T12:30:01.123, jane_doe, 0.6321, 412, 188, 530, 322, jane_doe_20260520_123001_123456.jpg
```

Plus `logs\crops\<name>_<timestamp>.jpg` — the cropped face at match time.

**Laptop side** — `dashboard\captures\<name>\<ms_timestamp>.jpg`. The
dashboard archives every match crop it receives over the WebSocket, so the
laptop has a durable copy even if the Pi loses its SD card.

---

## Tooling

Python 3.12 · OpenCV 4.10 · InsightFace 0.7.3 · ONNX Runtime 1.19 · NumPy
1.26 · FastAPI 0.115 · uvicorn 0.30 · httpx 0.27 · websockets 13.1 · python-dotenv
