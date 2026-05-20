# Drone-Based Face Recognition for Search & Rescue

A real-time, multi-face recognition system designed to identify missing persons
or persons-of-interest from a Raspberry Pi 5 + Camera Module 3 mounted on a
drone. Built with InsightFace (ArcFace embeddings) over an SCRFD detector,
running on ONNX Runtime.

This is an evolution of an earlier college-gate LBPH project. The use case
changed (gate → drone over a crowd / disaster zone), so the algorithm changed
with it.

---

## Why not LBPH (the original approach)?

LBPH (Local Binary Pattern Histograms) was adequate at a college gate where
faces are roughly frontal, well-lit, and ~1 m from the camera. For a drone:

- Faces are small, off-angle, partially occluded, and motion-blurred.
- Lighting varies wildly (open sky, dust, shadows, smoke).
- LBPH's histogram comparison degrades past ~15° pose deviation.
- It does not scale: matching against N missing persons means N model trainings
  or one large model with poor inter-class separation.

The fix: **deep-learning face embeddings**. Each detected face is mapped to a
512-dimensional vector by ArcFace. Identifying a person is then just a cosine
similarity lookup against a database of enrolled embeddings. This handles pose
and lighting far better and scales to thousands of identities.

---

## System Design

```
                       [ Camera Module 3 (Pi) / webcam (laptop) ]
                                       |
                                       v
                              +------------------+
                              |    Frame grab    |
                              +------------------+
                                       |
                                       v
                              +------------------+
                              |  Face detector   |   SCRFD-500MF (bundled)
                              |  + 5-pt landmarks|   swappable: YOLOv8n-face
                              +------------------+
                                       |  N bounding boxes per frame
                                       v
                              +------------------+
                              |  Face aligner    |   warp each face to 112x112
                              +------------------+
                                       |
                                       v
                              +------------------+
                              |     Embedder     |   ArcFace (MobileFaceNet,
                              |  (512-dim vec)   |   w600k_mbf.onnx)
                              +------------------+
                                       |
                                       v
                              +------------------+
                              |     Matcher      |   cosine similarity vs.
                              |                  |   enrolled embeddings
                              +------------------+
                                       |
                       +---------------+---------------+
                       |                               |
                       v                               v
              +-----------------+              +-----------------+
              | Per-name cooldown|              | Annotated frame |
              | (no spammy alerts)|              | (boxes + label)|
              +-----------------+              +-----------------+
                       |                               |
                       v                               v
              +-----------------+              +-----------------+
              | CSV log + face  |              | Live preview /  |
              | crop on disk    |              | RTSP stream     |
              +-----------------+              +-----------------+
```

Multiple simultaneous faces are handled by design: the detector returns N
bounding boxes per frame, and embedding + matching is run independently for
each. There is no per-identity model retraining — adding a new missing person
is just "drop more photos in a folder and re-run `enroll.py`."

---

## Technical Details

| Component         | Choice                          | Why                                                  |
|-------------------|---------------------------------|------------------------------------------------------|
| Detector          | SCRFD-500MF (via InsightFace)   | Strong on small/tilted faces; ships with the model pack |
| Recognizer        | MobileFaceNet (ArcFace, 512-d)  | Small enough for Pi 5 CPU; trained on 600k identities |
| Model pack        | `buffalo_sc`                    | Smallest InsightFace pack; ~14 MB                    |
| Inference runtime | ONNX Runtime (CPU)              | No CUDA, no PyTorch — clean Pi deployment            |
| Matching          | Brute-force cosine vs. matrix   | Optimal for 1-20 enrolled persons; no index overhead |
| Embedding storage | Pickled `(names, embeddings)`   | Trivial to inspect, rebuild, and version            |
| Capture (laptop)  | `cv2.VideoCapture` (DSHOW)      | Standard Windows path                                |
| Capture (Pi 5)    | `picamera2` (libcamera)         | Required for Camera Module 3 on Pi 5                 |
| Logging           | CSV + face crop JPEGs           | Trivially diffable; survives crashes; audit-friendly |

**Cosine-similarity threshold (`MATCH_THRESHOLD`):** start at `0.45`. Raise to
~`0.55` if you see false matches between similar-looking people; lower toward
`0.35` if real matches are being missed. ArcFace embeddings are L2-normalized
so the inner product *is* cosine similarity, bounded in `[-1, 1]`.

**Per-identity cooldown (`ALERT_COOLDOWN_SEC`, default 5 s):** prevents the
log from filling with one row per frame while a known face sits in view.

---

## Why this detector instead of YOLOv8n-face?

YOLOv8n-face was the original plan. In practice, the SCRFD-500MF model bundled
inside InsightFace's `buffalo_sc` is competitive on small/tilted faces and
costs zero extra integration effort (no separate weights file, no separate
runtime, landmarks already produced for alignment).

If field tests on drone footage show SCRFD missing small faces at altitude,
swap to YOLOv8n-face — it's a localized change (a single `detect()` function),
not an architectural rewrite. The downstream alignment + embedding pipeline is
unchanged.

---

## Phased Roadmap

**Phase 1 — Laptop prototype (current).** Webcam capture, enrollment, live
recognition, CSV + crop logging. Use this to tune `MATCH_THRESHOLD`, sanity-
check accuracy, and validate end-to-end flow.

**Phase 2 — Raspberry Pi 5 port.**
- Replace `cv2.VideoCapture` with `picamera2` (libcamera; required for Camera
  Module 3 on Pi 5).
- Keep `buffalo_sc`; expect ~5-10 FPS at 640×640 on Pi 5 CPU.
- Tune `FRAME_SKIP` and/or `DET_SIZE` to hit the latency budget.
- Add active cooling — sustained inference pushes the SoC near 80% CPU.
- Run headless with the log file as the source of truth; live preview optional.

**Phase 3 — Drone integration.**
- Pull GPS / altitude / heading from the flight controller over MAVLink
  (`pymavlink` or DroneKit). Tag every match row with `(lat, lon, alt, hdg)`.
- Stream annotated frames to a ground station over RTSP (e.g. `mediamtx`,
  `gst-launch-1.0`, or `picamera2`'s GStreamer sink).
- Optionally publish a MAVLink `STATUSTEXT` or custom message when a match
  fires, so a ground-station operator gets pinged in QGroundControl / Mission
  Planner.
- Power budget: ~5-7 W for Pi 5 + Camera Module 3. Plan thermals and battery
  draw before strapping it to the airframe.

---

## Quickstart (Phase 1, laptop)

Requirements: Windows + Python 3.12 (already installed; venv is pre-built).

```powershell
cd a:\face_recog
.venv\Scripts\Activate.ps1
```

**1. Add reference photos.** One subfolder per person. Use clear, well-lit,
mostly-frontal shots. Multiple photos per person improves robustness.

```
known_faces\
  alice\
    alice_1.jpg
    alice_2.jpg
  bob\
    bob_1.jpg
```

**2. Build the embedding database.**

```powershell
python enroll.py
```

Each photo is processed: detector finds the largest face, aligner warps it,
embedder produces a 512-d vector, and the vector is stored under the person's
folder name. Output: `embeddings.pkl`.

**3. Run live recognition.**

```powershell
python recognize.py
```

Press `q` or `Esc` to quit. Matches are printed, drawn as green boxes, logged
to `logs\detections_*.csv`, and the corresponding face crops are saved to
`logs\crops\`. Unknown faces get red boxes and the per-frame similarity score
(useful for threshold tuning).

---

## Project Layout

```
a:\face_recog\
  config.py             # thresholds, paths, model name
  db.py                 # EmbeddingDB: add / save / load / match
  enroll.py             # build embeddings.pkl from known_faces\
  recognize.py          # live webcam loop, matching, logging
  requirements.txt      # pinned dependencies
  known_faces\          # input: <person_name>\*.jpg
  logs\
    detections_*.csv    # one row per matched detection
    crops\              # face crops saved at match time
  embeddings.pkl        # generated by enroll.py
  .venv\                # Python 3.12 venv with deps installed
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

---

## Outputs

**`logs\detections_YYYYMMDD_HHMMSS.csv`** — one row per match event after
cooldown:

```
timestamp, name, score, x1, y1, x2, y2, crop_file
2026-05-20T12:30:01.123, alice, 0.6321, 412, 188, 530, 322, alice_20260520_123001_123456.jpg
```

**`logs\crops\<name>_<timestamp>.jpg`** — the cropped face at the moment of
the match. Useful for spot-checking false positives.

---

## Troubleshooting

| Symptom                                         | Likely cause / fix                                                  |
|-------------------------------------------------|---------------------------------------------------------------------|
| `Embedding DB is empty`                         | Run `python enroll.py` first; check `known_faces\` layout           |
| `no face found in <img>` during enroll          | Photo is too low-res, too dark, or face too oblique — replace it    |
| All known faces show as "Unknown"               | `MATCH_THRESHOLD` is too high; try `0.35` and re-tune up            |
| Two different people get the same name          | `MATCH_THRESHOLD` is too low; raise toward `0.55`                   |
| Low FPS on laptop                               | Raise `FRAME_SKIP` to `2` or `3`; shrink `DET_SIZE` to `(480, 480)` |
| Model download stalled on first run             | Re-run the script; InsightFace resumes from the partial download    |
| `cv2.VideoCapture` fails on Pi 5                | Use `picamera2` instead — see Phase 2 notes                         |

---

## Tooling

Python 3.12 · OpenCV 4.10 · InsightFace 0.7.3 · ONNX Runtime 1.19 · NumPy 1.26
