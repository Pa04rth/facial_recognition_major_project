"""Enroll known faces.

Layout:
  known_faces/
    alice/  alice_1.jpg, alice_2.jpg, ...
    bob/    bob_1.jpg

Run:
  python enroll.py
"""
import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config
from db import EmbeddingDB

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def embed_image(app: FaceAnalysis, image_bgr: np.ndarray) -> np.ndarray | None:
    """Return the L2-normalized embedding for the largest face in the image, or None."""
    faces = app.get(image_bgr)
    if not faces:
        return None
    faces.sort(
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    return faces[0].normed_embedding


def build_face_analysis() -> FaceAnalysis:
    app = FaceAnalysis(
        name=config.INSIGHTFACE_MODEL,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=config.DET_SIZE, det_thresh=config.DET_THRESHOLD)
    return app


def main():
    config.KNOWN_FACES_DIR.mkdir(exist_ok=True)
    config.LOGS_DIR.mkdir(exist_ok=True)

    app = build_face_analysis()

    db = EmbeddingDB(config.DB_PATH)
    db.names = []
    db.embeddings = np.zeros((0, 512), dtype=np.float32)

    people = [p for p in config.KNOWN_FACES_DIR.iterdir() if p.is_dir()]
    if not people:
        print(f"No subfolders in {config.KNOWN_FACES_DIR}.")
        print("Create one subfolder per person and drop their photos inside, e.g.:")
        print("  known_faces/john_doe/photo1.jpg")
        sys.exit(1)

    total = 0
    for person_dir in sorted(people):
        name = person_dir.name
        n_added = 0
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ! cannot read {img_path}")
                continue
            emb = embed_image(app, img)
            if emb is None:
                print(f"  ! no face found in {img_path.name}")
                continue
            db.add(name, emb)
            n_added += 1
        print(f"  {name}: enrolled {n_added} image(s)")
        total += n_added

    if total == 0:
        print("No usable face images found. Check filenames and image quality.")
        sys.exit(1)

    db.save()
    print(f"\nSaved {len(db.names)} embedding(s) for {len(people)} person(s) -> {config.DB_PATH}")


if __name__ == "__main__":
    main()
