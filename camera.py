"""Camera abstraction.

Picks the right backend at runtime:
- Linux + picamera2 available  -> Raspberry Pi camera (Camera Module 3 on Pi 5).
- Windows                       -> cv2.VideoCapture with DSHOW.
- Other                         -> cv2.VideoCapture default.

All backends expose a cv2-style .read() -> (ok, frame_bgr) and .release().
"""
import sys
import time
import numpy as np


def _picamera2_available() -> bool:
    if sys.platform != "linux":
        return False
    try:
        import picamera2  # noqa: F401
        return True
    except ImportError:
        return False


class _CV2Camera:
    def __init__(self, index: int):
        import cv2
        if sys.platform == "win32":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        self._cap = cap
        self.backend = "cv2"

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


class _PiCamera2:
    def __init__(self, size=(1280, 720)):
        from picamera2 import Picamera2
        import cv2  # noqa: F401  (forces import before we capture)
        self._cv2 = __import__("cv2")
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={"size": size, "format": "RGB888"}
        )
        self._cam.configure(cfg)
        self._cam.start()
        time.sleep(0.4)  # let AE/AWB settle
        self.backend = "picamera2"

    def read(self):
        arr = self._cam.capture_array()
        if arr is None:
            return False, None
        return True, self._cv2.cvtColor(arr, self._cv2.COLOR_RGB2BGR)

    def release(self):
        try:
            self._cam.stop()
        finally:
            self._cam.close()


def open_camera(index: int = 0):
    if _picamera2_available():
        return _PiCamera2()
    return _CV2Camera(index)
